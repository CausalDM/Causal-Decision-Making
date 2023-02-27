import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import glob
import re
import math
from torch.optim.adam import Adam


#========================================
# Data Generating Functions for Simulation
#========================================

def simulate_random_dag(d: int,
                        degree: float,
                        w_range: tuple = (1.0, 1.0)) -> nx.DiGraph:
    """Simulate random DAG with an expected degree by Erdos-Renyi model.
        
        Args:
        d: number of nodes
        degree: expected node degree, in + out
        w_range: weight range +/- (low, high)
        
        Returns:
        G: weighted DAG
        """
    prob = float(degree) / (d - 1)
    B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    
    # remove all in-edges (from precedent nodes) of the first node as A
    W[:, 0] = 0
    # remove all out-edges (from descendent nodes) of the last node as Y
    W[d-1, :] = 0
    # the remained nodes are the mediators M; and reset mediators if it has higher topological order than A or lower order than Y.
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(W)))
    j = 1
    while j < d - 1:
        if  ordered_vertices.index(j) < ordered_vertices.index(0):
            W[j, 1:(d - 1)] = np.zeros (d - 2)
        if  ordered_vertices.index(j) > ordered_vertices.index(d - 1):
            W[1:(d - 1), j] = np.zeros (d - 2)
        j = j + 1
    print("True weighted adjacency matrix B:\n", W)
    G = nx.DiGraph(W)
    calculate_effect(W)
    return G


def simulate_lsem(G: nx.DiGraph,
                 n: int, A_type: str,
                 x_dims: int = 1,
                 noise_scale: float = 0.5,
                 baseline: float = 1.0) -> np.ndarray:
    """Simulate samples from LSEM.
        
        Args:
        G: weigthed DAG
        n: number of samples
        A_type: the type of the exposure {Binary, Gaussian}
        x_dims: dimension of each node
        noise_scale: noise scale parameter of Gaussian distribution in the lSEM
        baseline: the baseline for the outcome
        
        Returns:
        X: [n, d] sample matrix
        """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    rank_A = ordered_vertices.index(0)
    for j in ordered_vertices:
        if ordered_vertices.index(j) > rank_A:
            parents = list(G.predecessors(j))
            X[:, j, 0] = X[:, parents, 0].dot(W[parents, j]) + np.random.normal(scale=noise_scale, size=n)
        elif ordered_vertices.index(j) < rank_A:
            X[:, j, 0] = np.random.normal(scale=noise_scale, size=n)
        else:
            if A_type == 'Binary':
                X[:, j, 0] = 2 * (np.random.binomial(1, 0.5, n) - 0.5)
            elif A_type == 'Gaussian':
                X[:, j, 0] = np.random.normal(scale=noise_scale, size=n)
            else:
                raise ValueError('unknown exposure type')
    X[:, d-1, 0] += baseline
    return X


#========================================
# Calculate Causal Effects in ANOCE Table
#========================================

def calculate_effect(predB):
    """Calculate causal effects in ANOCE based on estimated weighted adjacency matrix.
        
        Args:
        predB: estimated weighted adjacency matrix B
        d: number of nodes
        
        Returns:
        TE: total effect
        DE: natural direct effect
        IE: natural indirect effect
        DM: natural direct effect for mediators
        IM: natural indirect effect for mediators
        """
    # Number of nodes in the graph
    d = predB.shape[0]
    # Calculate causal effects in ANOCE
    alpha = predB[0, 1:(d - 1)]
    beta = predB[1:(d - 1), (d - 1)]
    DE = predB[0, (d - 1)] # natural direct effect DE
    
    trans_BM = predB[1:(d - 1), 1:(d - 1)]
    zeta = (np.dot(alpha, np.linalg.inv(np.identity(d - 2) - trans_BM))) # the causal effect of A on M
    IE = np.squeeze(np.dot(zeta, beta)) # natural indirect effect IE
    
    TE = DE + IE # total effect
    
    DM = np.multiply(beta.reshape((d - 2, 1)), zeta.reshape((1, d - 2))).diagonal() # natural direct effect for mediators
    
    eta = np.zeros((d - 2)) # the individual mediation effect in Chakrabortty et al. (2018)
    for i in range(1, (d - 1)):
        predB_1reduce = np.delete(predB, i, 0)
        predB_1reduce = np.delete(predB_1reduce, i, 1)
        alpha_R = predB_1reduce[0, 1:(d - 2)]
        beta_R = predB_1reduce[1:(d - 2), (d - 2)]
        trans_BM_R = predB_1reduce[1:(d - 2), 1:(d - 2)]
        zeta_R = (np.dot(alpha_R, np.linalg.inv(np.identity(d - 3) - trans_BM_R)))
        eta[i - 1] = IE - np.dot(zeta_R, beta_R)
    
    IM = eta - DM # natural indirect effect for mediators
    
#    print('The total effect (TE):', TE)
#    print('The natural direct effect (DE):', DE)
#    print('The natural indirect effect (IE):', IE)
#    print('The natural direct effect for mediators (DM):', DM)
#    print('The natural direct effect for mediators (IM):', IM)

    return TE, DE, IE, DM, IM

#========================================
# Calculate Constraints and Update Optimizer
#========================================

def fun_h1_B(B):
    '''compute constraint h1(B) value'''
    d = B.shape[0]
    expm_B = matrix_poly(B * B, d)
    h1_B = torch.trace(expm_B) - d
    return h1_B

def fun_h2_B(B):
    '''compute constraint h2(B) value'''
    d = B.shape[0]
    h2_B = sum(abs(B[:, 0]))+sum(abs(B[(d - 1), :]))-abs(B[(d - 1), 0])
    return h2_B
    
def update_optimizer(optimizer, old_lr, c_B, d_B):
    '''related LR to c_B and d_B, whenever c_B and d_B gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    
    estimated_lr = old_lr / (math.log10(c_B) + math.log10(d_B) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr
    
    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


#========================================
# VAE utility functions and modules
# Credit to DAG-GNN https://github.com/fishmoon1234/DAG-GNN
#========================================

_EPS = 1e-10
prox_plus = torch.nn.Threshold(0.,0.)

def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
        enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def kl_gaussian(preds):
    """compute the KL loss for Gaussian variables."""
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def nll_gaussian(preds, target, variance, add_const=False):
    """compute the loglikelihood of Gaussian variables."""
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def preprocess_adj(adj):
    """preprocess the adjacency matrix: adj_A = I-A^T."""
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_inv(adj):
    """preprocess the adjacency matrix: adj_A_inv = (I-A^T)^(-1)."""
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()
        
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor
        
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')
        
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        
        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj(adj_A1)
        
        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
        
        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class MLPDecoder(nn.Module):
    """MLP decoder module."""
    
    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()
        
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)
        
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        
        self.dropout_prob = do_prob
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_inv(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa
        
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)
        
        return mat_z, out, adj_A_tilt


def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph) -> tuple:
    """Compute FDR, TPR, and SHD for a matrix B.
        
        Args:
        G_true: ground truth graph
        G: predicted graph
        
        Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        shd: undirected extra + undirected missing + reverse
        """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    d = B.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, shd
