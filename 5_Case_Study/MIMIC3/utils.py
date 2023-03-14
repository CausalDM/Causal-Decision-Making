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
import seaborn as sn
import matplotlib.pyplot as plt

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
    U[np.random.rand(d, d) < 0.5] *= 1#-1
    W = (B_perm != 0).astype(float) * U
    
    # remove all in-edges (from precedent nodes) of the first node as A
    #W[:, 0] = 0
    # remove all out-edges (from descendent nodes) of the last node as Y
    W[:, d-1] = 0
    # the remained nodes are the mediators M; and reset mediators if it has higher topological order than A or lower order than Y.
#     ordered_vertices = list(nx.topological_sort(nx.DiGraph(W)))
#     j = 1
#     while j < d - 1:
# #         if  ordered_vertices.index(j) < ordered_vertices.index(0):
# #             W[j, 1:(d - 1)] = np.zeros (d - 2)
#         if  ordered_vertices.index(j) > ordered_vertices.index(d - 1):
#             W[1:(d - 1), j] = np.zeros (d - 2)
#         j = j + 1
#     print("True weighted adjacency matrix B:\n", W)
    G = nx.DiGraph(W)
#     calculate_effect(W)
    return W


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
#     rank_A = ordered_vertices.index(0)
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        X[:, j, 0] = X[:, parents, 0].dot(W[parents, j]) + np.random.binomial(1, 0.5, n)
#         if ordered_vertices.index(j) > rank_A:
#             parents = list(G.predecessors(j))
#             X[:, j, 0] = X[:, parents, 0].dot(W[parents, j]) + np.random.normal(scale=noise_scale, size=n)
#         elif ordered_vertices.index(j) < rank_A:
#             X[:, j, 0] = np.random.normal(scale=noise_scale, size=n)
#         else:
#             if A_type == 'Binary':
#                 X[:, j, 0] = 2 * (np.random.binomial(1, 0.5, n) - 0.5)
#             elif A_type == 'Gaussian':
#                 X[:, j, 0] = np.random.normal(scale=noise_scale, size=n)
#             else:
#                 raise ValueError('unknown exposure type')

    X[:, d-1, 0] += baseline
    return X


#========================================
# Calculate Causal Effects in ANOCE Table
#========================================

def calculate_effect(predB, ifprint=True):
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
    
    MAT = np.array(predB) 
    G = nx.from_numpy_matrix(np.array(MAT.T))
    FSIE_list = [] 

    for node in range(d-1):  
        FSIE = 0 
        for path in nx.all_simple_paths(G, source=node, target=d-1, cutoff = int(d/2)+1): 
            p_FSIE = 1 

            for i in range(len(path)-1): 
                if np.abs(MAT[path[i+1], path[i]]) > 0:
                    p_FSIE = p_FSIE * MAT[path[i+1], path[i]] 
                else:
                    p_FSIE = 0 
                    break 
            if i == len(path) - 2 and len(path) > 2:
                FSIE = FSIE + p_FSIE 

        FSIE_list.append(FSIE)  
        
    FSIE_list = np.array(FSIE_list)
    FSDE_list = MAT[-1,:(d-1)]
    FSTE_list = FSDE_list + FSIE_list
    
    if ifprint == True:
        print('The total effect (FSTE):', FSTE_list)
        print('The natural direct effect (FSDE):', FSDE_list)
        print('The natural indirect effect (FSIE):', FSIE_list)
    return FSTE_list, FSDE_list, FSIE_list

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
    h2_B =  sum(abs(B[(d - 1), :]))
#     for i in range(key_num):
#         h2_B =  h2_B + sum(abs(B[(key_num-1):(d - 1), i]))
#     #sum(abs(B[:, 0]))+sum(abs(B[(d - 1), :]))-abs(B[(d - 1), 0])
    return h2_B
    
def plot_mt(mt, labels_name=None, file_name=None, figsize=6):
    
    d = mt.shape[0]
    fig = plt.figure(figsize=(figsize,figsize))
    ax = fig.add_subplot(111)
    cax = ax.matshow(mt, cmap = 'RdBu', vmin = -0.5, vmax = 0.5)
    fig.colorbar(cax)
    
    xaxis = np.arange(d)
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    
    if labels_name ==None:
        ax.set_xticklabels(range(d), rotation=90)
        ax.set_yticklabels(range(d))
    else:
        ax.set_xticklabels(labels_name, rotation=90)
        ax.set_yticklabels(labels_name) 

    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    if labels_name != None:
        plt.savefig(file_name + '.pdf') # save as pdf
    plt.show() # display  
 
    
    
def plot_net(mt, labels_name=None, file_name=None):  
    # G= nx.from_numpy_matrix(np.array(est_mt))
    G = nx.DiGraph(mt.T) 
    d = mt.shape[0]
    weights = np.real([*nx.get_edge_attributes(G, 'weight').values()])
    pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels = True, edge_color = 'b', arrowsize=20, arrowstyle='fancy')   
    # labels_name = range(d)
    labels={}
    for i in range(d):
        if labels_name == None:
            labels[i]= i #labels_name[i]
        else:
            labels[i]= labels_name[i]

    nx.draw(G, pos, node_color='#A0CBE2', labels=labels, with_labels = True, #edge_color=weights, 
            node_size=4000, linewidths=0.25, font_size=8, 
            width=0.5, arrowstyle='->', arrows=True, arrowsize=15)
    
    if labels_name != None:
        plt.savefig(file_name + '.pdf') # save as pdf
    plt.show() # display  
    
    
#========================================
# Refit Causal Model
#========================================        
def refit(causal_feature, est_mt, labels):
    '''Plot the matrix B'''

    # Identification Constraint

    topo_list = np.array(labels)[list(nx.topological_sort(nx.DiGraph(est_mt)))].tolist()
    topo_list.reverse()
    topo_list.remove('Died within 48H') 
    topo_list.append('Died within 48H')          

    mt = np.zeros((len(labels),len(labels)))
    mt_sd = np.zeros((len(labels),len(labels)))
    for var_name in topo_list:
        if topo_list.index(var_name) > 0:
            
            ANC_list = topo_list[:topo_list.index(var_name)]
            
            Xmat = causal_feature[ANC_list]
            yval = causal_feature[var_name]

            dimp = np.shape(Xmat)[1] + 1  # plus one because LinearRegression adds an intercept term

            X_with_intercept = np.zeros((len(yval), dimp))
            X_with_intercept[:, 0] = 1
            X_with_intercept[:, 1:dimp] = Xmat

            beta = ((np.linalg.inv(X_with_intercept.T @ X_with_intercept + 1 * np.eye(dimp))).dot(X_with_intercept.T)).dot(yval)

            y_hat = X_with_intercept.dot(beta) #lreg.predict(Xmat)
            residuals = yval - y_hat
            residual_sum_of_squares = residuals.T @ residuals    

            sigma_squared_hat = residual_sum_of_squares / (len(yval) - dimp)
            var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept + 1 * np.eye(dimp)) * sigma_squared_hat
            sd_ = np.sqrt(np.diag(var_beta_hat)[1:])

            coef_ = beta[1:]#np.array(lreg.coef_)

            for i in range(len(ANC_list)):
                mt[labels.index(var_name),labels.index(ANC_list[i])] = coef_[i]
                mt_sd[labels.index(var_name),labels.index(ANC_list[i])] = sd_[i]    
    return mt, mt_sd    
    