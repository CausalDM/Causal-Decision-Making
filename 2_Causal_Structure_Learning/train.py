''''
    Main function for traininng ANOCE_CVAE
    
'''
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import random

import torch.optim as optim
from torch.optim import lr_scheduler
import math
from utils import *

from multiprocessing import Pool
import multiprocessing
n_cores = multiprocessing.cpu_count()
from numpy.random import randn
from random import seed as rseed
from numpy.random import seed as npseed

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()

#========================================
# Configurations
#========================================

# ----------- Data parameters ------------
parser.add_argument('--data_type', type = str, default = 'simulation',
                    choices = ['realdata','simulation', 'create_new'],
                    help = 'Choosing which experiment to do.')
parser.add_argument('--real_data_file', type = str, default = 'covid19.pkl',
                    help = 'The file containing the real data sample.')
parser.add_argument('--simu_G_file', type = str, default = 's1_trueG.pkl',
                    help = 'The file containing the true graph G to simulate sample.')
parser.add_argument('--graph_degree', type = int, default = 2,
                    help = 'The number of degree in generated DAG graph under erdos-renyi generation method')
parser.add_argument('--A_type', type = str, default = 'Binary',
                    help = 'The type of the exposure {Binary, Gaussian}.')
parser.add_argument('--sample_size', type = int, default = 500,
                    help = 'The number of samples of data.')
parser.add_argument('--node_number', type = int, default = 12,
                    help = 'The number of variables in data.')

# ----------- Training hyperparameters ------
parser.add_argument('--seed', type = int, default = 2333, help = 'Random seed.')
parser.add_argument('--rep_number', type = int, default = 100, help = 'The number of replication.')
parser.add_argument('--epochs', type = int, default = 200,
                    help ='Number of epochs to train.')
parser.add_argument('--batch_size', type = int, default = 100,
                    help = 'Number of samples per batch. note: should be divisible by sample size, otherwise throw an error.')
parser.add_argument('--k_max_iter', type = int, default = 1e2,
                    help = 'the max iteration number for searching parameters')
parser.add_argument('--original_lr', type = float, default = 3e-3, help = 'Initial learning rate.')

args = parser.parse_args()
print(args)

#========================================
# Main function
#========================================

def anoce_simu(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # ----------- Configurations:
    n = args.sample_size # The number of samples of data.
    d = args.node_number # The number of variables in data.
    x_dims = 1 # The number of input dimensions: default 1.
    z_dims = d # The number of latent variable dimensions: default the same as variable size.
    epochs = args.epochs # Number of epochs to train.
    batch_size = args.batch_size # Number of samples per batch. note: should be divisible by sample size, otherwise throw an error.
    k_max_iter = int(args.k_max_iter) # The max iteration number for searching parameters.
    original_lr = args.original_lr  # Initial learning rate.
    encoder_hidden = d^2 # Number of hidden units, adaptive to dimension of nodes (d^2).
    decoder_hidden = d^2 # Number of hidden units, adaptive to dimension of nodes (d^2).
    temp = 0.5 # Temperature for Gumbel softmax.
    factor = True # Factor graph model.
    encoder_dropout = 0.0 # Dropout rate (1 - keep probability).
    decoder_dropout = 0.0 # Dropout rate (1 - keep probability).
    tau_B = 0. # Coefficient for L-1 norm of matrix B.
    lambda1 = 0. # Coefficient for DAG constraint h1(B).
    lambda2 = 0. # Coefficient for identification constraint h2(B).
    c_B = 1 # Coefficient for absolute value h1(B).
    d_B = 1 # Coefficient for absolute value h2(B).
    h1_tol = 1e-8 # The tolerance of error of h1(B) to zero.
    h2_tol = 1e-8 # The tolerance of error of h2(B) to zero.
    lr_decay = 200 # After how many epochs to decay LR by a factor of gamma. 
    gamma = 1.0 # LR decay factor.  
    
    # ----------- Training:
    def train(epoch, lambda1, c_B, lambda2, d_B, optimizer, old_lr):
        
        nll_train = []
        kl_train = []
        mse_train = []
        encoder.train()
        decoder.train()
        scheduler.step()

        # Update optimizer
        optimizer, lr = update_optimizer(optimizer, old_lr, c_B, d_B)

        for batch_idx, (data, relations) in enumerate(train_loader):

            data, relations = Variable(data).double(), Variable(relations).double()
            relations = relations.unsqueeze(2) # Reshape data

            optimizer.zero_grad()

            enc_x, logits, origin_B, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send) 
            edges = logits # Logits is of size: [num_sims, z_dims]

            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, d * x_dims, rel_rec, rel_send, origin_B, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.
            
            # Compute constraint functions h1(B) and h2(B)
            h1_B = fun_h1_B(origin_B)
            h2_B = fun_h2_B(origin_B)

            # Reconstruction accuracy loss:
            loss_nll = nll_gaussian(preds, target, variance)
            # KL loss:
            loss_kl = kl_gaussian(logits)
            # ELBO loss:
            loss = loss_kl + loss_nll
            # Loss function:
            loss += lambda1 * h1_B + 0.5 * c_B * h1_B * h1_B + lambda2 * h2_B + 0.5 * d_B * h2_B * h2_B + 100. * torch.trace(origin_B * origin_B)

            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, tau_B * lr)

            if torch.sum(origin_B != origin_B):
                print('nan error\n')

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), origin_B, optimizer, lr

    
    # ----------- Generating data:
    if args.data_type == 'simulation':
        # Load true G
        with open(os.path.join('', args.simu_G_file), 'rb') as trueG:
            ground_truth_G = pickle.load(trueG)
        # Generate data
        X = simulate_lsem(ground_truth_G, args.sample_size, args.A_type, x_dims)
    elif args.data_type == 'realdata':
        # Load data
        with open(os.path.join("", args.real_data_file), 'rb') as data:
            X = pickle.load(data)
    elif args.data_type == 'create_new':
        # Create random DAG
        ground_truth_G = simulate_random_dag(args.node_number, args.degree)
        # Save created DAG
        with open('create_new_DAG.pkl', 'wb') as filehandle:
            pickle.dump(ground_truth_G, filehandle)
        # Generate data
        X = simulate_lsem(ground_truth_G, args.sample_size, args.A_type, x_dims)
    else:
        raise ValueError('unknown data type')
  
    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # Reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_loader = DataLoader(train_data, batch_size = batch_size)
    valid_loader = DataLoader(valid_data, batch_size = batch_size)
    test_loader = DataLoader(test_data, batch_size = batch_size)

    # ----------- Load modules:
    off_diag = np.ones([d, d]) - np.eye(d) # Generate off-diagonal interaction graph
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype = np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype = np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)
    adj_A = np.zeros((d, d)) # Add adjacency matrix

    encoder = MLPEncoder(d * x_dims, x_dims, encoder_hidden,
                             int(z_dims), adj_A,
                             batch_size = batch_size,
                             do_prob = encoder_dropout, factor = factor).double()
    decoder = MLPDecoder(d * x_dims,
                             z_dims, x_dims, encoder,
                             data_variable_size = d,
                             batch_size = batch_size,
                             n_hid=decoder_hidden,
                             do_prob=decoder_dropout).double()

    # ----------- Set up optimizer:
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = original_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = lr_decay,
                                    gamma = gamma)

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # ----------- Main:
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    h1_B_new = torch.tensor(1.)
    h2_B_new = 1
    h1_B_old = np.inf
    h2_B_old = np.inf
    lr = original_lr

    try:
        for step_k in range(k_max_iter):
            while c_B * d_B < 1e+20:
                for epoch in range(epochs):
                    old_lr = lr 
                    ELBO_loss, NLL_loss, MSE_loss, origin_B, optimizer, lr = train(epoch, lambda1, c_B, lambda2, d_B, optimizer, old_lr)

                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # Update parameters
                B_new = origin_B.data.clone()
                h1_B_new = fun_h1_B(B_new)
                h2_B_new = fun_h2_B(B_new)
                if h1_B_new.item() > 0.25 * h1_B_old and h2_B_new > 0.25 * h2_B_old:
                    c_B *= 10
                    d_B *= 10
                elif h1_B_new.item() > 0.25 * h1_B_old and h2_B_new < 0.25 * h2_B_old:
                    c_B *= 10
                elif h1_B_new.item() < 0.25 * h1_B_old and h2_B_new > 0.25 * h2_B_old:
                    d_B *= 10
                else:
                    break

            # Update parameters    
            h1_B_old = h1_B_new.item()
            h2_B_old = h2_B_new
            lambda1 += c_B * h1_B_new.item()
            lambda2 += d_B * h2_B_new

            if h1_B_new.item() <= h1_tol and h2_B_new <= h2_tol:
                break
                
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    predB = np.matrix(origin_B.data.clone().numpy())
    print('Best ELBO Loss :', best_ELBO_loss)
    print('Best NLL Loss :', best_NLL_loss)
    print('Best MSE Loss :', best_MSE_loss)
    calculate_effect(predB)
    
    return predB, best_ELBO_loss, best_NLL_loss, best_MSE_loss

#========================================
# Save Results
#========================================

if args.rep_number == 1:
    result = anoce_simu(args.seed)
    with open('ANOCE_Results.data', 'wb') as filehandle:
        pickle.dump(result, filehandle)
else:
    np.random.seed(args.seed) #Random seed
    seeds_list=np.random.randint(1, 1000000, size=args.rep_number)
    pool = Pool(n_cores)
    rep_res=pool.map(anoce_simu, seeds_list)
    with open('ANOCE_Results.data', 'wb') as filehandle:
        pickle.dump(rep_res, filehandle)