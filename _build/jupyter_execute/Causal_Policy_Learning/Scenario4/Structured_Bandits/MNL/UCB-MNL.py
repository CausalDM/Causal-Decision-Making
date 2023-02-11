#!/usr/bin/env python
# coding: utf-8

# # UCB_MNL
# 
# ## Main Idea
# UCB_MNL [1] is an UCB-based algorithm, using the epoch-type offering schedule to deal with dynamic assortment optimization problems. Adapted to the standard UCB-style framework, UCB_MNL estimates the upper confidence bound of $v_{i}$ at epoch $l$ by 
# \begin{equation}
#     v_{i,l}^{UCB}= \hat{v_{i}^{l}}+\sqrt{\hat{v_{i}^{l}}\frac{48log(\sqrt{N}l+1)}{s_{i,l}}}+\frac{48log(\sqrt{N}l+1)}{s_{i,l}}.
# \end{equation} Note that $s_{i,l}$ is the number of epochs offering an assortment including item $i$ before epoch $l$ and $\hat{v_{i}^{l}}$ is the average sample weight of item $i$ (i.e.,$\frac{\text{\# product $i$ was purchased}}{s_{i,l}}$). 
# 
# 
# 
# 💥 Application Situation?
# 
# ## Algorithm Details / Key Steps
# Initialization: $v_{i,0}^{UCB}=0$ for all $i$.
# 
# For epoch $l = 1,2,\cdots$:
# 1. Take the action $A^{l}$ w.r.t $\{v_{i,l-1}^{UCB}\}_{i=1}^{N}$ such that $A^{l} = argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}v_{i,l-1}^{UCB}}{1+\sum_{j\in a} v_{j,l-1}^{UCB}}$;
# 2. Offer $A^{l}$ until no purchase appears;
# 3. Receive reward $R^{l}$;
# 4. Update $v_{i,l}^{UCB}=0$ based on the observations as
# \begin{equation}
#     v_{i,l}^{UCB}= \hat{v_{i}^{l}}+\sqrt{\hat{v_{i}^{l}}\frac{48log(\sqrt{N}l+1)}{s_{i,l}}}+\frac{48log(\sqrt{N}l+1)}{s_{i,l}}
# \end{equation}, with $\hat{v_{i}^{l}}$ be the average sample weight of item $i$ (i.e.,$\frac{\text{\# product $i$ was purchased}}{s_{i,l}}$).
# 
# 
# ## Demo Code
# 💥 In the following, we exhibit how to apply the learner on real data.
# 
# *Notations can be found in the introduction of the combinatorial Semi-Bandit problems.

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
# code used to import the learner


# In[2]:


from causaldm.learners.Online.Slate.MNL import UCB_MNL
from causaldm.learners.Online.Slate.MNL import _env_MNL
import numpy as np


# In[14]:


T = 20000
L = 1000
update_freq = 500
update_freq_linear = 500

phi_beta = 1/4
n_init = 500
with_intercept = True
same_reward = False
p=3
K=5
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_gamma = sigma_gamma = np.identity(p)
mu_gamma = np.zeros(p)
seed = 0

env = _env_MNL.MNL_env(L, K, T, mu_gamma, sigma_gamma, X_mu, X_sigma,                                       
                        phi_beta, same_reward = same_reward, 
                        seed = seed, p = p, with_intercept = with_intercept)
UCB_agent = UCB_MNL.UCB_MNL(L, env.r, K, seed = 0)
S = UCB_agent.take_action()
t = 1
c, exp_R, R = env.get_reward(S)
UCB_agent.receive_reward(S, c, R, exp_R)


# In[20]:


S


# ## References
# 
# [1] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2019). Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5), 1453-1485.
