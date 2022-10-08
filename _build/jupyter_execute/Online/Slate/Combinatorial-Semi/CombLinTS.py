#!/usr/bin/env python
# coding: utf-8

# # CombLinTS
# 
# ## Overview
# - **Advantage**: It is scalable when the features are used. It outperforms algorithms based on other frameworks, such as UCB, in practice.
# - **Disadvantage**: It is susceptible to model misspecification.
# - **Application Situation**: Useful when presenting a list of items, each of which will generate a partial outcome (reward). The outcome is continuous.
# 
# ## Main Idea
# Noticing that feature information are common in practice, Wen et al. (2015) considers a generalization across items to reach a lower regret bound independent of $N$, by assuming a linear generalization model for $\boldsymbol{\theta}$. Specifically, we assume that
# \begin{equation}
# \theta_{i} = \boldsymbol{x}_{i,t}^{T}\boldsymbol{\gamma}.
# \end{equation}
# At each round $t$, **CombLinTS** samples $\tilde{\boldsymbol{\gamma}}_{t}$ from the updated posterior distribution $N(\hat{\boldsymbol{\gamma}}_{t},\hat{\Sigma}_{t})$ and get the $\tilde{\theta}_{i}^{t}$ as $\boldsymbol{x}_{i,t}^{T}\tilde{\boldsymbol{\gamma}}_{t}$, where $\hat{\boldsymbol{\gamma}}_{t}$ and $\hat{\Sigma}_{t}$ are updated by the Kalman Filtering algorithm[1]. Note that when the outcome distribution $\mathcal{P}$ is Gaussian, the updated posterior distribution is the exact posterior distribution of $\boldsymbol{\gamma}$ as **CombLinTS** assumes a Gaussian Prior. 
# 
# It's also important to note that, if necessary, the posterior updating step can be simply changed to accommodate various prior/reward distribution specifications. Further, for simplicity, we consider the most basic size constraint such that the action space includes all the possible subsets with size $K$. Therefore, the optimization process to find the optimal subset $A_{t}$ is equal to selecting a list of $K$ items with the highest attractiveness factors. Of course, users are welcome to modify the **optimization** function to satisfy more complex constraints.
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$ with a Gaussian prior;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $\tilde{\boldsymbol{\theta}}$ as $\boldsymbol{x}_{i,t}^T \tilde{\boldsymbol{\gamma}}$;
# 5. Take the action $A_{t}$ w.r.t $\tilde{\boldsymbol{\theta}}$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$;
# 6. Receive reward $R_{t}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the combinatorial Semi-Bandit problems.

# ## Demo Code

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
# code used to import the learner


# In[2]:


from causaldm.learners.Online.Slate.Combinatorial_Semi import MTSS_Comb
from causaldm.learners.Online.Slate.Combinatorial_Semi import _env_SemiBandit
import numpy as np


# In[5]:


L, T, K, p = 300, 1000, 5, 3
mu_gamma = np.zeros(p)
sigma_gamma = np.identity(p)
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
with_intercept = True
seed = 0
sigma_1 = .5
sigma_2 = 1

env = _env_SemiBandit.Semi_env(L, K, T, p, sigma_1, sigma_2
                               , mu_gamma, sigma_gamma, seed = seed
                               , with_intercept = with_intercept
                               , X_mu = X_mu, X_sigma = X_sigma)
MTSS_agent = MTSS_Comb.MTSS_Semi(sigma_2 = 1, L=L, T = T
                                 , gamma_prior_mean = np.zeros(p), gamma_prior_cov = np.identity(p)
                                 , sigma_1 = sigma_1
                                 , K = K
                                 , Xs = env.Phi# [L, p]
                                 , update_freq = 1)
S = MTSS_agent.take_action(env.Phi)
t = 1
obs_R, exp_R, R = env.get_reward(S, t)
MTSS_agent.receive_reward(t, S, obs_R, X = env.Phi)


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# [1] Wen, Z., Kveton, B., & Ashkan, A. (2015, June). Efficient learning in large-scale combinatorial semi-bandits. In International Conference on Machine Learning (pp. 1113-1122). PMLR.

# In[ ]:




