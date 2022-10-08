#!/usr/bin/env python
# coding: utf-8

# # CombTS
# 
# ## Overview
# - **Advantage**: In practice, it always outperforms algorithms that also do not use features but are based on other frameworks, such as UCB.
# - **Disadvantage**: When there are a large number of items, it is not scalable.
# - **Application Situation**: Useful when presenting a list of items, each of which will generate a partial outcome (reward). The outcome is continuous.
# 
# ## Main Idea
# Recently, researchers began adapting the TS framework for combinatorial semi-bandits problems from a Bayesian perspective. **CombTS** [2] has been developed for the general family of sub-Gaussian outcomes $Y_{i,t}(a)$ by assuming a Gaussian prior for each $\theta_i$ and then updating the posterior distribution using Bayes' rule. The optimal action can be obtained from a combinatorial optimization problem with estimates of the mean reward $\theta_i$ of each item, which can be efficiently solved by corresponding combinatorial optimization algorithms in most real-world applications [1]. 
# 
# It should be noted that the posterior updating step differs for different pairs of the prior distribution of expected potential reward of each item and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary. Further, for simplicity, we consider the most basic size constraint such that the action space includes all the possible subsets with size $K$. Therefore, the optimization process to find the optimal subset $A_{t}$ is equal to selecting a list of $K$ items with the highest attractiveness factors. Of course, users are welcome to modify the **optimization** function to satisfy more complex constraints.
# 
# ## Key Steps
# 1. Specifying a prior distirbution of each $\theta_i$, i.e., Normal(0,1).
# 2. For t = $0, 1,\cdots, T$:
#     - sample a $\tilde{\theta}^{t}$ from the posterior distribution of $\theta$ or prior distribution if in round $0$
#     - take action $A_t$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$ solving by appropriate combinatorial optimization algorithms
#     - receive the rewad $R_t$, and update the posterior distirbution accordingly.
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
# [1] Chen, W., Wang, Y., & Yuan, Y. (2013, February). Combinatorial multi-armed bandit: General framework and applications. In International conference on machine learning (pp. 151-159). PMLR.
# 
# [2] Perrault, P., Boursier, E., Valko, M., & Perchet, V. (2020). Statistical efficiency of thompson sampling for combinatorial semi-bandits. Advances in Neural Information Processing Systems, 33, 5429-5440.

# In[ ]:




