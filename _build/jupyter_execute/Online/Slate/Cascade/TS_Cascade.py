#!/usr/bin/env python
# coding: utf-8

# # TS_Cascade
# 
# ## Overview
# - **Advantage**: In practice, it always outperforms algorithms that also do not use features but are based on other frameworks, such as UCB.
# - **Disadvantage**: When there are a large number of items, it is not scalable.
# - **Application Situation**: It is useful when you want to present a ranked list of items, with only one selected at each interaction. Binary outcome.
# 
# ## Main Idea
# Recently, from a Bayesian perspective, [1] deals with the trade-off between exploration and exploitation by adapting the TS algorithm to the cascading bandits. Specifically, noticing that the potential observation $W(a)$ follows a Bernoulli distribution, **TS_Cascade** adapted the standard TS algorithms directly by assuming a prior distribution of $\theta_{i}=E[W_t(i)]$ to be Beta-distributed and use the Beta-Bernoulli conjugate update accordingly. For each round $t$, after updating the posterior distribution of $\theta_{i}$, we select the top $K$ items with the highest estimated attractiveness factors. It should be noted that the posterior updating step differs for different pairs of the prior distribution of expected potential reward and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# ## Key Steps
# 1. Specifying a prior distirbution of each $\theta_i$, i.e., Beta(1,1).
# 2. For t = $0, 1,\cdots, T$:
#     - sample a $\tilde{\theta}^{t}$ from the posterior distribution of $\theta$ or prior distribution if in round $0$
#     - select top $K$ items with the greatest $\tilde{\theta}_{a}$, i.e. $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}}^{t})$
#     - receive the rewad $R_t$, and update the posterior distirbution accordingly.
#     
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the cascading Bandit problems.

# ## Demo Code

# In[1]:


# After we publish the package, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
# code used to import the learner


# In[2]:


from causaldm.learners.Online.Slate.Cascade import TS_Cascade
from causaldm.learners.Online.Slate.Cascade import _env_Cascade
import numpy as np


# In[3]:


L, T, K, p = 250, 10000, 3, 5
update_freq = 500
update_freq_linear = 500

phi_beta = 1/4
n_init = 500
with_intercept = True
same_reward = True
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_gamma = sigma_gamma = np.identity(p)
mu_gamma = np.zeros(p)
seed = 0

env = _env_Cascade.Cascading_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                    X_mu, X_sigma,                                       
                                    phi_beta, same_reward = same_reward, 
                                    seed = seed, p = p, with_intercept = with_intercept)
TS_agent = TS_Cascade.TS_Cascade(K = K, L = L)
S = TS_agent.take_action(env.Phi)
t = 1
W, E, exp_R, R = env.get_reward(S)
TS_agent.receive_reward(S, W, E, exp_R, R, t, env.Phi)


# In[4]:


S


# ## References
# 
# [1] Cheung, W. C., Tan, V., & Zhong, Z. (2019, April). A Thompson sampling algorithm for cascading bandits. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 438-447). PMLR.
