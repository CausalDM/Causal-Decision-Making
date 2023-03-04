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
#     - sample a $\tilde{\boldsymbol{\theta}}^{t}$ from the posterior distribution of $\boldsymbol{\theta}$ or prior distribution if in round $0$
#     - take action $A_t$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}}^{t})$ solving by appropriate combinatorial optimization algorithms
#     - receive the rewad $R_t$, and update the posterior distirbution accordingly.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the combinatorial Semi-Bandit problems.

# ## Demo Code

# In[1]:


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.Structured_Bandits.Combinatorial_Semi import CombTS


# ### Generate the Environment
# 
# Here, we imitate an environment based on the Adult dataset. The length of horizon, $T$, is specified as $500$.

# In[3]:


from causaldm.learners.Online.Structured_Bandits.Combinatorial_Semi import _env_realComb as _env
env = _env.CombSemi_env(T = 500, seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - sigma: standard deviation of reward distribution (Note: we assume that the observed reward's random noise comes from the same distribution for all items.)
# - u_prior_mean: mean of the Gaussian prior of the mean rewards $\boldsymbol{\theta}$
# - u_prior_cov_diag: the diagonal of the covariance matrix of the Gaussian prior of the mean rewards $\boldsymbol{\theta}$
# - seed: random seed

# In[4]:


L = env.L
K = 10
sigma = 1
u_prior_mean = np.zeros(L)
u_prior_cov_diag = np.ones(L)
seed = 0
TS_agent = CombTS.TS_Semi(L = L, K = K, sigma = sigma, u_prior_mean = u_prior_mean,
                          u_prior_cov_diag = u_prior_cov_diag, seed = seed)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are four steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = TS_agent.take_action() </code>
# 2. Get the reward of each item recommended from the environment
# <code> R, _, tot_R = env.get_reward(A, t) </code>
# 3. Update the posterior distribution
# <code> TS_agent.receive_reward(t, A, R) </code>

# In[5]:


t = 0
A = TS_agent.take_action()
R, _, tot_R = env.get_reward(A, t)
TS_agent.receive_reward(t, A, R)
t, A, R, tot_R


# **Interpretation**: For step 0, the agent decides to send the advertisement to 10 potential customers (1054, 2060,  943,  494, 1488, 1351, 1816,  898, 1587, 1114), and then receives a total reward of $28.06$.

# ## References
# [1] Chen, W., Wang, Y., & Yuan, Y. (2013, February). Combinatorial multi-armed bandit: General framework and applications. In International conference on machine learning (pp. 151-159). PMLR.
# 
# [2] Perrault, P., Boursier, E., Valko, M., & Perchet, V. (2020). Statistical efficiency of thompson sampling for combinatorial semi-bandits. Advances in Neural Information Processing Systems, 33, 5429-5440.

# In[ ]:




