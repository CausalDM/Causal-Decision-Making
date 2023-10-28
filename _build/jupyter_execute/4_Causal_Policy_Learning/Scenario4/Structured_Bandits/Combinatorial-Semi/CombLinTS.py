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
# \theta_{i} = \boldsymbol{s}_{i,t}^{T}\boldsymbol{\gamma}.
# \end{equation}
# At each round $t$, **CombLinTS** samples $\tilde{\boldsymbol{\gamma}}_{t}$ from the updated posterior distribution $N(\hat{\boldsymbol{\gamma}}_{t},\hat{\boldsymbol{\Sigma}}_{t})$ and get the $\tilde{\theta}_{i}^{t}$ as $\boldsymbol{s}_{i,t}^{T}\tilde{\boldsymbol{\gamma}}_{t}$, where $\hat{\boldsymbol{\gamma}}_{t}$ and $\hat{\boldsymbol{\Sigma}}_{t}$ are updated by the Kalman Filtering algorithm[1]. Note that when the outcome distribution $\mathcal{P}$ is Gaussian, the updated posterior distribution is the exact posterior distribution of $\boldsymbol{\gamma}$ as **CombLinTS** assumes a Gaussian Prior. 
# 
# It's also important to note that, if necessary, the posterior updating step can be simply changed to accommodate various prior/reward distribution specifications. Further, for simplicity, we consider the most basic size constraint such that the action space includes all the possible subsets with size $K$. Therefore, the optimization process to find the optimal subset $A_{t}$ is equal to selecting a list of $K$ items with the highest attractiveness factors. Of course, users are welcome to modify the **optimization** function to satisfy more complex constraints.
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$ with a Gaussian prior;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $\tilde{\boldsymbol{\theta}}$ as $\boldsymbol{s}_{i,t}^T \tilde{\boldsymbol{\gamma}}$;
# 5. Take the action $A_{t}$ w.r.t $\tilde{\boldsymbol{\theta}}$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$;
# 6. Receive reward $R_{t}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the combinatorial Semi-Bandit problems.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.Structured_Bandits.Combinatorial_Semi import CombLinTS


# ### Generate the Environment
# 
# Here, we imitate an environment based on the Adult dataset. The length of horizon, $T$, is specified as $500$.

# In[2]:


from causaldm.learners.CPL4.Structured_Bandits.Combinatorial_Semi import _env_realComb as _env
env = _env.CombSemi_env(T = 500, seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - p: number of features (If the intercept is considerd, p includes the intercept as well.)
# - sigma: standard deviation of reward distribution (Note: we assume that the observed reward's random noise comes from the same distribution for all items.)
# - prior_gamma_mu: mean of the Gaussian prior of the $\boldsymbol{\gamma}$
# - prior_gamma_cov: the covariance matrix of the Gaussian prior of $\boldsymbol{\gamma}$
# - seed: random seed

# In[4]:


L = env.L
K = 10
p = env.p
sigma = 1
prior_gamma_mu = np.zeros(p)
prior_gamma_cov = np.identity(p)
seed = 0
LinTS_agent = CombLinTS.LinTS_Semi(sigma = sigma, prior_gamma_mu = prior_gamma_mu, 
                                   prior_gamma_cov = prior_gamma_cov,L = L, K = K, 
                                   p = p, seed = seed)


# ### Recommendation and Interaction
# We fisrt observe the feature information $\boldsymbol{S}$ by
# <code> S = env.Phi </code>. (Note: if an intercept is considered, the X should include a column of ones).
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = LinTS_agent.take_action(S) </code>
# 2. Get the reward of each item recommended from the environment
# <code> R, _, tot_R = env.get_reward(A, t) </code>
# 3. Update the posterior distribution
# <code> LinTS_agent.receive_reward(t, A, R, S) </code>

# In[5]:


S = env.Phi
t = 0
A = LinTS_agent.take_action(S)
R, _, tot_R = env.get_reward(A, t)
LinTS_agent.receive_reward(t, A, R, S)
t, A, R, tot_R


# **Interpretation**: For step 0, the agent decides to send the advertisement to 10 potential customers (480, 1895, 1700, 2219, 2807, 1593, 2784,  172, 2831, 1523), and then receives a total reward of $2.36$.

# ## References
# [1] Wen, Z., Kveton, B., & Ashkan, A. (2015, June). Efficient learning in large-scale combinatorial semi-bandits. In International Conference on Machine Learning (pp. 1113-1122). PMLR.
