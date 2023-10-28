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
#     - sample a $\tilde{\boldsymbol{\theta}}^{t}$ from the posterior distribution of $\boldsymbol{\theta}$ or prior distribution if in round $0$
#     - select top $K$ items with the greatest $\tilde{\theta}_{a}$, i.e. $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}}^{t})$
#     - receive the rewad $R_t$, and update the posterior distirbution accordingly.
#     
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the cascading Bandit problems.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.Structured_Bandits.Cascade import TS_Cascade


# ### Generate the Environment
# 
# Here, we imitate an environment based on the Yelp dataset. The number of items recommended at each round, $K$, is specified as $3$.

# In[2]:


from causaldm.learners.CPL4.Structured_Bandits.Cascade import _env_realCascade as _env
env = _env.Cascading_env(K = 3, seed = 0)


# ### Specify Hyperparameters
# 
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - u_prior_alpha: Alpha of the prior Beta distribution
# - u_prior_beta: Beta of the prior Beta distribution
# - seed: random seed

# In[4]:


K = env.K
L = env.L
u_prior_alpha = np.ones(L)
u_prior_beta = np.ones(L)
seed = 0

TS_agent = TS_Cascade.TS_Cascade(K = K, L = L, u_prior_alpha = u_prior_alpha, 
                                 u_prior_beta = u_prior_beta, seed = seed)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = TS_agent.take_action(X) </code>
# 3. Get the reward from the environment (i.e., $W$, $E$, and $R$)
# <code> W,E,R = env.get_reward(A) </code>
# 4. Update the posterior distribution
# <code> TS_agent.receive_reward(A,W,E,t) </code>

# In[5]:


t = 0
A = TS_agent.take_action()
W, E, R = env.get_reward(A)
TS_agent.receive_reward(A, W, E, t)
A, W, E, R


# **Interpretation**: For step 0, the agent decides to display three top restaurants, the first of which is restaurant 507, the second is restaurant 2690, and the third is restaurant 2006. Unfortunately, the customer does not show any interest in any of the recommended restaurants. As a result, the agent receives a zero reward at round $0$.

# ## References
# 
# [1] Cheung, W. C., Tan, V., & Zhong, Z. (2019, April). A Thompson sampling algorithm for cascading bandits. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 438-447). PMLR.

# In[ ]:




