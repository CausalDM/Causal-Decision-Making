#!/usr/bin/env python
# coding: utf-8

# # TS_Contextual_MNL
# 
# ## Overview
# - **Advantage**: It is scalable when the features are used. It outperforms algorithms based on other frameworks, such as UCB, in practice.
# - **Disadvantage**: It is susceptible to model misspecification.
# - **Application Situation**: Useful when a list of items is presented, each with a matching price or income, and only one is chosen for each interaction. Binary responses from users include click/don't-click and buy/don't-buy.
# 
# 
# ## Main Idea
# Feature-determined approaches have been developed recently to provide a more feasible approach for large-scale problems, by adapting either the UCB framwork or the TS framework. While all of them [1,2,3] are under the standard offering structure, here we modify the TS-type algorithm in [3] by adapting to the epoch-type offering framework and assuming a linear realtionship between the utility and the item features as 
# \begin{equation}
# \theta_i = \frac{logistic(\boldsymbol{s}_{i,t}^T \boldsymbol{\gamma})+ 1}{2},
# \end{equation} to tackle the challenge of a large item space. We named the proposed algorithm as **TS_Contextual_MNL**. At each decision round $t$, **TS_Contextual_MNL** samples $\tilde{\boldsymbol{\gamma}}_{t}$ from the posterior distribution, which is updated by **Pymc3**, and get the $\tilde{\theta}_{i}^{t}$ as $\frac{logistic(\boldsymbol{s}_{i,t}^T \text{ }\tilde{\boldsymbol{\gamma}})+ 1}{2}$ and $\tilde{v}_{i}^{l}$ as $1/\tilde{\theta}_{i}^{l}-1$. Finally, linear programming is employed to determine the optimal assortment $A^{l}$, such that
# \begin{equation}
#     A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}},
# \end{equation} where $t$ is the first round of epoch $l$.  
# 
# It should be noted that the posterior updating step differs for different pairs of the prior distribution of $\boldsymbol{\gamma}$ and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# ## Key Steps
# For epoch $l = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}^{l})$ by **Pymc3**;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}^{l})$;
# 3. Update $\tilde{\boldsymbol{\theta}} = \frac{logistic(\boldsymbol{s}_{i,t}^T \text{ }\tilde{\boldsymbol{\gamma}})+ 1}{2}$
# 4. Compute the utility $\tilde{v}_{i} = \frac{1}{\tilde{\theta}_{i}}-1$;
# 5. Take the action $A^{l}$ w.r.t $\{\tilde{v}_{i}\}_{i=1}^{N}$ such that $A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}}$;
# 6. Offer $A^{l}$ until no purchase appears;
# 7. Receive reward $R^{l}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the Multinomial Logit Bandit problems.

# ## Demo Code

# In[1]:


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.Structured_Bandits.MNL import TS_Contextual_MNL


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens dataset.

# In[3]:


from causaldm.learners.Online.Structured_Bandits.MNL import _env_realMNL as _env
env = _env.MNL_env(seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - Xs: feature informations $\boldsymbol{S}$ (Note: if an intercept is considered, the $\boldsymbol{S}$ should include a column of ones)
# - gamma_prior_mean: the mean of the prior distribution of $\boldsymbol{\gamma}$
# - gamma_prior_cov: the coveraince matrix of the prior distribution of $\boldsymbol{\gamma}$ 
# - r: revenue of items
# - same_reward: indicate whether the revenue of each item is the same or not
# - n_init: determine the number of samples that pymc3 will draw when updating the posterior of $\boldsymbol{\gamma}$ 
# - update_freq: frequency to update the posterior distribution of $\boldsymbol{\gamma}$ (i.e., update every update_freq steps)
# - seed: random seed

# In[4]:


L = env.L
K = 5
Xs = env.Phi
gamma_prior_mean = np.ones(env.p)
gamma_prior_cov = np.identity(env.p)
r = env.r
same_reward = False
n_init = 1000
update_freq = 100
seed = 0

LinTS_agent = TS_Contextual_MNL.MNL_TS_Contextual(L = L, K = K, Xs = Xs, gamma_prior_mean = gamma_prior_mean, 
                                                  gamma_prior_cov = gamma_prior_cov, r = r, same_reward = same_reward, 
                                                  n_init = n_init, update_freq=update_freq, seed = seed)


# ### Recommendation and Interaction
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = LinTS_agent.take_action() </code>
# 3. Get the item clicked and the corresponding revenue from the environment
# <code> c, _, R = env.get_reward(A) </code>
# 4. Update the posterior distribution
# <code> LinTS_agent.receive_reward(A,c,R) </code>

# In[5]:


t = 0
A = LinTS_agent.take_action()
c, _, R= env.get_reward(A)
LinTS_agent.receive_reward(A, c, R)
A, c, R


# **Interpretation**: For step 0, the agent recommends five movies to the customer, the ids of which are 20, 298, 421, 448, and 836. The customer finally clicks the movie 298 and the agent receives a revenue of .97.

# ## References
# 
# [1] Ou, M., Li, N., Zhu, S., & Jin, R. (2018). Multinomial logit bandit with linear utility functions. arXiv preprint arXiv:1805.02971.
# 
# [2] Agrawal, P., Avadhanula, V., & Tulabandhula, T. (2020). A tractable online learning algorithm for the multinomial logit contextual bandit. arXiv preprint arXiv:2011.14033.
# 
# [3] Oh, M. H., & Iyengar, G. (2019). Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32.
# 

# In[ ]:




