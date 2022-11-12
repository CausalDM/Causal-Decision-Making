#!/usr/bin/env python
# coding: utf-8

# # LinUCB
# 
# ## Overview
# - **Advantage**: It is more scalable and efficient than **UCB1** by utilizing features.
# - **Disadvantage**:  
# - **Application Situation**: discrete action space, binary/Gaussian reward space
# 
# ## Main Idea
# 
# 
# ## Key Steps

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


from causaldm.learners.Online.Single import LinUCB
from causaldm.learners.Online.Single import Env
import numpy as np


# In[3]:


T = 2000
K = 5
with_intercept = True
p=3
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_theta = sigma_gamma = np.identity(p)
mu_theta = np.zeros(p)
seed = 0
sigma = 1

env = Env.Single_Gaussian_Env(T, K, p, sigma
                         , mu_theta, Sigma_theta
                        , seed = 42, with_intercept = True
                         , X_mu = X_mu, X_Sigma = X_sigma)
LinUCB_Gaussian_agent = LinUCB.LinUCB_Gaussian(alpha = .5, K = K, p = p)
A = LinUCB_Gaussian_agent.take_action(env.Phi)
t = 0
R = env.get_reward(t,A)
LinUCB_Gaussian_agent.receive_reward(t,A,R, env.Phi)


# In[4]:


LinUCB_Gaussian_agent.cnts


# ## References
# 
# [1] Agrawal, S., & Goyal, N. (2013, May). Thompson sampling for contextual bandits with linear payoffs. In International conference on machine learning (pp. 127-135). PMLR.
# 
# [2] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., & Boutilier, C. (2020, June). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics (pp. 2066-2076). PMLR.
