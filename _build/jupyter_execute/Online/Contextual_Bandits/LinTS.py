#!/usr/bin/env python
# coding: utf-8

# # LinTS
# 
# ## Overview
# - **Advantage**: It is scalable by utilizing features. It outperforms algorithms based on other frameworks, such as UCB, in practice.
# - **Disadvantage**: It is susceptible to model misspecification.
# - **Application Situation**: discrete action space, binary/Gaussian reward space
# 
# ## Main Idea
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. Noticing that feature information are commonly avialable, the **LinTS** algorithm consdiers modeling the expectation of the potential reward $R_t(i)$ with features of item $i$. As an example, considering the Gaussian reward, we assume that 
# \begin{align}
# R_t(i)\sim \mathcal{N}(\boldsymbol{x}_i^T \boldsymbol{\gamma},\sigma^2).
# \end{align}
# As for the Bernoulli reward, we assume that 
# \begin{align}
# R_t(i) \sim Bernoulli(logistic(\boldsymbol{x}_i^T \boldsymbol{\gamma})).
# \end{align}where $logistic(x) \equiv 1 / (1 + exp^{-1}(x))$.
# Similar as the standard TS algorithm, the LinTS algorithm starts with specifying a prior distribution of the parameter $\boldsymbol{\gamma}$, and a variance of the reward, based on the domian knowledge. At each round $t$, the agent will samples a vector of $\tilde{\boldsymbol{\gamma}}^{t}$ from thecorresponding posterior distribution, and the mean reward $\tilde{\boldsymbol{\theta}}^{t}$ is then calculated accordingly. The action $a$ with the greatest $\tilde{\theta}_{a}^{t}$ is then selected. Finally, the posterior distribution would be updated after receiving the feedback at the end of each round. It should be noted that the posterior updating step differs for different pairs of the prior distribution of expected potential reward and reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# ## Key Steps
# 
# 1. Specifying a prior distirbution of $\boldsymbol{\gamma}$, and the variance of the reward distribution.
# 2. For t = $0, 1,\cdots, T$:
#     - sample a $\tilde{\boldsymbol{\gamma}}^{t}$ from the posterior distribution of $\boldsymbol{\gamma}$ or the prior distribution of it if in round $0$
#     - calculated the $\tilde{\boldsymbol{\theta}}^{t}$ based on the assumed linear relationship
#     - select action $A_t$ which has the greatest $\tilde{\theta}_{a}$, i.e. $A_t = \arg\max_{a \in \mathcal{A}} \tilde{\theta}_{a}^{t}$
#     - receive the rewad $R$, and update the posterior distirbution of $\boldsymbol{\gamma}$ accordingly.

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


from causaldm.learners.Online.Single import LinTS
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
LinTS_Gaussian_agent = LinTS.LinTS_Gaussian(sigma = 1
                                         , prior_theta_u = np.zeros(p), prior_theta_cov = np.identity(p)
                                         , K = K, p = p)
A = LinTS_Gaussian_agent.take_action(env.Phi)
t = 0
R = env.get_reward(t,A)
LinTS_Gaussian_agent.receive_reward(t,A,R, env.Phi)


# In[4]:


LinTS_Gaussian_agent.cnts


# In[5]:


T = 2000
K = 5
with_intercept = True
p=3
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_theta = sigma_gamma = np.identity(p)
mu_theta = np.zeros(p)
seed = 0
phi_beta = 1/4

env = Env.Single_Bernoulli_Env(T, K, p, phi_beta
                         , mu_theta, Sigma_theta
                        , seed = 42, with_intercept = True
                         , X_mu = X_mu, X_Sigma = X_sigma)
LinTS_GLM_agent = LinTS.LinTS_GLM(K = K, p = p , alpha = 1, retrain_freq = 1)
A = LinTS_GLM_agent.take_action(env.Phi)
t = 0
R = env.get_reward(t,A)
LinTS_GLM_agent.receive_reward(t,A,R, env.Phi)


# In[6]:


LinTS_Bernoulli_agent.cnts


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# 
# [1] Agrawal, S., & Goyal, N. (2013, May). Thompson sampling for contextual bandits with linear payoffs. In International conference on machine learning (pp. 127-135). PMLR.
# 
# [2] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., & Boutilier, C. (2020, June). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics (pp. 2066-2076). PMLR.

# In[ ]:




