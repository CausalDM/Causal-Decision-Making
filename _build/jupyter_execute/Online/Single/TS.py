#!/usr/bin/env python
# coding: utf-8

# # TS
# 
# ## Main Idea
# Thompson Sampling, also known as posterior sampling, solves the exploration-exploitation dilemma by selecting an action according to its posterior distribution [8].  At each round $t$, the agent sample the rewards from the corresponding posterior distributions and then select the action with the highest sampled reward greedily. It has been shown that, when the true reward distribution is known, a TS algorithm with the true reward distribution as the prior is nearly optimal [9]. However, such a distribution is always unknown in practice. Therefore, one of the major objectives of TS-based algorithms is to find an informative prior to guide the exploration.
# 
# 
# ## Algorithms Details
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. The TS algorithm starts with specifying a prior distribution of the reward, based on the domian knowledge. At each round $t$, the agent will samples a vector of $\theta^{t}$ from the posterior distribution of the rewards. The action $a$ with the greatest $\theta_{a}^{t}$ is then selected. Finally, the posterior distribution would be updated after receiving the feedback at the end of each round. Note that the posterior updating step differs for different pairs of prior distribution of the mean reward and reward distribution. Here, we consider two classical examples of the TS algorithm, including Gaussian reward with Gaussian prior and Bernoulli with Breward with Beta prior. The posterior updating is straightforward for both cases, since the nice conjugate property. In both cases, the variance of reward is assumed to be known, and need to be specified manually. Note that code can be easily modified to different specifications of the prior/reward distribution.
# 
# ## Key Steps
# 
# 1. Specifying a prior distirbution of $\theta$, and the variance of the reward distribution.
# 2. For t = $0, 1,\cdots, T$:
#     - sample a $\tilde{\theta}^{t}$ from the posterior distribution of $\theta$ or prior distribution if in round $0$
#     - select action $A_t$ which has the greatest $\tilde{\theta}_{a}$, i.e. $A_t = argmax_{a \in \mathcal{A}} \tilde{\theta}_{a}^{t}$
#     - receive the rewad $R$, and update the posterior distirbution accordingly.

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


from causaldm.learners.Online.Single import TS
from causaldm.learners.Online.Single import Env
import numpy as np


# In[8]:


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
TS_Gaussian_agent = TS.TS(Reward_Type = "Gaussian", sigma = sigma, u_prior_mean = np.ones(K), u_prior_cov = np.identity(K), prior_phi_beta = None)
A = TS_Gaussian_agent.take_action()
t = 0
R = env.get_reward(t,A)
TS_Gaussian_agent.receive_reward(t,A,R)


# In[9]:


TS_Gaussian_agent.posterior_u


# In[4]:


T = 2000
K = 5

phi_beta = 1/4
with_intercept = True
p=3
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_theta = sigma_gamma = np.identity(p)
mu_theta = np.zeros(p)
seed = 0

env = Env.Single_Bernoulli_Env(T, K, p, phi_beta
                         , mu_theta, Sigma_theta
                        , seed = 42, with_intercept = True
                         , X_mu = X_mu, X_Sigma = X_sigma)
TS_Bernoulli_agent = TS.TS(Reward_Type = "Bernoulli", sigma = 1, u_prior_mean = .5*np.ones(K), u_prior_cov = None, prior_phi_beta = phi_beta)
A = TS_Bernoulli_agent.take_action()
t = 0
R = env.get_reward(t,A)
TS_Bernoulli_agent.receive_reward(t,A,R)


# In[5]:


TS_Bernoulli_agent.posterior_alpha


# In[6]:


TS_Bernoulli_agent.posterior_beta


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# 
# [1] Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on thompson sampling. Foundations and Trends® in Machine Learning, 11(1), 1-96.

# In[ ]:




