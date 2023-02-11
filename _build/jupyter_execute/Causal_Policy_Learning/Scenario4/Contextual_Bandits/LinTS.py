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
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. Noticing that feature information are commonly avialable, the **LinTS**[1,2] algorithm consdiers modeling the expectation of the potential reward $R_t(i)$ with features of item $i$. As an example, considering the Gaussian reward, we assume that 
# \begin{align}
# R_t(i)\sim \mathcal{N}(\boldsymbol{s}_i^T \boldsymbol{\gamma},\sigma^2).
# \end{align}
# As for the Bernoulli reward, we assume that 
# \begin{align}
# R_t(i) \sim Bernoulli(logistic(\boldsymbol{s}_i^T \boldsymbol{\gamma})).
# \end{align}where $logistic(s) \equiv 1 / (1 + exp^{-1}(s))$.
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


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.CMAB import LinTS


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[3]:


from causaldm.learners.Online.CMAB import _env_realCMAB as _env
env = _env.Single_Contextual_Env(seed = 0, Binary = False)


# ### Specify Hyperparameters
# 
# - K: number of arms
# - p: number of features per arm
# - sigma: the standard deviation of the reward distributions
# - prior_theta_u: mean of the prior distribution of the coefficients
# - prior_theta_cov: Covaraince matrix of the prior distribution of the coefficients
# - seed: random seed

# In[4]:


K = env.K
p = env.p
seed = 42
sigma = 1
prior_theta_u = np.zeros(p)
prior_theta_cov = np.identity(p)

LinTS_Gaussian_agent = LinTS.LinTS_Gaussian(sigma = sigma, prior_theta_u = prior_theta_u, 
                                            prior_theta_cov = prior_theta_cov, 
                                            K = K, p = p,seed = seed)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Observe the feature information
# <code> X = env.get_Phi(t) </code>
# 2. Recommend an action 
# <code> A = LinTS_Gaussian_agent.take_action(X) </code>
# 3. Get the reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 4. Update the posterior distribution
# <code> LinTS_Gaussian_agent.receive_reward(t,A,R,X) </code>

# In[5]:


t = 0
X, feature_info = env.get_Phi(t)
A = LinTS_Gaussian_agent.take_action(X)
R = env.get_reward(t,A)
LinTS_Gaussian_agent.receive_reward(t,A,R,X)
t,A,R,feature_info


# **Interpretation**: For step 0, the TS agent encounter a male user who is a 25-year-old college/grad student. Given the information, the agent recommend a Comedy (arm 0), and receive a rate of 3 from the user.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit. Note that, when specifying the prior distribution of the expected reward, the mean-precision form of the Beta distribution is used here, i.e., Beta($\mu$, $\phi$), where $\mu$ is the mean reward of each arm and $\phi$ is the precision of the Beta distribution. 

# In[6]:


env = _env.Single_Contextual_Env(seed = 0, Binary = True)
K = env.K
p = env.p
seed = 42
alpha = 1 # exploration rate
retrain_freq = 1 #frequency to train the GLM model

LinTS_GLM_agent = LinTS.LinTS_GLM(K = K, p = p , alpha = alpha, retrain_freq = retrain_freq)
t = 0
X, feature_info = env.get_Phi(t)
A = LinTS_GLM_agent.take_action(X)
R = env.get_reward(t,A)
LinTS_GLM_agent.receive_reward(t,A,R,X)
t,A,R,feature_info


# **Interpretation**: For step 0, the TS agent encounter a male user who is a 25-year-old college/grad student. Given the information, the agent recommend a Sci-Fi (arm 4), and receive a rate of 1 from the user.

# ## References
# 
# [1] Agrawal, S., & Goyal, N. (2013, May). Thompson sampling for contextual bandits with linear payoffs. In International conference on machine learning (pp. 127-135). PMLR.
# 
# [2] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., & Boutilier, C. (2020, June). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics (pp. 2066-2076). PMLR.
