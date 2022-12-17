#!/usr/bin/env python
# coding: utf-8

# # LinUCB
# 
# ## Overview
# - **Advantage**: It is more scalable and efficient than **UCB** by utilizing features.
# - **Disadvantage**:  
# - **Application Situation**: discrete action space, Gaussian reward space
# 
# ## Main Idea
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. **LinUCB** uses feature information to guide exploration by assuming a linear model between the expected potential reward and the features. Specifcially, for the Gaussain bandits, we assume that 
# \begin{align}
# E(R_{t}(a)) = \theta_a = \boldsymbol{x}_a^T \boldsymbol{\gamma}.
# \end{align} Solving a linear gression model, at each round $t$, the corresponding estimated upper confidence interval of the mean potential reward is then updated as
# \begin{align}
# U_a^t = \boldsymbol{x}_a^T \hat{\boldsymbol{\gamma}} + \alpha\sqrt{\boldsymbol{x}_a^T V^{-1}  \boldsymbol{x}_a},
# \end{align} where $\alpha$ is a tuning parameter that controls the rate of exploration, $V^{-1} = \sum_{j=0}^{t-1}\boldsymbol{x}_{a_j}\boldsymbol{x}_{a_j}^T$, and $\hat{\boldsymbol{\gamma}} = V^{-1}\sum_{j=0}^{t-1}\boldsymbol{x}_{a_j}R_j$.
# 
# As for the Bernoulli bandits, we assume that 
# \begin{align}
# \theta_{a} = logistic(\boldsymbol{x}_a^T \boldsymbol{\gamma}),
# \end{align}where $logistic(x) \equiv 1 / (1 + exp^{-1}(x))$. At each round $t$, by fitting a generalized linear model to all historical observations, we obtain the maximum likelihood estimator of $\boldsymbol{\gamma}$. The corresponding estimated confidence upper bound is then calculated in the same way as for Gaussian bandits, such that
# \begin{align}
# U_a^t = \boldsymbol{x}_a^T \hat{\boldsymbol{\gamma}} + \alpha\sqrt{\boldsymbol{x}_a^T V^{-1}  \boldsymbol{x}_a},
# \end{align}where $\alpha$ and $V$ are defined in the same way as before. 
# 
# Finally, using the estimated upper confidence bounds, $A_t = \arg \max_{a \in \mathcal{A}} U_a^t$ would be selected.
# 
# 
# ## Key Steps
# 
# 1. Initializing $\hat{\boldsymbol{\gamma}}=\boldsymbol{0}$ and $V = I$, and specifying $\alpha$;
# 2. For t = $0, 1,\cdots, T$:
#     - Calculate the upper confidence bound $U_a^t$;
#     - Select action $A_t$ as the arm with the maximum $U_a^t$;
#     - Receive the reward R, and update $\hat{\boldsymbol{\gamma}}$, $V$.

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
# [1] Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011, June). Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (pp. 208-214). JMLR Workshop and Conference Proceedings.

# In[ ]:




