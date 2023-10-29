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
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. **LinUCB**[1] uses feature information to guide exploration by assuming a linear model between the expected potential reward and the features. Specifcially, for the Gaussain bandits, we assume that 
# \begin{align}
# E(R_{t}(a)) = \theta_a = \boldsymbol{s}_a^T \boldsymbol{\gamma}.
# \end{align} Solving a linear gression model, at each round $t$, the corresponding estimated upper confidence interval of the mean potential reward is then updated as
# \begin{align}
# U_a^t = \boldsymbol{s}_a^T \hat{\boldsymbol{\gamma}} + \alpha\sqrt{\boldsymbol{s}_a^T \boldsymbol{V}^{-1}  \boldsymbol{s}_a},
# \end{align} where $\alpha$ is a tuning parameter that controls the rate of exploration, $\boldsymbol{V}^{-1} = \sum_{j=0}^{t-1}\boldsymbol{s}_{a_j}\boldsymbol{s}_{a_j}^T$, and $\hat{\boldsymbol{\gamma}} = \boldsymbol{V}^{-1}\sum_{j=0}^{t-1}\boldsymbol{s}_{a_j}R_j$.
# 
# As for the Bernoulli bandits, we assume that 
# \begin{align}
# \theta_{a} = logistic(\boldsymbol{s}_a^T \boldsymbol{\gamma}),
# \end{align}where $logistic(c) \equiv 1 / (1 + exp^{-1}(c))$. At each round $t$, by fitting a generalized linear model to all historical observations, we obtain the maximum likelihood estimator of $\boldsymbol{\gamma}$. The corresponding estimated confidence upper bound is then calculated in the same way as for Gaussian bandits, such that
# \begin{align}
# U_a^t = \boldsymbol{s}_a^T \hat{\boldsymbol{\gamma}} + \alpha\sqrt{\boldsymbol{s}_a^T \boldsymbol{V}^{-1}  \boldsymbol{s}_a},
# \end{align}where $\alpha$ and $\boldsymbol{V}$ are defined in the same way as before. 
# 
# Finally, using the estimated upper confidence bounds, $A_t = \arg \max_{a \in \mathcal{A}} U_a^t$ would be selected.
# 
# 
# ## Key Steps
# 
# 1. Initializing $\hat{\boldsymbol{\gamma}}=\boldsymbol{0}$ and $\boldsymbol{V} = I$, and specifying $\alpha$;
# 2. For t = $0, 1,\cdots, T$:
#     - Calculate the upper confidence bound $U_a^t$;
#     - Select action $A_t$ as the arm with the maximum $U_a^t$;
#     - Receive the reward R, and update $\hat{\boldsymbol{\gamma}}$, $V$.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.CMAB import LinUCB


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[2]:


from causaldm.learners.CPL4.CMAB import _env_realCMAB as _env
env = _env.Single_Contextual_Env(seed = 0, Binary = False)


# ### Specify Hyperparameters
# 
# - K: number of arms
# - p: number of features per arm
# - alpha: rate of exploration
# - seed: random seed
# - exploration_T: number of rounds to do random exploration at the beginning

# In[3]:


alpha = .1
K = env.K
p = env.p
seed = 42
exploration_T = 10
LinUCB_Gaussian_agent = LinUCB.LinUCB_Gaussian(alpha = .1, K = K, p = p, seed = seed, exploration_T = exploration_T)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Observe the feature information
# <code> X = env.get_Phi(t) </code>
# 2. Recommend an action 
# <code> A = LinUCB_Gaussian_agent.take_action(X) </code>
# 3. Get the reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 4. Update the posterior distribution
# <code> LinUCB_Gaussian_agent.receive_reward(t,A,R,X) </code>

# In[4]:


t = 0
X, feature_info = env.get_Phi(t)
A = LinUCB_Gaussian_agent.take_action(X)
R = env.get_reward(t,A)
LinUCB_Gaussian_agent.receive_reward(t,A,R,X)
t,A,R,feature_info


# **Interpretation**: For step 0, the TS agent encounter a male user who is a 25-year-old college/grad student. Given the information, the agent recommend a Sci-Fi (arm 4), and receive a rate of 4 from the user.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit. 

# In[5]:


env = _env.Single_Contextual_Env(seed = 0, Binary = True)
K = env.K
p = env.p
seed = 42
alpha = 1 # exploration rate
retrain_freq = 1 #frequency to train the GLM model
exploration_T = 10
LinUCB_GLM_agent = LinUCB.LinUCB_GLM(K = K, p = p , alpha = alpha, retrain_freq = retrain_freq, 
                                     seed = seed, exploration_T = exploration_T)
t = 0
X, feature_info = env.get_Phi(t)
A = LinUCB_GLM_agent.take_action(X)
R = env.get_reward(t,A)
LinUCB_GLM_agent.receive_reward(t,A,R,X)
t,A,R,feature_info


# **Interpretation**: For step 0, the TS agent encounter a male user who is a 25-year-old college/grad student. Given the information, the agent recommend a Thriller (arm 3), and receive a rate of 0 from the user.

# ## References
# 
# [1] Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011, June). Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (pp. 208-214). JMLR Workshop and Conference Proceedings.

# In[ ]:




