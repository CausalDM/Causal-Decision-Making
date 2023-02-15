#!/usr/bin/env python
# coding: utf-8

# # CascadeLinTS
# 
# ## Overview
# - **Advantage**: It is scalable when the features are used. It outperforms algorithms based on other frameworks, such as UCB, in practice.
# - **Disadvantage**: It is susceptible to model misspecification.
# - **Application Situation**: Useful when presenting a ranked list of items, with only one selected at each interaction. The outcome is binary.
# 
# ## Main Idea
# 
# Motivated by observations in most real-world applications, which have a large number of candidate items, Zong et al. (2016) proposed using feature information that is widely available to improve learning efficiency. Utilizing the feature information of each item $i$, **CascadeLinTS** [1] characterize $\theta_{i}=E[W_t(i)]$ by assuming that
# \begin{equation}
# \theta_{i} = logistic(\boldsymbol{s}_{i,t}^T \boldsymbol{\gamma}),
# \end{equation}where $logistic(s) \equiv 1 / (1 + exp^{-1}(s))$. 
# 
# Similar to the Thompson Sampling algorithm with generalized linear bandits [2], we approximate the posterior distribution of $\boldsymbol{\gamma}$ by its Laplace approximation. Specifically, we approximate the posterior of $\boldsymbol{\gamma}$ as:
# \begin{equation}
#     \begin{split}
#     \tilde{\boldsymbol{\gamma}}^{t} &\sim \mathcal{N}\Big(\hat{\boldsymbol{\gamma}}_{t}, \alpha^2 \boldsymbol{H}_{t}^{-1}\Big),\\
#     \boldsymbol{H}_{t} &= \sum_{t}\mu^{'}(\boldsymbol{S}_{t}^{T}\hat{\boldsymbol{\gamma}}^{t})\boldsymbol{S}_{t}\boldsymbol{S}_{t}^{T},
#     \end{split}
# \end{equation} where $\alpha$ is a pre-specified constant to control the degree of exploration, and $\mu^{'}(\cdot)$ is the derivative of the mean function. It should be noted that the posterior updating step differs for different pairs of the prior distribution of $\boldsymbol{\gamma}$ and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$ by the Laplace approximation;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $\tilde{\boldsymbol{\theta}}$ as $logistic(\boldsymbol{s}_{i,t}^T \tilde{\boldsymbol{\gamma}})$;
# 5. Take the action $A_{t}$ w.r.t $\tilde{\boldsymbol{\theta}}$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$;
# 6. Receive reward $R_{t}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the cascading Bandit problems.

# ## Demo Code

# In[1]:


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.Structured_Bandits.Cascade import CascadeLinTS


# ### Generate the Environment
# 
# Here, we imitate an environment based on the Yelp dataset. The number of items recommended at each round, $K$, is specified as $3$.

# In[3]:


from causaldm.learners.Online.Structured_Bandits.Cascade import _env_realCascade as _env
env = _env.Cascading_env(K = 3, seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - p: number of features (If the intercept is considerd, p includes the intercept as well.)
# - alpha: degree of exploration (default = 1)
# - retrain_freq: frequency to train the generalized linear model (i.e., update every retrain_freq steps)
# - seed: random seed

# In[4]:


K = env.K
L = env.L
p = env.p
alpha = 1
retrain_freq = 1
seed = 0
LinTS_agent = CascadeLinTS.CascadeLinTS(K = K, L = L, p = p, alpha = alpha, 
                                        retrain_freq = retrain_freq, seed = seed)


# ### Recommendation and Interaction
# We fisrt observe the feature information $X$ by
# <code> X = env.Phi </code>. (Note: if an intercept is considered, the X should include a column of ones). Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = LinTS_agent.take_action(X) </code>
# 2. Get the reward from the environment (i.e., $W$, $E$, and $R$)
# <code> W,E,R = env.get_reward(A) </code>
# 3. Update the posterior distribution
# <code> LinTS_agent.receive_reward(A,W,E,t,X) </code>

# In[5]:


t = 0
X = env.Phi
A = LinTS_agent.take_action(X)
W,E,R = env.get_reward(A)
LinTS_agent.receive_reward(A,W,E,t,X)
A, W, E, R


# **Interpretation**: For step 0, the agent decides to display three top restaurants, the first of which is restaurant 1301, the second is restaurant 2087, and the third is restaurant 1123. Unfortunately, the customer does not show any interest in any of the recommended restaurants. As a result, the agent receives a zero reward at round $0$.

# ## References
# 
# [1] Zong, S., Ni, H., Sung, K., Ke, N. R., Wen, Z., & Kveton, B. (2016). Cascading bandits for large-scale recommendation problems. arXiv preprint arXiv:1603.05359.
# 
# [2] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., & Boutilier, C. (2020, June). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics (pp. 2066-2076). PMLR.
