#!/usr/bin/env python
# coding: utf-8

# # $\epsilon$-Greedy
# 
# ## Overview
# - **Advantage**: Simple and easy to understand. Compared to random policy, it makes better use of observations. 
# - **Disadvantage**:  It is difficult to determine an ideal $\epsilon$: if $\epsilon$ is large, exploration will dominate; otherwise, eploitation will dominate. To address this issue, we offer a more adaptive version—$\epsilon_t$-greedy, where $\epsilon_t$ decreases as $t$ increases.
# - **Application Situation**: discrete action space, binary/Gaussian reward space
# 
# ## Main Idea
# $\epsilon$-Greedy is an intuitive algorithm to incorporate the exploration and exploitation. It is simple and widely used [1]. Specifically, at each round $t$, we will select a random action with probability $\epsilon$, and select an action with the highest estimated mean potential reward, $\theta_a$, for each arm $a$ based on the history so far with probability $1-\epsilon$. Specifically,
# $$
# \begin{aligned}
# \theta_a = \hat{E}(R_t(a)|\{A_t, R_t\})
# \end{aligned}
# $$
# 
# For example, in movie recommendation, the agent would either recommend a random genere of movies to the user or recommend the genere that the user watched the most in the past. Here the parameter $\epsilon$ is pre-specified. A more adaptive variant is $\epsilon_{t}$-greedy, where the probability of taking a random action is defined as a decreasing function of $t$. Auer et al. [2] showed that $\epsilon_{t}$-greedy performs well in practice with $\epsilon_{t}$ decreases to 0 at a rate of $\frac{1}{t}$. Note that, the reward can be either binary or continuous.
# 
# 
# ## Algorithms Details
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. The $\epsilon$-greedy algorithm start with initializing the estimated values $\theta_a^0$ and the count of being pulled $C_a^0$ for each action $a$ as 0. At each round $t$, we either take an action with the maximum estimated value $\theta_a$ with probability $1-\epsilon_{t}$ or randomly select an action with probability $\epsilon_t$. After observing the rewards corresponding to the selected action $A_t$, we updated the total number of being pulled for $A_t$, and estimated the $\theta_{A_{t}}$ by with the sample average for $A_t$.
# 
# Remark that both the time-adaptive and the time-fixed version of $\epsilon$-greedy algorithm are provided. By setting **decrease_eps=True**, the $\epsilon_{t}$ in round $t$ is calculated as $\frac{K}{T}$. Otherwise, $\epsilon_{t}$ is a fixed value specfied by users.
# 
# ## Key Steps
# 
# 1. Initializing the $\boldsymbol{\theta}^0$ and $\boldsymbol{C}^0$ for $K$ items as 0
# 2. For t = $0, 1,\cdots, T$:
# 
#     2.1. select action $A_t$ as the arm with the maximum $\theta_a^t$ with probability $1-\epsilon_t$, or randomly select an action $A_t$ with probability $\epsilon_t$
#     
#     2.2. Received the reward $R_t$, and update $C$ and $Q$ with
#     \begin{align}
#     C_{A_{t}}^{t+1} &= C_{A_{t}}^{t} + 1 \\
#     \theta_{A_{t}}^{t+1} &=\theta_{A_{t}}^{t} + \frac{1}{C_{A_{t+1}}^{t+1}}*(R_t-\theta_{A_{t}}^{t})
#     \end{align}

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.MAB import Epsilon_Greedy


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[2]:


from causaldm.learners.CPL4.MAB import _env_realMAB as _env
env = _env.Single_Gaussian_Env(seed = 42)


# ### Specify Hyperparameters
# 
# - K: # of arms
# - epsilon: fixed $\epsilon$ for time-fixed version of $\epsilon$-greedy algorithm
# - decrease_eps: indicate if a time-adaptive $\epsilon_t = min(1,\frac{K}{t})$ employed.

# In[3]:


K = env.K
greedy_agent = Epsilon_Greedy.Epsilon_Greedy(K, epsilon = None, decrease_eps = True, seed = 0)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action 
# <code> A = greedy_agent.take_action() </code>
# 2. Get the reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 3. Update the posterior distribution
# <code> greedy_agent.receive_reward(t,A,R) </code>

# In[4]:


t = 0
A = greedy_agent.take_action()
R = env.get_reward(A)
greedy_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the $\epsilon-$greedy agent recommend a Thriller (arm 3), and received a rate of 4 from the environment.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit.

# In[5]:


env = _env.Single_Bernoulli_Env(seed=42)

K = env.K
greedy_agent = Epsilon_Greedy.Epsilon_Greedy(K, epsilon = None, decrease_eps = True, seed = 42)

t = 0
A = greedy_agent.take_action()
R = env.get_reward(A)
greedy_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the $\epsilon-$greedy agent recommend a Comedy (arm 0), and received a reward of 1 from the environment.

# ## References
# 
# [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
# 
# [2] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2), 235-256.
