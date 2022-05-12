#!/usr/bin/env python
# coding: utf-8

# # Epsilon_Greedy
# 
# ## Main Idea
# 
# An intuitive algorithm to incorporate the exploration and exploitation is $\epsilon$-Greedy, which is simple and widely used [6]. Specifically, at each round $t$, we will select a random action with probability $\epsilon$, and select an action with the highest estimated mean reward based on the history so far with probability $1-\epsilon$. Here the parameter $\epsilon$ is pre-specified. A more adaptive variant is $\epsilon_{t}$-greedy, where the probability of taking a random action is defined as a decreasing function of $t$. Auer et al. [7] showed that $\epsilon_{t}$-greedy performs well in practice with $\epsilon_{t}$ decreases to 0 at a rate of $\frac{1}{t}$.
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
#     2.2. Received the reward R, and update $C$ and $Q$ with
#     \begin{align}
#     C_{A_{t}}^{t+1} &= C_{A_{t}}^{t} + 1 \\
#     \theta_{A_{t}}^{t+1} &=\theta_{A_{t}}^{t} + 1/C_{A_{t+1}}^{t+1}*(R-\theta_{A_{t}}^{t})
#     \end{align}

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


from causaldm.learners.Online.Single import Epsilon_Greedy
from causaldm.learners.Online.Single import Env
import numpy as np


# In[3]:


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

env = Env.Single_Gaussian_Env(T, K, p, phi_beta
                         , mu_theta, Sigma_theta
                        , seed = 42, with_intercept = True
                         , X_mu = X_mu, X_Sigma = X_sigma)
#time-adaptive. for time-fixed version, specifiying epsilon and setting decrease_eps=False
greedy_agent = Epsilon_Greedy.Epsilon_Greedy(K, epsilon = None, decrease_eps = True)
A = greedy_agent.take_action()
t = 0
R = env.get_reward(t,A)
greedy_agent.receive_reward(t,A,R)


# In[9]:


greedy_agent.cnts


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# 
# [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
# 
# [2] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2), 235-256.
