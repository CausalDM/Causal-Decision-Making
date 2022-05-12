#!/usr/bin/env python
# coding: utf-8

# # UCB1
# 
# ## Main Idea
# As the name suggested, the UCB algorithm estimates the upper confidence bound $U_{i}^{t}$ of the mean rewards based on the observations and then choose the action has the highest estimates. The class of UCB-based algorithms is firstly introduced by Auer et al. [2]. Generally, at each round $t$, $U_{i}^{t}$ is calculated as the sum of the estimated reward (exploitation) and the estimated confidence radius (exploration) of item $i$ based on $\mathcal{H}_{t}$. Then, $A_{t}$ is selected as 
# \begin{equation}
#     A_t = argmax_{a \in \mathcal{A}} E(R_t \mid a,\{ U_{i}^{t}\}_{i=1}^{N}).
# \end{equation} 
# 
# Doing so, either the item with a large average reward or the item with limited exploration will be selected.
# 
# 
# ## Algorithms Details
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. The UCB1 algorithm start with initializing the estimated upper confidence bound $U_a^{0}$ and the count of being pulled $C_a^{0}$ for each action $a$ as 0. At each round $t$, we greedily select an action $A_t$ as 
# \begin{align}
# A_t = arg max_{a\in \mathcal{A}} U_{a}^{t}.
# \end{align}
# 
# After observing the rewards corresponding to the selected action $A_t$, we first update the total number of being pulled for $A_t$ accordingly. Then, we estimate the upper confidence bound for each action $a$ as
# \begin{align}
# U_{a}^{t+1} = \frac{1}{C_a^{t+1}}\sum_{t'=0}^{t}R_{t'}I(A_{t'}=a) + \sqrt{\frac{2*log(t+1)}{C_a^{t+1}}} 
# \end{align}, where $R_{t'}$ is the reward received at round $t'$. Intuitively, $U_{a}^{t}$ is the sum of the sample average reward of action $a$ for expolitation and a confidence radius for exploration.
# 
# ## Key Steps
# 
# 1. Initializing the $\boldsymbol{U}^0$ and $\boldsymbol{C}^0$ for $K$ items as 0
# 2. For t = $0, 1,\cdots, T$:
# 
#     2.1. select action $A_t$ as the arm with the maximum $U_a^t$
#     
#     2.2. Received the reward R, and update $C$ and $U$ with
#     \begin{align}
#     C_{A_{t}}^{t+1} &= C_{A_{t}}^{t} + 1 \\
#     U_{A_{t}}^{t+1} &= \frac{1}{C_a^{t+1}}\sum_{t'=0}^{t}R_{t'}I(A_{t'}=a) + \sqrt{\frac{2*log(t+1)}{C_a^{t+1}}} 
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


from causaldm.learners.Online.Single import UCB1
from causaldm.learners.Online.Single import Env
import numpy as np


# In[4]:


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
UCB_agent = UCB1.UCB1(K)
A = UCB_agent.take_action()
t = 0
R = env.get_reward(t,A)
UCB_agent.receive_reward(t,A,R)


# In[5]:


UCB_agent.Rs


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# 
# [1] Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on thompson sampling. Foundations and Trends® in Machine Learning, 11(1), 1-96.
# 
# [2] Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256.
# 

# In[ ]:




