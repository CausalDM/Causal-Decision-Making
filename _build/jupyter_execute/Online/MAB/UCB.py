#!/usr/bin/env python
# coding: utf-8

# # UCB
# 
# ## Overview
# - **Advantage**: There is no need to specify any hyper-parameters. It carefully quantifies the uncertainty of the estimation to better combine exploration and exploitation. 
# - **Disadvantage**: Inefficient if there is a large number of action items. 
# - **Application Situation**: discrete action space, binary/Gaussian reward space
# 
# ## Main Idea
# As the name suggested, the UCB algorithm estimates the upper confidence bound $U_{a}^{t}$ of the mean of the potential reward of arm $a$, $R_t(a)$, based on the observations and then choose the action has the highest estimates. The class of UCB-based algorithms is firstly introduced by Auer et al. [1]. Generally, at each round $t$, $U_{a}^{t}$ is calculated as the sum of the estimated reward (exploitation) and the estimated confidence radius (exploration) of item $i$ based on previous observations. Then, $A_{t}$ is selected as 
# $$
# \begin{equation}
#     A_t = argmax_{a \in \mathcal{A}} U_a^t.
# \end{equation} 
# $$
# As an example, **UCB** [1] estimates the confidence radius as $\sqrt{\frac{2log(t)}{\text{\# item $i$ played so far}}}$. Doing so, either the item with a large average reward or the item with limited exploration will be selected. Note that this algorithm support cases with either binary reward or continuous reward.
# 
# ## Algorithms Details
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. The UCB1 algorithm start with initializing the estimated upper confidence bound $U_a^{0}$ and the count of being pulled $C_a^{0}$ for each action $a$ as 0. At each round $t$, we greedily select an action $A_t$ as 
# \begin{align}
# A_t = arg max_{a\in \mathcal{A}} U_{a}^{t}.
# \end{align}
# 
# After observing the rewards corresponding to the selected action $A_t$, we first update the total number of being pulled for $A_t$ accordingly. Then, we estimate the upper confidence bound for each action $a$ as
# \begin{align}
# U_{a}^{t+1} = \frac{1}{C_a^{t+1}}\sum_{t'=0}^{t}R_{t'}I(A_{t'}=a) + \sqrt{\frac{2*log(t+1)}{C_a^{t+1}}} ,
# \end{align}where $R_{t'}$ is the reward received at round $t'$. Intuitively, $U_{a}^{t}$ is the sum of the sample average reward of action $a$ for expolitation and a confidence radius for exploration.
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


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[13]:


import numpy as np
from causaldm.learners.Online.MAB import UCB


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[14]:


from causaldm.learners.Online.MAB import _env_realMAB as _env
env = _env.Single_Gaussian_Env(seed = 42)


# ### Specify Hyperparameters
# 
# - K: # of arms

# In[15]:


UCB_agent = UCB.UCB1(env.K)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action 
# <code> A = UCB_agent.take_action() </code>
# 2. Get the reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 3. Update the posterior distribution
# <code> UCB_agent.receive_reward(t,A,R) </code>

# In[16]:


t = 0
A = UCB_agent.take_action()
R = env.get_reward(A)
UCB_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the UCB agent recommend a Comedy (arm 0), and received a rate of 2 from the environment.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit.

# In[18]:


env = _env.Single_Bernoulli_Env(seed=42)

UCB_agent = UCB.UCB1(env.K)

t = 0
A = UCB_agent.take_action()
R = env.get_reward(A)
UCB_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the UCB agent recommend a Comedy (arm 0), and received a reward of 0 from the environment.

# ## References
# 
# [1] Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256.
