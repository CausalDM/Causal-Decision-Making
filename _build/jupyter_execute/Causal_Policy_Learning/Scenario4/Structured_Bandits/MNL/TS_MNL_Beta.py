#!/usr/bin/env python
# coding: utf-8

# # TS_MNL
# 
# ## Overview
# - **Advantage**: In practice, it always outperforms algorithms that also do not use features but are based on other frameworks, such as UCB.
# - **Disadvantage**: When there are a large number of items, it is not scalable.
# - **Application Situation**: Useful when a list of items is presented, each with a matching price or income, and only one is chosen for each interaction. Binary responses from users include click/don't-click and buy/don't-buy.
# 
# ## Main Idea
# The first TS-based algorithm is developed by [1]. Noticing that direct inference under the standard multinomila logit model is intractable due to the complex dependency of the reward distribution on action slate $a$, an epoch-based algorithmic structure is introduced and is being more popular in recent bandit literature [1,2,3]. Under the epoch-type offering framework,  
# \begin{align}
#     Y_{i}^l(a) &\sim Geometric(\theta_i), \forall i \in a,\\
#     R^l(a) &= \sum_{i\in a}Y_{i}^l(a)\eta_{i}.
# \end{align} 
# Taking the advantage of the nice conjugate relationship between the geometric distribution and the Beta distribution, the TS-based algorithm **TS_MNL** [1] is tractable and computationally efficient. Assuming a Beta prior over parameters $\theta_{i}$, at each epoch $l$, **TS_MNL** updates the posterior distribution of $\theta_{i}$ according to the property of the Beta-Geometric conjugate pair, from which we then sample a $\tilde{\theta}_{i}^{l}$, and $\tilde{v}_{i}^{l}$ is calculated directly as $\tilde{v}_{i}^{l}=1/\tilde{\theta}_{i}^{l}-1$. Finally, the optimal assortment $A^{l}$ is determined efficiently through linear programming [1], such that
# \begin{equation}
#     A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}},
# \end{equation} where $t$ is the first round of epoch $l$.  It should be noted that the posterior updating step differs for different pairs of the prior distribution of $\theta_i$ and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# 
# ## Key Steps
# 1. Specifying a prior distirbution of each $\theta_i$, i.e., Beta(1,1).
# 2. For l = $0, 1,\cdots$:
#     - sample a $\tilde{\theta}^{l}$ from the posterior distribution of $\theta$ or prior distribution if in epoch $0$
#     - compute the utility $\tilde{v}_{i}^{l} = \frac{1}{\tilde{\theta}_{i}^{l}}-1$;
#     - at the first round $t$ of epoch $l$ select top $K$ items by linear programming such that $A^{l} = A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{v}}^{l})$
#     - keep offering $A^{l}$ untile no-purchase appears
#     - receive the rewad $R^l$, and update the posterior distirbution accordingly.
#     
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the Multinomial Logit Bandit problems.
# 

# ## Demo Code

# In[1]:


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.Structured_Bandits.MNL import TS_MNL_Beta


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens dataset.

# In[3]:


from causaldm.learners.Online.Structured_Bandits.MNL import _env_realMNL as _env
env = _env.MNL_env(seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - u_prior_alpha: Alpha of the prior Beta distribution
# - u_prior_beta: Beta of the prior Beta distribution
# - r: revenue of items
# - same_reward: indicate whether the revenue of each item is the same or not
# - clip: indicate whether we clip the $\boldsymbol{\theta}$ to be between $.5$ and $.999$
# - seed: random seed

# In[4]:


L = env.L
K = 5
u_prior_alpha = np.ones(L)
u_prior_beta = np.ones(L)
r = env.r
same_reward = False
clip = True 
seed = 0

TS_agent = TS_MNL_Beta.MNL_TS(L = L, K = K, u_prior_alpha = u_prior_alpha, u_prior_beta = u_prior_beta, 
                              r = r, same_reward = same_reward, clip = clip, seed = seed)


# ### Recommendation and Interaction
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = TS_agent.take_action() </code>
# 3. Get the item clicked and the corresponding revenue from the environment
# <code> c, _, R = env.get_reward(A) </code>
# 4. Update the posterior distribution
# <code> TS_agent.receive_reward(A,c,R) </code>

# In[5]:


t = 0
A = TS_agent.take_action()
c, _, R= env.get_reward(A)
TS_agent.receive_reward(A, c, R)
A, c, R


# **Interpretation**: For step 0, the agent recommends five movies to the customer, the ids of which are 864, 394, 776, 911, and 430. The customer finally clicks the movie 394 and the agent receives a revenue of .033.

# ## References
# 
# [1] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2017, June). Thompson sampling for the mnl-bandit. In Conference on Learning Theory (pp. 76-78). PMLR.
# 
# [2] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2019). Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5), 1453-1485.
# 
# [3] Dong, K., Li, Y., Zhang, Q., & Zhou, Y. (2020, November). Multinomial logit bandit with low switching cost. In International Conference on Machine Learning (pp. 2607-2615). PMLR.

# In[ ]:




