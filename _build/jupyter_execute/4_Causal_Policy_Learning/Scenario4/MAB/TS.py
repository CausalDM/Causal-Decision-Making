#!/usr/bin/env python
# coding: utf-8

# # TS
# 
# ## Overview
# - **Advantage**: Be able to incorporate prior knowledge about reward distribution, which is especially useful when the prior knowledge is informative. Taking uncertainty into account by updating the posterior distribution of the expectation of potential reward to achieve a good balance between exploration and exploitation.
# - **Disadvantage**: Inefficient if there is a large number of action items. 
# - **Application Situation**: discrete action space, binary/Gaussian reward space
# 
# ## Main Idea
# 
# Thompson Sampling (TS), also known as posterior sampling, solves the exploration-exploitation dilemma by selecting an action according to its posterior distribution [1].  At each round $t$, the agent sample the rewards from the corresponding posterior distributions of the expectation of the potential reward (i.e., $E[R_t(a)]$) and then select the action with the highest sampled reward greedily. It has been shown that, when the true reward distribution is known, a TS algorithm with the true reward distribution as the prior is nearly optimal [2]. However, such a distribution is always unknown in practice. Therefore, one of the major objectives of TS-based algorithms is to find an informative prior to guide the exploration. Note that the algorithm here supports bandit problem with either binary reward or continuous reward.
# 
# ## Algorithms Details
# Supposed there are $K$ options, and the action space is $\mathcal{A} = \{0,1,\cdots, K-1\}$. The TS algorithm starts with specifying a prior distribution of the potential reward, based on the domian knowledge. At each round $t$, the agent will samples a vector of $\boldsymbol{\theta}^{t}$ from the posterior distribution of the potential rewards. The action $a$ with the greatest $\theta_{a}^{t}$ is then selected. Finally, the posterior distribution would be updated after receiving the realized reward at the end of each round. Note that the posterior updating step differs for different pairs of prior distribution of the mean reward and reward distribution. Here, we consider two classical examples of the TS algorithm, including
# 
# - Gaussian Bandits
# \begin{aligned}
# \boldsymbol{\theta} &\sim Q(\boldsymbol{\theta}),\\
# R_t(a) &\sim \mathcal{N}(\theta_a,\sigma^2),
# \end{aligned}
# - Bernoulli Bandits
# \begin{aligned}
# \boldsymbol{\theta} &\sim Q(\boldsymbol{\theta}),\\
# R_t(a) &\sim Bernoulli(\theta_a).
# \end{aligned}
# 
# Assuming a Gaussian prior for the Gaussian bandits and a Beta prior for the Bernoulli bandits, the posterior updating is straightforward with closed-form expression. In the Gaussian bandits, the variance of reward $\sigma^2$ is assumed to be known, and need to be specified manually. Note that code can be easily modified to different specifications of the prior/potential reward distribution.
# 
# ## Key Steps
# 
# 1. Specifying a prior distirbution of $E[R_0(a)]$, $a \in \mathcal{A}$, and the variance of the reward distribution.
# 2. For t = $0, 1,\cdots, T$:
#     - sample a $\boldsymbol{\theta}^{t}$ from the posterior distribution of $E[R_t(a)]$ or prior distribution if in round $0$
#     - select action $A_t$ which has the greatest $\theta^{t}_{a}$, i.e. $A_t = \arg\max_{a \in \mathcal{A}} \theta_{a}^{t}$
#     - receive the rewad $R_t$, and update the posterior distirbution accordingly.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.MAB import TS


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[2]:


from causaldm.learners.CPL4.MAB import _env_realMAB as _env
env = _env.Single_Gaussian_Env(seed = 42)


# ### Specify Hyperparameters
# 
# - Reward_Type: the type of the MAB, i.e., "Gaussian"/"Bernoulli"
# - sigma: the standard deviation of the reward distributions
# - u_prior_mean: mean of the prior distribution of the mean reward
# - u_prior_cov: Covaraince matrix of the prior distribution of the mean reward
# - seed: random seed

# In[7]:


Reward_Type = "Gaussian"
sigma = 1
K = env.K
u_prior_mean = np.ones(K)
u_prior_cov = 10000*np.identity(K)
seed = 0
TS_Gaussian_agent = TS.TS(Reward_Type = Reward_Type, sigma = sigma, 
                          u_prior_mean = u_prior_mean, u_prior_cov = u_prior_cov, 
                          seed = seed)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action 
# <code> A = TS_Gaussian_agent.take_action() </code>
# 2. Get the reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 3. Update the posterior distribution
# <code> TS_Gaussian_agent.receive_reward(t,A,R) </code>

# In[8]:


t = 0
A = TS_Gaussian_agent.take_action()
R = env.get_reward(A)
TS_Gaussian_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the TS agent recommend a Thriller (arm 3), and received a rate of 2 from the environment.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit. Note that, when specifying the prior distribution of the expected reward, the mean-precision form of the Beta distribution is used here, i.e., Beta($\mu$, $\phi$), where $\mu$ is the mean reward of each arm and $\phi$ is the precision of the Beta distribution. 

# In[13]:


env = _env.Single_Bernoulli_Env(seed=42)

K = env.K
Reward_Type = "Bernoulli"
## specify the mean of the prior beta distribution
u_prior_mean = .5*np.ones(K)
## specify the precision of the prior beta distribution
prior_phi_beta = 1
TS_Bernoulli_agent = TS.TS(Reward_Type = Reward_Type,
                           u_prior_mean = u_prior_mean,
                           prior_phi_beta = prior_phi_beta,
                           seed = seed)
t = 0
A = TS_Bernoulli_agent.take_action()
R = env.get_reward(A)
TS_Bernoulli_agent.receive_reward(t,A,R)
t, A, R


# **Interpretation**: For step 0, the TS agent recommend a Sci-Fi (arm 4), and received a reward of 0 from the environment.

# ## References
# [1] Russo, D., Van Roy, B., Kazerouni, A., Osband, I., and Wen, Z. (2017). A tutorial on thompson sampling. arXiv preprint arXiv:1707.0203
# 
# [2] Lattimore, T. and Szepesv´ari, C. (2020). Bandit algorithms. Cambridge University Press.
