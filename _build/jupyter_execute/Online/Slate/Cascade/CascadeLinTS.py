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
# \theta_{i} = logistic(\boldsymbol{x}_{i,t}^T \boldsymbol{\gamma}),
# \end{equation}where $logistic(x) \equiv 1 / (1 + exp^{-1}(x))$. 
# 
# Similar to the Thompson Sampling algorithm with generalized linear bandits [2], we approximate the posterior distribution of $\boldsymbol{\gamma}$ by its Laplace approximation. Specifically, we approximate the posterior of $\boldsymbol{\gamma}$ as:
# \begin{equation}
#     \begin{split}
#     \tilde{\boldsymbol{\gamma}}^{t} &\sim \mathcal{N}\Big(\hat{\boldsymbol{\gamma}}_{t}, \alpha^2 \boldsymbol{H}_{t}^{-1}\Big),\\
#     \boldsymbol{H}_{t} &= \sum_{t}\mu^{'}(\boldsymbol{X}_{t}^{T}\hat{\boldsymbol{\gamma}}^{t})\boldsymbol{X}_{t}\boldsymbol{X}_{t}^{T},
#     \end{split}
# \end{equation} where $\alpha$ is a pre-specified constant to control the degree of exploration, and $\mu^{'}(\cdot)$ is the derivative of the mean function. It should be noted that the posterior updating step differs for different pairs of the prior distribution of $\boldsymbol{\gamma}$ and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$ by the Laplace approximation;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $\tilde{\boldsymbol{\theta}}$ as $logistic(\boldsymbol{x}_{i,t}^T \tilde{\boldsymbol{\gamma}})$;
# 5. Take the action $A_{t}$ w.r.t $\tilde{\boldsymbol{\theta}}$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$;
# 6. Receive reward $R_{t}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the cascading Bandit problems.

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


from causaldm.learners.Online.Slate.Cascade import CascadeLinTS
from causaldm.learners.Online.Slate.Cascade import _env_Cascade
import numpy as np


# In[3]:


L, T, K, p = 250, 10000, 3, 5
update_freq = 500
update_freq_linear = 500

phi_beta = 1/4
n_init = 500
with_intercept = True
same_reward = True
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_gamma = sigma_gamma = np.identity(p)
mu_gamma = np.zeros(p)
seed = 0

env = _env_Cascade.Cascading_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                    X_mu, X_sigma,                                       
                                    phi_beta, same_reward = same_reward, 
                                    seed = seed, p = p, with_intercept = with_intercept)
LinTS_agent = CascadeLinTS.CascadeLinTS(L = L, K = K, p = p)
S = LinTS_agent.take_action(env.Phi)
t = 1
W, E, exp_R, R = env.get_reward(S)
LinTS_agent.receive_reward(S, W, E, exp_R, R, t, env.Phi)


# In[4]:


S


# ## References
# 
# [1] Zong, S., Ni, H., Sung, K., Ke, N. R., Wen, Z., & Kveton, B. (2016). Cascading bandits for large-scale recommendation problems. arXiv preprint arXiv:1603.05359.
# 
# [2] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., & Boutilier, C. (2020, June). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics (pp. 2066-2076). PMLR.
