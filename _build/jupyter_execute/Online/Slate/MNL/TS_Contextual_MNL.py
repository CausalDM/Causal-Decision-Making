#!/usr/bin/env python
# coding: utf-8

# # TS_Contextual_MNL
# 
# ## Overview
# - **Advantage**: It is scalable when the features are used. It outperforms algorithms based on other frameworks, such as UCB, in practice.
# - **Disadvantage**: It is susceptible to model misspecification.
# - **Application Situation**: Useful when a list of items is presented, each with a matching price or income, and only one is chosen for each interaction. Binary responses from users include click/don't-click and buy/don't-buy.
# 
# 
# ## Main Idea
# Feature-determined approaches have been developed recently to provide a more feasible approach for large-scale problems, by adapting either the UCB framwork or the TS framework. While all of them [1,2,3] are under the standard offering structure, here we modify the TS-type algorithm in [3] by adapting to the epoch-type offering framework and assuming a linear realtionship between the utility and the item features as 
# \begin{equation}
# \theta_i = \frac{logistic(\boldsymbol{x}_{i,t}^T \boldsymbol{\gamma})+ 1}{2},
# \end{equation} to tackle the challenge of a large item space. We named the proposed algorithm as **TS_Contextual_MNL**. At each decision round $t$, **TS_Contextual_MNL** samples $\tilde{\boldsymbol{\gamma}}_{t}$ from the posterior distribution, which is updated by **Pymc3**, and get the $\tilde{\theta}_{i}^{t}$ as $\frac{logistic(\boldsymbol{x}_{i,t}^T \text{ }\tilde{\boldsymbol{\gamma}})+ 1}{2}$ and $\tilde{v}_{i}^{l}$ as $1/\tilde{\theta}_{i}^{l}-1$. Finally, linear programming is employed to determine the optimal assortment $A^{l}$, such that
# \begin{equation}
#     A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}},
# \end{equation} where $t$ is the first round of epoch $l$.  
# 
# It should be noted that the posterior updating step differs for different pairs of the prior distribution of $\boldsymbol{\gamma}$ and the reward distribution, and the code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# ## Key Steps
# For epoch $l = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}^{l})$ by **Pymc3**;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}^{l})$;
# 3. Update $\tilde{\boldsymbol{\theta}} = \frac{logistic(\boldsymbol{x}_{i,t}^T \text{ }\tilde{\boldsymbol{\gamma}})+ 1}{2}$
# 4. Compute the utility $\tilde{v}_{i} = \frac{1}{\tilde{\theta}_{i}}-1$;
# 5. Take the action $A^{l}$ w.r.t $\{\tilde{v}_{i}\}_{i=1}^{N}$ such that $A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}}$;
# 6. Offer $A^{l}$ until no purchase appears;
# 7. Receive reward $R^{l}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the Multinomial Logit Bandit problems.

# ## Demo Code

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
# code used to import the learner


# In[9]:


from causaldm.learners.Online.Slate.MNL import MTSS_MNL
from causaldm.learners.Online.Slate.MNL import _env_MNL
import numpy as np


# In[10]:


T = 20000
L = 1000
update_freq = 500
update_freq_linear = 500

phi_beta = 1/4
n_init = 500
with_intercept = True
same_reward = True
p=3
K=5
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
Sigma_gamma = sigma_gamma = np.identity(p)
mu_gamma = np.zeros(p)
seed = 0

env = _env_MNL.MNL_env(L, K, T, mu_gamma, sigma_gamma, X_mu, X_sigma,                                       
                        phi_beta, same_reward = same_reward, 
                        seed = seed, p = p, with_intercept = with_intercept)
MTSS_agent = MTSS_MNL.MTSS_MNL(L, env.r, K, env.Phi, phi_beta = phi_beta,n_init = n_init,
                                    gamma_prior_mean = mu_gamma, gamma_prior_cov = Sigma_gamma,
                                    update_freq=update_freq, seed = seed, pm_core = 1, same_reward = same_reward, clip = True)
S = MTSS_agent.take_action(env.Phi)
t = 1
c, exp_R, R = env.get_reward(S)
MTSS_agent.receive_reward(S, c, R, exp_R)


# In[11]:


S


# ## References
# 
# [1] Ou, M., Li, N., Zhu, S., & Jin, R. (2018). Multinomial logit bandit with linear utility functions. arXiv preprint arXiv:1805.02971.
# 
# [2] Agrawal, P., Avadhanula, V., & Tulabandhula, T. (2020). A tractable online learning algorithm for the multinomial logit contextual bandit. arXiv preprint arXiv:2011.14033.
# 
# [3] Oh, M. H., & Iyengar, G. (2019). Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32.
# 

# In[ ]:




