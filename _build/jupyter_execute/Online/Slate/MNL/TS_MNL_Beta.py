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
# [1] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2017, June). Thompson sampling for the mnl-bandit. In Conference on Learning Theory (pp. 76-78). PMLR.
# 
# [2] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2019). Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5), 1453-1485.
# 
# [3] Dong, K., Li, Y., Zhang, Q., & Zhou, Y. (2020, November). Multinomial logit bandit with low switching cost. In International Conference on Machine Learning (pp. 2607-2615). PMLR.

# In[ ]:




