#!/usr/bin/env python
# coding: utf-8

# # MTSS_Comb
# 
# ## Overview
# - **Advantage**: It is both scalable and robust. Furthermore, it also accounts for the iter-item heterogeneity.
# - **Disadvantage**:
# - **Application Situation**: Useful when presenting a list of items, each of which will generate a partial outcome (reward). The outcome is continuous. Static feature information.
# 
# ## Main Idea
# MTSS_Comb is an example of the general Thompson Sampling(TS)-based framework, MTSS [1], to deal with online combinatorial optimization problems.
# 
# **Review of MTSS:** MTSS[1] is a meta-learning framework designed for large-scale structured bandit problems [2]. Mainly, it is a TS-based algorithm that learns the information-sharing structure while minimizing the cumulative regrets. Adapting the TS framework to a problem-specific Bayesian hierarchical model, MTSS simultaneously enables information sharing among items via their features and models the inter-item heterogeneity. Specifically, it assumes that the item-specific parameter $\theta_i = E[Y_{t}(i)]$ is sampled from a distribution $g(\theta_i|\boldsymbol{x}_i, \boldsymbol{\gamma})$ instead of being entirely determined by $\boldsymbol{x}_i$ via a deterministic function. Here, $g$ is a model parameterized by an **unknown** vector $\boldsymbol{\gamma}$. The following is the general feature-based hierarchical model MTSS considered. 
# \begin{equation}\label{eqn:general_hierachical}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad\quad\quad\quad\quad\quad\quad
# \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}),\\
# &\text{(Generalization function)} \;
# \;    \theta_i| \boldsymbol{x}_i, \boldsymbol{\gamma}  &&\sim g(\theta_i|\boldsymbol{x}_i, \boldsymbol{\gamma}), \forall i \in [N],\\ 
# &\text{(Observations)} \quad\quad\quad\quad\quad\quad\;
# \;    \boldsymbol{Y}_t(a) &&\sim f(\boldsymbol{Y}_t(a)|\boldsymbol{\theta}),\\
# &\text{(Reward)} \quad\quad\quad\quad\quad\quad\quad\quad\;
# \;   R_t(a) &&= f_r(\boldsymbol{Y}_t(a) ; \boldsymbol{\eta}), 
#       \end{alignedat}
# \end{equation}
# where $Q(\boldsymbol{\gamma})$ is the prior distribution for $\boldsymbol{\gamma}$. 
# Overall, MTTS is a **general** framework that subsumes a wide class of practical problems, **scalable** to large systems, and **robust** to the specification of the generalization model.
# 
# **Review of MTSS_Comb:** In this tutorial, as an example, we focus on the combinatorial semi-bandits with Gaussian outcome, $Y_{i, t}$, and consider using a linear mixed model (LMM) as the generalization model to share information. Specifically, the full model is as follows: 
# \begin{equation}\label{eqn:LMM}
#     \begin{split}
#      \theta_i &\sim \mathcal{N}(\boldsymbol{x}_i^T \boldsymbol{\gamma}, \sigma_1^2), \forall i \in [N],\\
#     Y_{i, t}(a) &\sim \mathcal{N}(\theta_i, \sigma_2^2), \forall i \in a,\\
#     R_t(a) &= \sum_{i \in a} Y_{i,t}(a), 
#     \end{split}
# \end{equation}
# where it is typically assumed that $\sigma_1$ and $\sigma_2$ are known. We choose the prior $\boldsymbol{\gamma} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\gamma}}, {\boldsymbol{\Sigma}}_{\boldsymbol{\gamma}})$ with parameters as known. It is worth noting that many other outcome distributions (e.g., Bernoulli) and model assumptions (e.g., Gaussian process) can be formulated similarly, depending on the applications. Users can directly modify the code for posterior updating to adapt different model assumptions. Further, for simplicity, we consider the most basic size constraint such that the action space includes all the possible subsets with size $K$. Therefore, the optimization process to find the optimal subset $A_{t}$ is equal to selecting a list of $K$ items with the highest attractiveness factors. Of course, users are welcome to modify the **optimization** function to satisfy more complex constraints.
# 
# ## Algorithm Details
# Under the assumption of a LMM, the posteriors can be derived explicitly, following the Bayes' theorem. At each round $t$, given the feedback $\mathcal{H}_{t}$ received from previous rounds, there are two major steps including posterior sampling and combinatorial optimization. Specifically, the posterior sampling step is decomposed into four steps: 1. updating the posterior distribution of $\boldsymbol{\gamma}$, $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$; 2. sampling a $\tilde{\boldsymbol{\gamma}}$ from $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$; 3. updating the posterior distribution of $\boldsymbol{\theta}$ conditional on $\tilde{\boldsymbol{\gamma}}$, $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$; 4. sampling $\tilde{\boldsymbol{\theta}}$ from $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$. Then, the action $A_{t}$ is selected greedily as $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$. Considering the simple size constraint, $A_{t}$ is the list of $K$ items with the highest $\tilde{\theta}_{i}$. Note that $\tilde{\boldsymbol{\gamma}}$ can be sampled in a batch mode to further facilitate computationally efficient online deployment.
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Update $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$;
# 4. Sample $\tilde{\boldsymbol{\theta}} \sim P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$;
# 5. Take the action $A_{t}$ w.r.t $\tilde{\boldsymbol{\theta}}$ such that $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$;
# 6. Receive reward $R_{t}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the combinatorial Semi-Bandit problems.

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


from causaldm.learners.Online.Slate.Combinatorial_Semi import MTSS_Comb
from causaldm.learners.Online.Slate.Combinatorial_Semi import _env_SemiBandit
import numpy as np


# In[5]:


L, T, K, p = 300, 1000, 5, 3
mu_gamma = np.zeros(p)
sigma_gamma = np.identity(p)
X_mu = np.zeros(p-1)
X_sigma = np.identity(p-1)
with_intercept = True
seed = 0
sigma_1 = .5
sigma_2 = 1

env = _env_SemiBandit.Semi_env(L, K, T, p, sigma_1, sigma_2
                               , mu_gamma, sigma_gamma, seed = seed
                               , with_intercept = with_intercept
                               , X_mu = X_mu, X_sigma = X_sigma)
MTSS_agent = MTSS_Comb.MTSS_Semi(sigma_2 = 1, L=L, T = T
                                 , gamma_prior_mean = np.zeros(p), gamma_prior_cov = np.identity(p)
                                 , sigma_1 = sigma_1
                                 , K = K
                                 , Xs = env.Phi# [L, p]
                                 , update_freq = 1)
S = MTSS_agent.take_action(env.Phi)
t = 1
obs_R, exp_R, R = env.get_reward(S, t)
MTSS_agent.receive_reward(t, S, obs_R, X = env.Phi)


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ## References
# [1] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# [2] Van Parys, B., & Golrezaei, N. (2020). Optimal learning for structured bandits. Available at SSRN 3651397.
