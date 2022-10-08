#!/usr/bin/env python
# coding: utf-8

# # MTSS_Cascade
# 
# ## Overview
# - **Advantage**: It is both scalable and robust. Furthermore, it also accounts for the iter-item heterogeneity.
# - **Disadvantage**:
# - **Application Situation**: Useful when presenting a ranked list of items, with only one selected at each interaction. The outcome is binary. Static feature information.
# 
# ## Main Idea
# MTSS_Cascade is an example of the general Thompson Sampling(TS)-based framework, MTSS [1], to deal with online learning to rank problems.
# 
# **Review of MTSS:** MTSS[1] is a meta-learning framework designed for large-scale structured bandit problems [2]. Mainly, it is a TS-based algorithm that learns the information-sharing structure while minimizing the cumulative regrets. Adapting the TS framework to a problem-specific Bayesian hierarchical model, MTSS simultaneously enables information sharing among items via their features and models the inter-item heterogeneity. Specifically, it assumes that the item-specific parameter $\theta_i = E[W_t(i)]$ is sampled from a distribution $g(\theta_i|\boldsymbol{x}_i, \boldsymbol{\gamma})$ instead of being entirely determined by $\boldsymbol{x}_i$ via a deterministic function. Here, $g$ is a model parameterized by an **unknown** vector $\boldsymbol{\gamma}$. The following is the general feature-based hierarchical model MTSS considered. 
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
# **Review of MTSS_Cascade:** To characterize the relationship between items using their features, one example choice of $g$ is the popular Beta-Bernoulli logistic model [1,2], where $\theta_i {\sim} Beta(logistic(\boldsymbol{x}_i^T \boldsymbol{\gamma}), \psi)$ for some known parameter $\psi$. We adopt the mean-precision parameterization of the Beta distribution, with $logistic(\boldsymbol{x}_i^T \boldsymbol{\gamma})$ being the mean and $\psi$ being the precision parameter. Specifically, the full model is as follows: 
# \begin{equation}\label{eqn:model_cascading}
#     \begin{split}
#     \theta_i & \sim Beta(logistic(\boldsymbol{x}_i^T \boldsymbol{\gamma}), \psi), \forall i \in [N]\\
#     W_{k, t}(a) &\sim Bernoulli(\theta_{a^k}), \forall k \in [K], \\
#     Y_{k,t}(a) &= W_{k,t}(a) E_{k,t}(a), \forall k \in [K],\\
#     E_{k,t}(a) &= [1-Y_{k-1}(a)] E_{k-1,t}(a), \forall k \in [K],\\
#     R_t(a) &= \sum_{k \in [K]} Y_{k,t}(a), 
#     \end{split}
# \end{equation} with $E_{1,t}(a) \equiv 1$.   
# The prior $Q(\boldsymbol{\gamma})$ can be chosen as many appropriate distributions. For instance, we choose the prior $\boldsymbol{\gamma} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\gamma}}, {\boldsymbol{\Sigma}}_{\boldsymbol{\gamma}})$ with parameters as known. To update the posterior of $\boldsymbol{\gamma}$, we utilize the **Pymc3** package [3]. With a given $\boldsymbol{\gamma}$, the posterior of $\boldsymbol{\theta}$ enjoys the Beta-Geometric conjugate relationship and hence can be updated explicitly and efficiently. Finally, for each round, we select the top $K$ items with the highest estimated attractiveness factors. It should be noted that the posterior updating step differs for different pairs of the $Q(\boldsymbol{\gamma})$ and the reward distribution, and the corresponding code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# 
# ## Algorithm Details
# At each round $t$, given the feedback $\mathcal{H}_{t}$ received from previous rounds, there are two major steps including posterior sampling and combinatorial optimization. Specifically, the posterior sampling step is decomposed into four steps: 1. approximating a posterior distribution of $\boldsymbol{\gamma}$, $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$, by **Pymc3**; 2. sampling a $\tilde{\boldsymbol{\gamma}}$ from $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$; 3. updating the posterior distribution of $\boldsymbol{\theta}$ conditional on $\tilde{\boldsymbol{\gamma}}$, $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$, which has an explicit form under the assumption of a Beta-Bernoulli logistic model; 4. sampling $\tilde{\boldsymbol{\theta}}$ from $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$. Then, the action $A_{t}$ is selected greedily as $A_t = arg max_{a \in \mathcal{A}} E(R_t(a) \mid \tilde{\boldsymbol{\theta}})$. Specifically, the top $K$ items with the highest $\tilde{\theta}_{i}$ will be displayed. Note that $\tilde{\boldsymbol{\gamma}}$ can be sampled in a batch mode to further facilitate computationally efficient online deployment.
# 
# ## Key Steps
# For round $t = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{t})$ by **Pymc3**;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{t})$;
# 3. Update $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$;
# 4. Sample $\tilde{\boldsymbol{\theta}} \sim P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{t})$;
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


from causaldm.learners.Online.Slate.Cascade import MTSS_Cascade
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
MTSS_agent = MTSS_Cascade.MTSS_Cascade(phi_beta = phi_beta, K = K
                                              , gamma_prior_mean = mu_gamma, gamma_prior_cov = Sigma_gamma
                                              , Xs = env.Phi
                                              , update_freq = update_freq, n_init = n_init)
S = MTSS_agent.take_action(env.Phi)
t = 1
W, E, exp_R, R = env.get_reward(S)
MTSS_agent.receive_reward(S, W, E, exp_R, R, t, env.Phi)


# In[4]:


S


# ## References
# 
# [1] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [2] Forcina, A. and Franconi, L. Regression analysis with the beta-binomial distribution. Rivista di Statistica Applicata, 21(1), 1988.
# 
# [3] Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 DOI: 10.7717/peerj-cs.55.

# In[ ]:




