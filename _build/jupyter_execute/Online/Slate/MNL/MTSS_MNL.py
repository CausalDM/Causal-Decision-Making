#!/usr/bin/env python
# coding: utf-8

# # MTSS_MNL
# 
# ## Main Idea
# MTSS_MNL is an example of the general Thompson Sampling(TS)-based framework, MTSS [1], to deal with dynamic assortment optimization problems.
# 
# **Review of MTSS:** MTSS[1] is a meta-learning framework designed for large-scale structured bandit problems [2]. Mainly, it is a TS-based algorithm that learns the information-sharing structure while minimizing the cumulative regrets. Adapting the TS framework to a problem-specific Bayesian hierarchical model, MTSS simultaneously enables information sharing among items via their features and models the inter-item heterogeneity. Specifically, it assumes that the item-specific parameter $\theta_i$ is sampled from a distribution $g(\theta_i|\boldsymbol{x}_i, \boldsymbol{\gamma})$ instead of being entirely determined by $\boldsymbol{x}_i$ via a deterministic function. Here, $g$ is a model parameterized by an **unknown** vector $\boldsymbol{\gamma}$. The following is the general feature-based hierarchical model MTSS considered. 
# \begin{equation}\label{eqn:general_hierachical}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad\quad\quad\quad\quad\quad\quad
# \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}),\\
# &\text{(Generalization function)} \;
# \;    \theta_i| \boldsymbol{x}_i, \boldsymbol{\gamma}  &&\sim g(\theta_i|\boldsymbol{x}_i, \boldsymbol{\gamma}), \forall i \in [N],\\ 
# &\text{(Observations)} \quad\quad\quad\quad\quad\quad\;
# \;    \boldsymbol{Y}_t &&\sim f(\boldsymbol{Y}_t|A_t, \boldsymbol{\theta}),\\
# &\text{(Reward)} \quad\quad\quad\quad\quad\quad\quad\quad\;
# \;   R_t &&= f_r(\boldsymbol{Y}_t ; \boldsymbol{\eta}), 
#       \end{alignedat}
# \end{equation}
# where $Q(\boldsymbol{\gamma})$ is the prior distribution for $\boldsymbol{\gamma}$. 
# Overall, MTTS is a **general** framework that subsumes a wide class of practical problems, **scalable** to large systems, and **robust** to the specification of the generalization model.
# 
# **Review of MTSS_MNL:** In this tutorial, as a concrete example, we focus on the epoch-type offering schedule and consider modeling the relationship between $\theta_i$ and $\boldsymbol{x}_i$ with the following Beta-Geometric logistic model:
# \begin{equation}\label{eqn1}
#     \begin{split}
#      \theta_i &\sim Beta \big(\frac{logistic(\boldsymbol{x}_i^T \boldsymbol{\gamma})+ 1}{2}, \psi \big) , \forall i \in [N],\\
#     Y_{i}^l &\sim Geometric(\theta_i), \forall i \in A^l,\\
#     R_l &= \sum_{i\in A^{l}}Y_{i}^l\eta_{i},
#     \end{split}
# \end{equation}where we adopt the mean-precision parameterization of the Beta distribution and $\psi$ and $\boldsymbol{\eta}$ are known. We choose this specific form as it is widely observed that $v_i < 1$ [2,3], i.e., no item is more popular than the no-purchase option. This is equal to $\theta_i \in (1/2, 1)$. Other generalization models are also possible with minor modifications to the posterior sampling code. 
# The prior $Q(\boldsymbol{\gamma})$ can be chosen as many appropriate distributions. For instance, we choose the prior $\boldsymbol{\gamma} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\gamma}}, {\boldsymbol{\Sigma}}_{\boldsymbol{\gamma}})$ with parameters as known. To update the posterior of $\boldsymbol{\gamma}$, we utilize the **Pymc3** package [4]. With a given $\boldsymbol{\gamma}$, the posterior of $\boldsymbol{\theta}$ enjoys the Beta-Geometric conjugate relationship and hence can be updated explicitly and efficiently. Finally, for each epoch $l$, $A^{l}$ is selected by linear programming as
# \begin{equation}
#     A^{l} = argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}v_{i}}{1+\sum_{j\in a} v_{j}}.
# \end{equation}
# 
# 
# 💥 Application Situation?
# 
# ## Algorithm Details
# At each epoch $l$, given the feedback $\mathcal{H}_{l}$ received from previous rounds, there are two major steps, including posterior sampling and combinatorial optimization. Specifically, the posterior sampling step is decomposed into four steps: 1. approximating a posterior distribution of $\boldsymbol{\gamma}$, $P(\boldsymbol{\gamma}|\mathcal{H}_{l})$, by **Pymc3**; 2. sampling a $\tilde{\boldsymbol{\gamma}}$ from $P(\boldsymbol{\gamma}|\mathcal{H}_{l})$; 3. updating the posterior distribution of $\boldsymbol{\theta}$ conditional on $\tilde{\boldsymbol{\gamma}}$, $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{l})$, which has an explicit form under the assumption of a Beta-Geometric logistic model; 4. sampling $\tilde{\boldsymbol{\theta}}$ from $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{l})$. Then $\tilde{v}_{i}$ is calculated as $\frac{1}{\tilde{\theta}_{i}}-1$. Then, the action $A^{l}$ is selected greedily via linear programming. Note that $\tilde{\boldsymbol{\gamma}}$ can be sampled in a batch mode to further facilitate computationally efficient online deployment.
# 
# 
# ## Key Steps
# For epoch $l = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}_{l})$ by **Pymc3**;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}_{l})$;
# 3. Update $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{l})$
# 4. Sample $\tilde{\boldsymbol{\theta}} \sim P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}_{l})$;
# 5. Compute the utility $\tilde{v}_{i} = \frac{1}{\tilde{\theta}_{i}}-1$
# 6. Take the action $A^{l}$ w.r.t $\{\tilde{v}_{i}\}_{i=1}^{N}$ such that $A^{l} = argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}}$;
# 7. Offer $A^{l}$ until no purchase appears;
# 8. Receive reward $R^{l}$.
# 
# 
# ## Demo Code
# 💥 In the following, we exhibit how to apply the learner on real data.
# 
# *Notations can be found in the introduction of the combinatorial Semi-Bandit problems.

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
# code used to import the learner


# In[5]:


from causaldm.learners.Online.Slate.MNL import MTSS_MNL
from causaldm.learners.Online.Slate.MNL import _env_MNL
import numpy as np


# In[6]:


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


# In[7]:


S


# ## References
# 
# [1] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [2] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2017, June). Thompson sampling for the mnl-bandit. In Conference on Learning Theory (pp. 76-78). PMLR.
# 
# [3] Oh, M. H., & Iyengar, G. (2019). Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32.
# 
# [4] Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 DOI: 10.7717/peerj-cs.55.
