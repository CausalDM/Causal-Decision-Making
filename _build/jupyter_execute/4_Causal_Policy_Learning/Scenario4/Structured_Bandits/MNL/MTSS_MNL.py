#!/usr/bin/env python
# coding: utf-8

# # MTSS_MNL
# 
# ## Overview
# - **Advantage**: It is both scalable and robust. Furthermore, it also accounts for the iter-item heterogeneity.
# - **Disadvantage**:
# - **Application Situation**: Useful when a list of items is presented, each with a matching price or income, and only one is chosen for each interaction. Binary responses from users include click/don't-click and buy/don't-buy.
# 
# ## Main Idea
# MTSS_MNL is an example of the general Thompson Sampling(TS)-based framework, MTSS [1], to deal with dynamic assortment optimization problems.
# 
# **Review of MTSS:** MTSS[1] is a meta-learning framework designed for large-scale structured bandit problems [2]. Mainly, it is a TS-based algorithm that learns the information-sharing structure while minimizing the cumulative regrets. Adapting the TS framework to a problem-specific Bayesian hierarchical model, MTSS simultaneously enables information sharing among items via their features and models the inter-item heterogeneity. Specifically, it assumes that the item-specific parameter $\theta_i$ is sampled from a distribution $g(\theta_i|\boldsymbol{s}_i, \boldsymbol{\gamma})$ instead of being entirely determined by $\boldsymbol{s}_i$ via a deterministic function. Here, $g$ is a model parameterized by an **unknown** vector $\boldsymbol{\gamma}$. The following is the general feature-based hierarchical model MTSS considered. 
# \begin{equation}\label{eqn:general_hierachical}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad\quad\quad\quad\quad\quad\quad
# \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}),\\
# &\text{(Generalization function)} \;
# \;    \theta_i| \boldsymbol{s}_i, \boldsymbol{\gamma}  &&\sim g(\theta_i|\boldsymbol{s}_i, \boldsymbol{\gamma}), \forall i \in [N],\\ 
# &\text{(Observations)} \quad\quad\quad\quad\quad\quad\;
# \;    \boldsymbol{Y}_t(a) &&\sim f(\boldsymbol{Y}_t(a)|\boldsymbol{\theta}),\\
# &\text{(Reward)} \quad\quad\quad\quad\quad\quad\quad\quad\;
# \;   R_t(a) &&= f_r(\boldsymbol{Y}_t(a) ; \boldsymbol{\eta}), 
#       \end{alignedat}
# \end{equation}
# where $Q(\boldsymbol{\gamma})$ is the prior distribution for $\boldsymbol{\gamma}$. 
# Overall, MTTS is a **general** framework that subsumes a wide class of practical problems, **scalable** to large systems, and **robust** to the specification of the generalization model.
# 
# **Review of MTSS_MNL:** In this tutorial, as a concrete example, we focus on the epoch-type offering schedule and consider modeling the relationship between $\theta_i$ and $\boldsymbol{s}_i$ with the following Beta-Geometric logistic model:
# \begin{equation}\label{eqn1}
#     \begin{split}
#      \theta_i &\sim Beta \big(\frac{logistic(\boldsymbol{s}_i^T \boldsymbol{\gamma})+ 1}{2}, \phi \big) , \forall i \in [N],\\
#     Y_{i}^l(a) &\sim Geometric(\theta_i), \forall i \in a,\\
#     R^l(a) &= \sum_{i\in a}Y_{i}^l(a)\eta_{i},
#     \end{split}
# \end{equation}where we adopt the mean-precision parameterization of the Beta distribution and $\phi$ and $\boldsymbol{\eta}$ are known. We choose this specific form as it is widely observed that $v_i < 1$ [2,3], i.e., no item is more popular than the no-purchase option. This is equal to $\theta_i \in (1/2, 1)$. Other generalization models are also possible with minor modifications to the posterior sampling code. 
# The prior $Q(\boldsymbol{\gamma})$ can be chosen as many appropriate distributions. For instance, we choose the prior $\boldsymbol{\gamma} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\gamma}}, {\boldsymbol{\Sigma}}_{\boldsymbol{\gamma}})$ with parameters as known. To update the posterior of $\boldsymbol{\gamma}$, we utilize the **Pymc3** package [4]. With a given $\boldsymbol{\gamma}$, the posterior of $\boldsymbol{\theta}$ enjoys the Beta-Geometric conjugate relationship and hence can be updated explicitly and efficiently. Finally, for each epoch $l$, $A^{l}$ is selected by linear programming as
# \begin{equation}
#     A^{l} = argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}v_{i}}{1+\sum_{j\in a} v_{j}}.
# \end{equation}
# It should be noted that the posterior updating step differs for different pairs of the $Q(\boldsymbol{\gamma})$ and the reward distribution, and the corresponding code can be easily modified to different prior/reward distribution specifications if necessary.
# 
# ## Algorithm Details
# At each epoch $l$, given the feedback $\mathcal{H}^{l}$ received from previous rounds, there are two major steps, including posterior sampling and combinatorial optimization. Specifically, the posterior sampling step is decomposed into four steps: 1. approximating a posterior distribution of $\boldsymbol{\gamma}$, $P(\boldsymbol{\gamma}|\mathcal{H}^{l})$, by **Pymc3**; 2. sampling a $\tilde{\boldsymbol{\gamma}}$ from $P(\boldsymbol{\gamma}|\mathcal{H}^{l})$; 3. updating the posterior distribution of $\boldsymbol{\theta}$ conditional on $\tilde{\boldsymbol{\gamma}}$, $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}^{l})$, which has an explicit form under the assumption of a Beta-Geometric logistic model; 4. sampling $\tilde{\boldsymbol{\theta}}$ from $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}^{l})$. Then $\tilde{v}_{i}$ is calculated as $\frac{1}{\tilde{\theta}_{i}}-1$. Then, the action $A^{l}$ is selected greedily via linear programming. Note that $\tilde{\boldsymbol{\gamma}}$ can be sampled in a batch mode to further facilitate computationally efficient online deployment.
# 
# 
# ## Key Steps
# For epoch $l = 1,2,\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H}^{l})$ by **Pymc3**;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H}^{l})$;
# 3. Update $P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}^{l})$
# 4. Sample $\tilde{\boldsymbol{\theta}} \sim P(\boldsymbol{\theta}|\tilde{\boldsymbol{\gamma}},\mathcal{H}^{l})$;
# 5. Compute the utility $\tilde{v}_{i} = \frac{1}{\tilde{\theta}_{i}}-1$;
# 6. Take the action $A^{l}$ w.r.t $\{\tilde{v}_{i}\}_{i=1}^{N}$ such that $A^{l} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\tilde{\boldsymbol{v}})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}\tilde{v}_{i}}{1+\sum_{j\in a} \tilde{v}_{j}}$;
# 7. Offer $A^{l}$ until no purchase appears;
# 8. Receive reward $R^{l}$.
# 
# *Notations can be found in either the inroduction of the chapter "Structured Bandits" or the introduction of the Multinomial Logit Bandit problems.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.Structured_Bandits.MNL import MTSS_MNL


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens dataset.

# In[2]:


from causaldm.learners.CPL4.Structured_Bandits.MNL import _env_realMNL as _env
env = _env.MNL_env(seed = 0)


# ### Specify Hyperparameters
# - K: number of itmes to be recommended at each round
# - L: total number of candidate items
# - Xs: feature informations $\boldsymbol{S}$ (Note: if an intercept is considered, the $\boldsymbol{S}$ should include a column of ones)
# - phi_beta: precision of the Beta distribution (i.e., $\phi$)
# - gamma_prior_mean: the mean of the prior distribution of $\boldsymbol{\gamma}$
# - gamma_prior_cov: the coveraince matrix of the prior distribution of $\boldsymbol{\gamma}$ 
# - r: revenue of items
# - same_reward: indicate whether the revenue of each item is the same or not
# - n_init: determine the number of samples that pymc3 will draw when updating the posterior of $\boldsymbol{\gamma}$ 
# - update_freq: frequency to update the posterior distribution of $\boldsymbol{\gamma}$ (i.e., update every update_freq steps)
# - clip: indicate whether we clip the $\boldsymbol{\theta}$ to be between $.5$ and $.999$
# - seed: random seed

# In[4]:


L = env.L
K = 5
Xs = env.Phi
phi_beta = .002
gamma_prior_mean = np.ones(env.p)
gamma_prior_cov = np.identity(env.p)
r = env.r
same_reward = False
n_init = 1000
update_freq = 100
clip = True
seed = 0

MTSS_agent = MTSS_MNL.MTSS_MNL(L = L, K = K, Xs = Xs, phi_beta = phi_beta, gamma_prior_mean = gamma_prior_mean, 
                                gamma_prior_cov = gamma_prior_cov, r = r, same_reward = same_reward, 
                                n_init = n_init, update_freq=update_freq, clip = clip, seed = seed)


# ### Recommendation and Interaction
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action (a set of ordered restaturants)
# <code> A = MTSS_agent.take_action() </code>
# 3. Get the item clicked and the corresponding revenue from the environment
# <code> c, _, R = env.get_reward(A) </code>
# 4. Update the posterior distribution
# <code> MTSS_agent.receive_reward(A,c,R) </code>

# In[5]:


t = 0
A = MTSS_agent.take_action()
c, _, R= env.get_reward(A)
MTSS_agent.receive_reward(A, c, R)
A, c, R


# **Interpretation**: For step 0, the agent recommends five movies to the customer, the ids of which are 20, 275, 421, 448, and 836. The customer finally clicks the movie 275 and the agent receives a revenue of .95.

# ## References
# 
# [1] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [2] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2017, June). Thompson sampling for the mnl-bandit. In Conference on Learning Theory (pp. 76-78). PMLR.
# 
# [3] Oh, M. H., & Iyengar, G. (2019). Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32.
# 
# [4] Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 DOI: 10.7717/peerj-cs.55.
