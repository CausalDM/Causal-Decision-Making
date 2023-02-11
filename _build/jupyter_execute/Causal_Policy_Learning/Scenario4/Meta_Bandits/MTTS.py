#!/usr/bin/env python
# coding: utf-8

# # Multi-Task Thompson Sampling (MTTS)
# 
# ## Overview
# - **Advantage**: It is both scalable and robust. Furthermore, it also accounts for the iter-task heterogeneity.
# - **Disadvantage**:
# - **Application Situation**: Useful when there are a large number of tasks to learn, especially when new tasks are introduced on a regular basis. The outcome can be either binary or continuous. Static baseline information.
# 
# ## Main Idea
# The **MTTS**[1] utilize baseline information to share information among different tasks efficiently, by constructing a Bayesian hierarchical model. Specifically, it assumes that
# \begin{equation}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Inter-task)} \quad
# \;    \boldsymbol{\mu}_j | \boldsymbol{s}_j, \boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j | \boldsymbol{s}_j, \boldsymbol{\gamma})=\boldsymbol{s}_j^{T}\boldsymbol{\gamma} + \boldsymbol{\delta}_{j}, \\
# &\text{(Intra-task)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&= \mu_{j,a} + \epsilon_{j,t}, 
#       \end{alignedat}
# \end{equation} where $\boldsymbol{\delta}_{j} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$, and $\epsilon_{j,t} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \sigma^{2})$. For simplicity, we assume a Normal prior, which resulted in a Normal posterior with explicit form. Note that, if we replace the inter-task layer to a deterministic model (i.e., $g(\boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma})=\boldsymbol{s}_j^{T}\boldsymbol{\gamma}$), **MTTS** is reduced to an algorithm similar to **AdaTS** with linear bandits and Gaussian rewards discussed in Section 3.2 [2]. In contrast to **MTSS**, the **AdaTS** fail to address the issue of heterogeneous tasks.
# 
# Similarly, considering the Bernoulli bandit, it assumes that
# \begin{equation}\label{eqn:hierachical_model}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Inter-task)} \quad
# \;    \boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma})=\text{Beta}\big(logistic(\boldsymbol{x}_j^T \boldsymbol{\gamma}), \psi \big), \\
# &\text{(Intra-task)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&\sim  \text{Bernoulli} \big( \mu_{j, a} \big), 
#       \end{alignedat}
# \end{equation}
# where  $logistic(s) \equiv 1 / (1 + exp^{-1}(s))$, $\psi$ is a known parameter, and  $\text{Beta}(\mu, \psi)$ denotes a Beta distribution with mean $\mu$ and precision $\psi$. Still, we assume a Normal prior of $\boldsymbol{\gamma}$. As there is no explicit form of the corresponding posterior, we update the posterior distribution by **Pymc3**.
# 
# Under the TS framework, at each round $t$ with task $j$, the agent will sample a $\tilde{\boldsymbol{\mu}}_{j}$ from its posterior distribution updated according to the hierarchical model, then the action $a$ with the maximum sampled $\tilde{\mu}_{j,a}$ will be pulled. Mathmetically,
# \begin{equation}
#     A_{j,t} = argmax_{a \in \mathcal{A}} \hat{E}(R_{j,t}(a)) = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}.
# \end{equation}
# 
# Essentially, **MTTS** assumes that the mean reward $\boldsymbol{\mu}_{j}$ is sampled from model $g$ parameterized by unknown parameter $\boldsymbol{\gamma}$ and conditional on task feature $\boldsymbol{s}_{j}$. Instead of assuming that $\boldsymbol{\mu}_j$ is fully determined by its features through a deterministic function, **MTTS** adds an item-specific noise term to account for the inter-task heterogeneity. Simultaneously modeling heterogeneity and sharing information across tasks via $g$, **MTTS** is able to provide an informative prior distribution to guide the exploration. Appropriately addressing the heterogeneity between tasks, the MTTS has been shown to have a superior performance in practice[1].
# 
# ## Key Steps
# For $(j,t) = (0,0),(0,1),\cdots$:
# 1. Approximate meta-posterior $P(\boldsymbol{\gamma}|\mathcal{H})$ either by implementing **Pymc3** or by calculating the explicit form of the posterior distribution;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H})$;
# 3. Update $P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$ and sample $\tilde{\boldsymbol{\mu}} \sim P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$;
# 4. Take the action $A_{j,t}$ such that $A_{j,t} = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}$;
# 6. Receive reward $R_{j,t}$.
# 

# ## Demo Code

# In[1]:


import os
os.getcwd()
os.chdir('D:\GitHub\CausalDM')


# ### Import the learner.

# In[2]:


import numpy as np
from causaldm.learners.Online.Meta_Bandits import MTTS_Gaussian


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[3]:


from causaldm.learners.Online.Meta_Bandits import _env_realMultiTask as _env
env = _env.MultiTask_Env(seed = 0, Binary = False)


# ### Specify Hyperparameters
# 
# - `sigma`: the standard deviation of the reward distributions
# - `order`: = 'episodic', if a sequential schedule (i.e., a new task will not be interacted with until the preceding task is completed) is applied; ='concurrent', if a concurrent schedule (i.e., at every step $t$, the agent will interact with all $N$ tasks) is applied
# - `T`: number of total steps per task
# - `theta_prior_mean`: mean of the meta prior distribution
# - `theta_prior_cov`: covariance matrix of the meta prior distribution
# - `delta_cov`: covariance of $\delta_j$
# - `Xs`: Baseline information for all Tasks, (N,K,p) matrix
# - `approximate_solution`: if `True`, we implement the Algorithm 2 in [1]. Specifically, if order = 'episodic', the meta-posterior distribution is updated once a task is finished; if order = 'concurrent, the meta-posterior distribution is updated once the agent finishes interacting with all tasks at each step $t$.
# - `update_freq`: if `approximate_solution = False`, then the meta-posterior is updated every `update_freq` steps
# - `seed`: random seed

# In[4]:


sigma = 1
order="episodic"
T = 100
theta_prior_mean = np.zeros(env.p)
theta_prior_cov = np.identity(env.p)
delta_cov = np.identity(env.K)
Xs = env.Phi
update_freq = 1
approximate_solution = False
seed = 42

MTTS_Gaussian_agent = MTTS_Gaussian.MTTS_agent(sigma = sigma, order=order,T = T, theta_prior_mean = theta_prior_mean, theta_prior_cov = theta_prior_cov,
                                               delta_cov = delta_cov, Xs = Xs, update_freq = update_freq, approximate_solution = approximate_solution, 
                                               seed = seed)


# ### Recommendation and Interaction
# 
# Starting from i = 0, t = 0, for each (i,t), there are four steps:
# 1. Observe the feature information
# <code> X, feature_info = env.get_Phi(i, t) </code>
# 2. Recommend an action 
# <code> A = MTTS_Gaussian_agent.take_action(i, t) </code>
# 3. Get a reward from the environment 
# <code> R = env.get_reward(i, t, A) </code>
# 4. Update the posterior distribution of the mean reward of each arm
# <code> MTTS_Gaussian_agent.receive_reward(i, t, A, R, X) </code>

# In[5]:


i = 0
t = 0
X, feature_info = env.get_Phi(i, t)
A = MTTS_Gaussian_agent.take_action(i, t)
R = env.get_reward(i, t, A)
MTTS_Gaussian_agent.receive_reward(i, t, A, R, X)
i,t,A,R,feature_info


# **Interpretation**: Interacting with the first customer (25-year-old male who is a college/grad student), at step 0, the TS agent recommends a Thriller (arm 3) and receives a rate of 3 from the user.
# 
# **Remark**: use `MTTS_Gaussian_agent.posterior_u[i]` to get the most up-to-date posterior mean of $\mu_{i,a}$ and `MTTS_Gaussian_agent.posterior_cov_diag[0]` to get the corresponding posterior covariance.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit. Note that, when specifying the prior distribution of the expected reward, the mean-precision form of the Beta distribution is used here, i.e., Beta($\mu$, $\phi$), where $\mu$ is the mean reward of each arm and $\phi$ is the precision of the Beta distribution. By default, Algorithm 2 in [1] is applied to save computational time updating meta-posteriors. If `update_freq` is specified, then the meta-posterior will be updated every `update_freq` rounds of interactions.

# In[6]:


from causaldm.learners.Online.Meta_Bandits import MTTS_Binary
env = _env.MultiTask_Env(seed = 0, Binary = True)
theta_prior_mean = np.zeros(env.p)
theta_prior_cov = np.identity(env.p)
phi_beta = 1/4
order="episodic"
T = 100
Xs = env.Phi
update_freq = None
seed = 42

MTTS_Binary_agent = MTTS_Binary.MTTS_agent(T = T, theta_prior_mean = theta_prior_mean, theta_prior_cov = theta_prior_cov,
                                           phi_beta = phi_beta, order = order, Xs = Xs, update_freq = update_freq, 
                                           seed = seed)
i = 0
t = 0
X, feature_info = env.get_Phi(i, t)
A = MTTS_Binary_agent.take_action(i, t)
R = env.get_reward(i, t, A)
MTTS_Binary_agent.receive_reward(i, t, A, R, X)
i,t,A,R,feature_info


# **Interpretation**: Interacting with the first customer (25-year-old male who is a college/grad student), at step 0, the TS agent recommends a Comedy (arm 0) and receives a rate of 1 from the user.
# 
# **Remark**: use `MTTS_Binary_agent.theta` to get the most up-to-date sampled $\tilde{\boldsymbol{\gamma}}$; use `(MTTS_Binary_agent.posterior_alpha[i], MTTS_Binary_agent.posterior_beta[i])` to get the most up-to-date posterior Beta($\alpha$,$\beta$) distribution of $\mu_{i,a}$.

# ## References
# 
# [1] Wan, R., Ge, L., & Song, R. (2021). Metadata-based multi-task bandits with bayesian hierarchical models. Advances in Neural Information Processing Systems, 34, 29655-29668.
# 
# [2] Basu, S., Kveton, B., Zaheer, M., & Szepesvári, C. (2021). No regrets for learning the prior in bandits. Advances in Neural Information Processing Systems, 34, 28029-28041.
# 
