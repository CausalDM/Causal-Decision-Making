#!/usr/bin/env python
# coding: utf-8

# # Meta Thompson Sampling
# 
# ## Overview
# - **Advantage**: When task instances are sampled from the same unknown instance prior (i.e., the tasks are similar), it efficiently learns the prior distribution of the mean potential rewards to achieve a regret bound that is comparable to that of the TS algorithm with known priors. 
# - **Disadvantage**: When there is a large number of different tasks, the algorithm is not scalable and inefficient.
# - **Application Situation**: Useful when there are multiple **similar** multi-armed bandit tasks, each with the same action space. The reward space can be either binary or continuous.
# 
# ## Main Idea
# The **Meta-TS**[1] assumes that the mean potential rewards, $\mu_{j,a} = E(R_{j,t}(a))$, for each task $j$ are i.i.d sampled from some distribution parameterized by $\boldsymbol{\gamma}$. Specifically, it assumes that
# \begin{equation}
#   \begin{alignedat}{2}
# &\text{(meta-Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Prior)} \quad
# \; \quad\quad\quad\quad   \boldsymbol{\mu}_j | \boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j | \boldsymbol{\gamma})\\
# &\text{(Reward)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&\sim f(Y_{j,t}(a)|\mu_{j,a}).
#       \end{alignedat}
# \end{equation}
# To learn the prior distribution of $\boldsymbol{\mu}_{j}$, it introduces a meta-parameter $\boldsymbol{\gamma}$ with a meta-prior distribution $Q(\boldsymbol{\gamma})$. The **Meta-TS** efficiently leverages the knowledge received from different tasks to learn the prior distribution and to guide the exploration of each task by maintaining the meta-posterior distribution of $\boldsymbol{\gamma}$. Theoretically, it is demonstrated to have a regret bound comparable to that of the Thompson sampling method with known prior distribution of $\mu_{j,a}$. Both the 
# 
# Considering a Gaussian bandits, we assume that
# \begin{equation}
#   \begin{alignedat}{2}
# &\text{(meta-Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Prior)} \quad
# \;  \quad\quad\quad\quad \boldsymbol{\mu}_j |\boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j |\boldsymbol{\gamma})=\boldsymbol{\gamma}+ \boldsymbol{\delta}_{j}, \\
# &\text{(Reward)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&= \mu_{j,a} + \epsilon_{j,t}, 
#       \end{alignedat}
# \end{equation} where $\boldsymbol{\delta}_{j} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$, and $\epsilon_{j,t} \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \sigma^{2})$. The $\boldsymbol{\Sigma}$ and $\sigma$ are both supposed to be known. A Gaussian meta-prior is employed by default with explicit forms of posterior distributions for simplicity. However, d ifferent meta-priors are welcome, with only minor modifications needed, such as using the **Pymc3** to accomplish posterior updating instead if there is no explicit form.
# 
# Similarly, considering the Bernoulli bandits, we assume that 
# \begin{equation}
#   \begin{alignedat}{2}
# &\text{(meta-Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Prior)} \quad
# \;  \quad\quad\quad\quad \boldsymbol{\mu}_j |\boldsymbol{\gamma} &&\sim Beta(\boldsymbol{\gamma}), \\
# &\text{(Reward)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&= Bernoulli(\mu_{j,a}). 
#       \end{alignedat}
# \end{equation}
# While various meta-priors can be used, by default, we consider a finite space of $\boldsymbol{\gamma}$,
# \begin{equation}
# \mathcal{P} = \{(\alpha_{i,j})_{i=1}^{K}, (\beta_{i,j})_{i=1}^{K}\}_{j=1}^{L},
# \end{equation}
# which contains **L** potential instance priors and assume a categorical distribution over the $\mathcal{P}$ as the meta-prior. See [1] for more information about the corresponding meta-posterior updating.
# 
# **Remark.** While the original system only supported a sequential schedule of interactions (i.e., a new task will not be interacted with until the preceding task is completed), we adjusted the algorithm to accommodate different recommending schedules. 
# 
# ## Key Steps
# For $(j,t) = (0,0),(0,1),\cdots$:
# 1. Approximate the meta-posterior distribution $P(\boldsymbol{\gamma}|\mathcal{H})$ either by implementing **Pymc3** or by calculating the explicit form of the posterior distribution;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H})$;
# 3. Update $P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$ and sample $\tilde{\boldsymbol{\mu}} \sim P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$;
# 4. Take the action $A_{j,t}$ such that $A_{j,t} = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}$;
# 6. Receive reward $R_{j,t}$.

# ## Demo Code

# ### Import the learner.

# In[1]:


import numpy as np
from causaldm.learners.CPL4.Meta_Bandits import meta_TS_Gaussian


# ### Generate the Environment
# 
# Here, we imitate an environment based on the MovieLens data.

# In[2]:


from causaldm.learners.CPL4.Meta_Bandits import _env_realMultiTask as _env
env = _env.MultiTask_Env(seed = 0, Binary = False)


# ### Specify Hyperparameters
# 
# - `K`: number of arms
# - `N`: number of tasks
# - `sigma`: the standard deviation of the reward distributions
# - `sigma_0`: the standard deviation of the prior distribution of the mean reward of each arm
# - `sigma_q`: the standard deviation of the meta prior distribution
# - `order`: = 'episodic', if a sequential schedule is applied (Note: When order = 'episodic', meta-posterior is updated once a task is finished; otherwise, meta-posterior will be updated at every step)
# - `seed`: random seed

# In[4]:


K = env.K
N = env.N
seed = 42
order = 'episodic'
sigma = 1
sigma_0 = 1 
sigma_q = 1

meta_TS_Gaussian_agent = meta_TS_Gaussian.meta_TS_agent(sigma = sigma, sigma_0 = sigma_0, sigma_q = sigma_q,
                                                        N = N, K = K, order = order, seed = seed)


# ### Recommendation and Interaction
# 
# Starting from t = 0, for each step t, there are three steps:
# 1. Recommend an action 
# <code> A = LinTS_Gaussian_agent.take_action(X) </code>
# 2. Get a reward from the environment 
# <code> R = env.get_reward(t,A) </code>
# 3. Update the posterior distribution of the mean reward of each arm
# <code> meta_TS_Gaussian_agent.receive_reward(i, t, A, R, episode_finished = False) </code>
#     - if a sequential schedule is applied and a task is finished, we would update the meta posterior by setting  `episode_finished = True`;
#     - if the schedule is not sequential, no specification of the parameter ` episode_finished` is needed, and the meta-posterior distribution would be updated at every step automatically

# In[5]:


i = 0
t = 0
A = meta_TS_Gaussian_agent.take_action(i,t)
R = env.get_reward(i,t,A)
meta_TS_Gaussian_agent.receive_reward(i, t, A, R, episode_finished = False)
i,t,A,R


# **Interpretation**: Interacting with Task 0, at step 0, the TS agent recommends a Thriller (arm 3) and receives a rate of 2 from the user.
# 
# **Remark**: use `meta_TS_Gaussian_agent.meta_post` to get the most up-to-date meta-posterior; use `meta_TS_Gaussian_agent.posterior_u[i][A]` to get the most up-to-date posterior mean of $\mu_{i,a}$ and `meta_TS_Gaussian_agent.posterior_cov_diag[i][A]` to get the corresponding posterior covariance.

# ### Demo Code for Bernoulli Bandit
# The steps are similar to those previously performed with a Gaussian Bandit. Note that, when specifying the prior distribution of the expected reward, the mean-precision form of the Beta distribution is used here, i.e., Beta($\mu$, $\phi$), where $\mu$ is the mean reward of each arm and $\phi$ is the precision of the Beta distribution. As we discussed before, we consider using a meta prior with L potential instance priors. Therefore, in implementing the meta-TS algorithm under Bernoulli bandits, we need to specify a list of candidate means of Beta priors. Specifically, three additional parameters are introduced:
# - `phi_beta`: the precision of Beta priors
# - `candi_means`: the candidate means of Beta priors
# - `Q`: categorical distribution of `candi_means`, i.e., each entry is the probability of each candidate mean being the true mean ($\gamma$)
# - `update_freq`: if the recommending schedule is not sequential, we will update the meta-posterior every `update_freq` steps

# In[6]:


from causaldm.learners.Online.Meta_Bandits import meta_TS_Binary
env = _env.MultiTask_Env(seed = 0, Binary = True)
K = env.K
N = env.N
seed = 42
phi_beta = 1/4
candi_means = [np.ones(K)*.1,np.ones(K)*.2,np.ones(K)*.3,np.ones(K)*.4,np.ones(K)*.5]
Q =  np.ones(len(candi_means)) / len(candi_means)
update_freq = 1
order = 'episodic'

meta_TS_Binary_agent = meta_TS_Binary.meta_TS_agent(phi_beta=phi_beta, candi_means = candi_means, Q = Q,
                                                    N = N, K = K, update_freq=update_freq, order = order,
                                                    seed = seed)
i = 0
t = 0
A = meta_TS_Binary_agent.take_action(i,t)
R = env.get_reward(i,t,A)
meta_TS_Binary_agent.receive_reward(i, t, A, R, episode_finished = False)
i,t,A,R


# **Interpretation**: Interacting with Task 0, at step 0, the TS agent recommends a Drama (arm 1) and receives a rate of 0 from the user.
# 
# **Remark**: use `meta_TS_Binary_agent.Q` to get the most up-to-date meta-posterior; use `(meta_TS_Binary_agent.posterior_alpha[i], meta_TS_Binary_agent.posterior_beta[i])` to get the most up-to-date posterior Beta($\alpha$,$\beta$) distribution of $\mu_{i,a}$.

# ## References
# 
# [1] Kveton, B., Konobeev, M., Zaheer, M., Hsu, C. W., Mladenov, M., Boutilier, C., & Szepesvari, C. (2021, July). Meta-thompson sampling. In International Conference on Machine Learning (pp. 5884-5893). PMLR.
# 
# [2] Basu, S., Kveton, B., Zaheer, M., & Szepesvári, C. (2021). No regrets for learning the prior in bandits. Advances in Neural Information Processing Systems, 34, 28029-28041.
# 
