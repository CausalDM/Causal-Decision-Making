#!/usr/bin/env python
# coding: utf-8

# # Importance Sampling for Policy Evaluation (Infinite Horizon)

# Another important approach is importance sampling (IS), also known as inverse propensity score or inverse propensity weighting methods. 
# IS has been widely used in statistics, and the idea can be extended to OPE after appropriately handling the temporal dependency. 
# 
# ***Advantages***:
# 
# 1. Conceptually simple and easy to implement
# 2. Low bias. Specifically, with known propensity scores, the vanilla version is unbiased.
# 
# ***Appropriate application situations***:
# 
# Due to the large variance and the curse of horizon, IS generally performs well in problems with
# 
# 1. Short horizon
# 2. Sufficient policy match between the behaviour policy and the target policy. 
# 
# ## Main Idea
# 
# IS estimates the value by reweighting the observed rewards with importance ratios between the target and behavior policy [1]. For simplicity, we assume the behaviour policy $b$ is known. 
# 
# To begin with, for every trajectory index $i$ and any $t \in \{0, 1, \dots, T - 1\}$, we define the $t$-step cumulative **importance ratio** between the target policy $\pi$ and the behaviour policy $b$ as 
# \begin{align*}
#     \rho^i_t = \prod_{t'=0}^{t} \frac{\pi(A_{i,t'}|S_{i,t'})}{b(A_{i,t'}|S_{i,t'})}. 
# \end{align*}
# Since the transition and reward generation probabilities are shared  between both policies, this ratio is equal to the probability ratio of observing the $i$th trajectory until time point $t$. 
# 
# The standard **(trajectory-wise) IS** estimator [2] regards each trajectory (and the corresponding observed cumulative reward, $\sum_{t=0}^{T-1} \gamma^t R_{i,t}$) as one realization, and it estimates $\eta^{\pi}$ by 
# \begin{align}\label{eqn:IS}
#     \hat{\eta}^{\pi}_{IS} = \frac{1}{n} \sum_{i=1}^n \rho^i_T (\sum_{t=0}^{T-1} \gamma^t R_{i,t}). 
# \end{align}
# 
# In contrast, the **step-wise IS** [2]  focuses on reweighting each immediate reward $R_{i,t}$ and typically yields a lower variance than the trajectory-wise IS. It is defined as 
# \begin{align}\label{eqn:stepIS}
#     \hat{\eta}^{\pi}_{StepIS} = \frac{1}{n} \sum_{i=1}^n \Big[ \sum_{t=0}^{T-1} \rho^i_t  \gamma^t R_{i,t} \Big]. 
# \end{align}
# 
# In addition to these two IS-type estimators, their **self-normalized variants** are also commonly considered [3]. 
# Specifically, we can define the normalization factor $\bar{\rho}_t = N^{-1} \sum_{i=1}^N \rho^i_t$, and replace the $\rho^i_t$ term by $\rho^i_t / \bar{\rho}_t$. 
# The resulting estimators are biased but consistent, and they generally yield lower variance than their counterparts. 
# This comparison reflects the bias-variance trade-off. 

# ## Breaking the curse of horizon with stationary distribution
# 
# Traditional IS methods (and related DR methods) have exponential variance with the number of steps and hence will soon become unstable when the trajectory is long.  To avoid this issue,  [4] made an important step forward by proposing to utilize the stationary distributions of the Markov process to marginalize the importance ratio. We need to assume the stationarity assumption (SA), that the state process $\{S_{i,t}\}_{t \ge 0}$ is strictly stationary. 
# 
# 
# Let $p_b(s)$ and  $p_b(s, a)$ denote the stationary density function of the state and the state-action pair under the policy $b$, respectively.
# The key observation is that, under the stationary assumption and when the data is weakly dependent, we can consider the importance ratios computed on each state-action pair rather than on each  trajectory, and hence break the curse of horizon. 
# We introduce the average visitation distribution under a policy $\pi$ as $d^{\pi}(s)= (1 - \gamma)^{-1} \sum_{t=0}^{+\infty} \gamma^{t} p_t^{\pi}(s)$, where $p_t^{\pi}(s)$ denotes the probability of $\{S_t = s\}$ following policy $\pi$ with  $S_{0}\sim \mathbb{G}$. 
# Define $\widetilde{\omega}^{\pi}(s) = d^{\pi}(s) / d^{b}(s)$. 
# Therefore, $\widetilde{\omega}^{\pi}(s)$ can be understood as a marginalized version of the importance ratio. With a similar change-of-measure trick as in IS, we can obtain the relationship that 
# \begin{equation}\label{eqn:breaking}
#     \eta^{\pi} =  \mathbb{E}_{(s,a) \sim p_b(s, a), r \sim \mathcal{R}(\cdot; s, a)} \widetilde{\omega}^{\pi}(s) \frac{\pi(a|s)}{b(a|s)} r. 
# \end{equation}
# According to this relationship, we can construct an estimator by replacing the nuisance functions  with their estimates and then approximating the expectation by its empirical mean over $\{(S_{i,t},A_{i,t},R_{i,t},S_{i,t+1})\}$. 
# The nuisance function $\widetilde{\omega}^{\pi}(s)$ is typically learned by solving an optimization problem, which we will omit to save space. 
# The optimization is similar to a relevant task that we will discuss in the next section, which is more related with our proposal. 

# ## Demo [TODO]

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## References
# [1] Precup D. Eligibility traces for off-policy policy evaluation[J]. Computer Science Department Faculty Publication Series, 2000: 80.
# 
# [2] Thomas P S. Safe reinforcement learning[J]. 2015.
# 
# [3] Jiang N, Li L. Doubly robust off-policy value evaluation for reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2016: 652-661.
# 
# [4] Liu Q, Li L, Tang Z, et al. Breaking the curse of horizon: Infinite-horizon off-policy estimation[J]. Advances in Neural Information Processing Systems, 2018, 31.

# ## Note
# 1. When the behaviour policy is unknown, we can estimate it from data by regarding the task as a classification problem and using methods such as logistic regression. 
# 2. We note that, in principle, IS-based methods (and doubly robust methods to be reviewed in the next section) only apply to the finite-horizon setting, where the  trajectory is truncated at a finite time step $T$. 
# The estimand is 
# $\mathbb{E}^{\pi}_{s \sim \mathbb{G}} (\sum_{t=0}^{T-1} \gamma^t R_{t}|S_{0}=s)$ instead of 
# $\mathbb{E}^{\pi}_{s \sim \mathbb{G}} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|S_{0}=s)$. 
# However, when $T$ is relatively large and $\gamma$ is not quite close to $1$, the difference between $\sum_{t=0}^{T-1} \gamma^t$ and $\sum_{t=0}^{\infty} \gamma^t$ is negligible and is usually ignored, and they are still commonly used as baselines. 
# 3. We note that (SA) is not a strong assumption. Recall that $\{S_{i,t}\}_{t \ge 0}$ is generated by following the stationary policy $b$. (SA) is automatically  satisfied when the initial distribution equals the stationary distribution. Besides, When the MDP is a Harris ergodic chain , the process will eventually mix well and we can replace the stationary distribution with its limiting assumption and the following discussions will continue to hold. 

# In[ ]:




