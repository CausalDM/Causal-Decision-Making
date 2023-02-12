#!/usr/bin/env python
# coding: utf-8

# # Doubly Robust Estimator for Policy Evaluation (Infinite Horizon)

# The third category, the doubly robust (DR) approach, combines DM and IS to achieve low variance and bias. The DR technique has also been widely studied in statistics. 
# 
# ***Advantages***:
# 
# 1. Doubly robustness: consistent when either component is
# 2. Fast convergence rate when both components have decent convergence rates. 
# 
# ***Appropriate application situations***:
# 
# In the MDP setup, due to the large variance and curse of horizon introduced by the IS component, it is observed that DR [6] generally performs better than DM when 
# 
# 1. Horizon is short 
# 2. Policy match is sufficient
# 3. The Q-function model might exist significant bias. 
# 
# ## Main Idea
# 
# In OPE, a DR estimator first requires a Q-function estimator, denoted as $\widehat{Q}$, which can be learned by various methods in the literature, such as FQE [**CROSS-REFER**]. 
# Denote the corresponding plug-in V-function estimator as $\widehat{V}$. 
# These estimators will then be integrated with importance ratios in a form typically motivated by the Bellman equation 
# \begin{equation}\label{eqn:bellman_Q}
#     Q^\pi(a, s) = \mathbb{E}^\pi \Big(R_t + \gamma Q^\pi(A_{t + 1}, S_{t+1})  | A_t = a, S_t = s \Big).  \;\;\;\;\; \text{(1)} 
# \end{equation}
# 
# For example, based on the step-wise IS [**CROSS-REFER**], [1] proposed to construct the estimator as 
# \begin{align*}%\label{eqn:stepIS}
#     \hat{\eta}^{\pi}_{StepDR} = \frac{1}{n} \sum_{i=1}^n  \widehat{V}(S_{i,0}) + 
#     \frac{1}{n} \sum_{i=1}^n \sum_{t=0}^{T - 1} \rho^i_t  \gamma^t \Big[
#     R_{i,t} - \widehat{Q}(A_{i,t}, S_{i,t}) + \gamma \widehat{V}(S_{i,t + 1})
#     \Big]. 
# \end{align*}
# The self-normalized version can be similarly constructed [1]. 
# 
# Besides directly applying the DR technique to the value estimator, we can utilize the recursive form 
# to debias the Q- or V-function recursively. 
# For example, [5] considered the following estimator. 
# Let $\widehat{V}_{DR}^T = 0$. 
# For $t = T - 1, \dots, 0$, we recursively define 
# \begin{equation*}
#     \widehat{V}_{DR}^t = \frac{1}{n} \sum_{i=1}^n \Big\{ \widehat{V}(S_{i,t}) + \rho^i_t \big[R_{i,t} + \gamma \widehat{V}_{DR}^{t+1}(S_{i,t + 1}) - \widehat{Q}(A_{i,t}, S_{i,t})
#     \big] \Big\}. 
# \end{equation*}
# The final value estimator is then defined as $\widehat{V}_{DR}^0$. 
# 
# The name, doubly robust, reflects the fact that the DR estimators are typically consistent as long as one of the two components is consistent, and hence the estimator is doubly robust to model mis-specifications. 
# Besides, a DR estimator typically has lower (or comparable) bias and variance than its components, in the asymptotic sense. However, similar with the standard IS methods, standard DR estimators also rely on per-step importance ratios  and hence will suffer from huge variance when the horizon is long. 
# 
# ## Double Reinforcement Learning with Stationary Distribution
# To avoid the curse of horizon, a few extensions of the stationary distribution-based approach have been proposed in the literature. 
# For example, [2] designed a DR version, and [3] proposed to learn a single nuisance function $\widetilde{\xi}^{\pi}(s,a) \equiv \widetilde{\omega}^{\pi}(s) [\pi(a|s) / b(a|s)]$ instead of learning $\widetilde{\omega}^{\pi}(s)$ and $b$ separately. 
# 
# In particular, following this line of research, [4] recently proposed a state-of-the-art method named double reinforcement learning (DRL) that achieves the semiparametric efficiency bound for OPE. DRL is a doubly robust-type method. 
# 
# To begin with, we first define the marginalized density ratio under the target policy $\pi$ as  
# \begin{eqnarray}\label{eqn:omega}
# 	\omega^{\pi}(a,s)=\frac{(1-\gamma)\sum_{t=0}^{+\infty} \gamma^{t} p_t^{\pi}(a,s)}{p_b(a, s)}, 
# \end{eqnarray}
# where $p_t^{\pi}(a, s)$ denotes the probability of $\{S_t = s, A_t = a\}$ following policy $\pi$ with  $S_{0}\sim \mathbb{G}$. 
# Recall that $p_b(s, a)$ is the stationary density function of the state-action pair under the policy $b$. 
# 
# 
# 
# Let $\widehat{Q}$ and $\widehat{\omega}$ be some estimates of  $Q^{\pi}$ and $\omega^{\pi}$,  respectively. 
# DRL first constructs the following estimator for every $(i,t)$ in a doubly robust manner: 
# \begin{eqnarray}\label{term}
# \begin{split}
# 	\psi_{i,t}
# 	\equiv
# 	\frac{1}{1-\gamma}\widehat{\omega}(A_{i,t},S_{i,t})\{R_{i,t} 
# 	-\widehat{Q}(A_{i,t},S_{i,t})+
# 	\gamma 
# 	\mathbb{E}_{a \sim \pi(\cdot| S_{i,t+1})}\widehat{Q}(a, S_{i,t+1})\}
# 	+ \mathbb{E}_{s \sim \mathbb{G}, a \sim \pi(\cdot| s)}\widehat{Q}(a, s). 
# \end{split}	
# \end{eqnarray}
# The resulting value estimator is then given by
# \begin{eqnarray*}
# 	\widehat{\eta}_{\tiny{\textrm{DRL}}}=\frac{1}{nT}\sum_{i=1}^n\sum_{t=0}^{T-1} \psi_{i,t}.
# \end{eqnarray*}
# One can show that 
# $\widehat{\eta}_{\tiny{\textrm{DRL}}}$ is consistent when either $\widehat{Q}$ or $\widehat{\omega}$ is consistent, and hence is doubly robust. 
# In addition, under mild conditions, we can prove that $\sqrt{nT} (\widehat{\eta}_{\tiny{\textrm{DRL}}} - \eta^{\pi})$ converges weakly to a normal distribution with mean zero and variance $\sigma^2$ as 
# \begin{eqnarray}\label{lower_bound}
#     \sigma^2 = 
#     \frac{1}{(1-\gamma)^2}\mathbb{E} \left[ 
#     \omega^{\pi}(A, S) \{R + \gamma V^{\pi}(S') -  Q^{\pi}(A,S)\}
#     \right]^2,
# \end{eqnarray}
# where the expectation is over tuples following 
# the stationary distribution of the process $\{(S_t,A_t,R_t,S_{t+1})\}_{t\ge 0}$, generated by $b$. Moreover, this asymptotic variance is proven to be the semiparametric efficiency bound for  infinite-horizon OPE [4]. 
# Roughly speaking, this implies the algorithm is statistically most efficient. 

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
# [1] Thomas P, Brunskill E. Data-efficient off-policy policy evaluation for reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2016: 2139-2148.
# 
# [2] Tang Z, Feng Y, Li L, et al. Doubly robust bias reduction in infinite horizon off-policy estimation[J]. arXiv preprint arXiv:1910.07186, 2019.
# 
# [3] Uehara M, Huang J, Jiang N. Minimax weight and q-function learning for off-policy evaluation[C]//International Conference on Machine Learning. PMLR, 2020: 9659-9668.
# 
# [4] Kallus N, Uehara M. Efficiently breaking the curse of horizon in off-policy evaluation with double reinforcement learning[J]. Operations Research, 2022.
# 
# [5] Jiang N, Li L. Doubly robust off-policy value evaluation for reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2016: 652-661.
# 
# [6] Voloshin C, Le H M, Jiang N, et al. Empirical study of off-policy policy evaluation for reinforcement learning[J]. arXiv preprint arXiv:1911.06854, 2019.

# ## Note
# 1. 
# One critical question is how to estimate the nuisance function $\omega^{\pi}$. 
# The following observation forms the basis: $\omega^{\pi}$ is the only function that satisfies the equation $\mathbb{E} L(\omega^{\pi},f)=0$ for any function $f$, where $L(\omega^{\pi},f)$ equals 
# \begin{eqnarray}\label{eqn_omega}
# \begin{split}
# 	\Big[\mathbb{E}_{a \sim \pi(\cdot|S_{t+1})} \{\omega^{\pi}(A_{t},S_{t})
# 	(\gamma f(a, S_{t+1})- f(A_{t},S_{t}) ) \}
# 	+ (1-\gamma) \mathbb{E}_{s \sim \mathbb{G}, a \sim \pi(\cdot|s)} f(a, s). 
# \end{split}
# \end{eqnarray} 
# As such, $\omega^{\pi}$ can be learned by solving the following mini-max problem, 
# \begin{eqnarray}\label{eqn:solveL}
# \arg \min_{\omega\in \Omega} \sup_{f\in \mathcal{F}} \{\mathbb{E} L(\omega, f)\}^2, 
# \end{eqnarray}
# for some function classes $\Omega$ and $\mathcal{F}$. 
# To simplify the calculation, we can choose $\mathcal{F}$ to be a reproducing kernel Hilbert space (RKHS). 
# This yields a closed-form expression for $\sup_{f\in \mathcal{F}} \{\mathbb{E} L(\omega,f)\}^2$, for any $\omega$. Consequently, $\omega^{\pi}$ can be learned by solving the outer minimization via optimization methods such as stochastic gradient descent, 
# with the expectation approximated by the sample mean. 
# $\widetilde{\omega}^{\pi}(s)$ can be learned in a similar manner. 

# In[ ]:




