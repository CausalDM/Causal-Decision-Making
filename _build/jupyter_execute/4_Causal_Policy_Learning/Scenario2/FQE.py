#!/usr/bin/env python
# coding: utf-8

# (section:FQE)=
# # Fitted-Q Evaluation
# 
# The most straightforward approach for OPE is the direct method (DM). 
# As suggested by the name, 
# methods belonging to this category will first directly impose a model for either the environment or the Q-function, and then learn the model by regarding the task as a regression (or classification) problem, and finally calculate the value of the target policy via a plug-in estimator according to the definition of $\eta^\pi$
# The Q-function based approach and the environment-based approach are also called as model-free and  model-based, respectively. 
# 
# Among the many model-free DM estimators, we will focus on the most classic one, the fitted-Q evaluation (FQE) {cite:p}`le2019batch`. 
# It is observed to perform consistently well in a large-scale empirical study {cite:p}`voloshin2019empirical`. 
# 
# ***Advantages***:
# 
# 1. Conceptually simple and easy to implement
# 2. Good numerical results when the the model class is chosen appropriately 
# 
# ***Appropriate application situations***:
# 
# Due to the potential bias, FQE generally performs well in problems where
# 
# 1. The model class can be chosen appropriately 
# 

# ## Main Idea
# 
# **Q-function.**
# The Q-function-based approach aims to direct learn the state-action value function (referred to as the Q-function) 
# \begin{eqnarray}
# Q^\pi(a,s)&= \mathbb{E}^{\pi} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|A_{0}=a,S_{0}=s)   
# \end{eqnarray}
# of the policy $\pi$ that we aim to evaluate. 
# 
# The final estimator can then be constructed by plugging $\hat{Q}^{\pi}$ in the definition $\eta^{\pi} = \mathbb{E}_{s \sim \mathbb{G}, a \sim \pi(\cdot|s)} Q^{\pi}(a, s)$. 
# 
# 
# **Bellman equations.**
# The Q-learning-type evaluation is commonly based on the Bellman equation for the Q-function of a given policy $\pi$ 
# \begin{equation}\label{eqn:bellman_Q}
#     Q^\pi(a, s) = \mathbb{E}^\pi \Big(R_t + \gamma Q^\pi(A_{t + 1}, S_{t+1})  | A_t = a, S_t = s \Big).  \;\;\;\;\; \text{(1)} 
# \end{equation}
# 
# 
# **FQE.**
# FQE is mainly motivated by the fact that, the true value function $Q^\pi$ is the unique solution to the Bellman equation (1). 
# Besides, the right-hand side of (1) is a contraction mapping. 
# Therefore, we can consider a fixed-point method: 
# with an initial estimate $\widehat{Q}^{0}$, 
# FQE iteratively solves the following optimization problem, 
# 
# 
# \begin{eqnarray}
# 	\widehat{Q}^{{\ell}}=\arg \min_{Q} 
# 	\sum_{\substack{i \le n}}\sum_{t<T}
# 	\Big\{
# 	\gamma \mathbb{E}_{a' \sim \pi(\cdot| S_{i, t+1})} \widehat{Q}^{\ell-1}(a',S_{i, t+1}) 
# 	+R_{i,t}- Q(A_{i, t}, S_{i, t})  
# \Big\}^2,
# \end{eqnarray}
# 
# 
# for $\ell=1,2,\cdots$, until convergence. 
# The final estimator is denoted as $\widehat{Q}_{FQE}$. 

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
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
