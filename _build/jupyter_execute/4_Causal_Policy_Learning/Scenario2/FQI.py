#!/usr/bin/env python
# coding: utf-8

# # Fitted-Q Iteration
# 

# ## Main Idea
# 
# **Q-function.**
# The Q-function-based approach aims to direct learn the state-action value function (referred to as the Q-function) 
# \begin{eqnarray}
# Q^\pi(a,s)&= \mathbb{E}^{\pi} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|A_{0}=a,S_{0}=s)   
# \end{eqnarray}
# of either the policy $\pi$ that we aim to evaluate or the optimal policy $\pi = \pi^*$. 
# 
# **Bellman optimality equations.**
# The Q-learning-type policy learning is commonly based on the Bellman optimality equation, which characterizes the optimal policy $\pi^*$ and is commonly used in policy optimization. 
# Specifically, $Q^*$ is the unique solution of 
# \begin{equation}
#     Q(a, s) = \mathbb{E} \Big(R_t + \gamma \arg \max_{a'} Q(a, S_{t+1})  | A_t = a, S_t = s \Big).  \;\;\;\;\; \text{(2)} 
# \end{equation}

# **FQI.**
# Similar to [FQE](section:FQE), the fitted-Q iteration (FQI) {cite:p}`ernst2005tree` algorithm is also popular due to its simple form and good numerical performance. 
# It is mainly motivated by the fact that, the optimal value function $Q^*$ is the unique solution to the Bellman optimality equation (2). 
# Besides, the right-hand side of (2) is a contraction mapping. 
# Therefore, we can consider a fixed-point method: 
# with an initial estimate $\widehat{Q}^{0}$, 
# FQI iteratively solves the following optimization problem, 
# 
# \begin{eqnarray}
# 	\widehat{Q}^{{\ell}}=\arg \min_{Q} 
# 	\sum_{\substack{i \le n}}\sum_{t<T}
# 	\Big\{
# 	\gamma \max_{a'} \widehat{Q}^{\ell-1}(a',S_{i, t+1}) 
# 	+R_{i,t}- Q(A_{i, t}, S_{i, t})  
# \Big\}^2,
# \end{eqnarray}
# 
# 
# for $\ell=1,2,\cdots$, until convergence. 
# The final estimate is denoted as $\widehat{Q}_{FQI}$. 
# 

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
