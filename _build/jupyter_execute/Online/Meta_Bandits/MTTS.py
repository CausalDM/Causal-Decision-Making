#!/usr/bin/env python
# coding: utf-8

# # Multi-Task Thompson Sampling (MTTS)
# 
# ## Main Idea
# While most existing algorithms (e.g., \textbf{meta-TS} \citep{kveton2021meta}) fail to leverage feature information, our proposal is more closely related to \textbf{MTTS} \citep{wan2021metadata}, which constructs a Bayesian hierarchical model to share information efficiently. Specifically, 
# \begin{equation}\label{eqn:hierachical_model}
#   \begin{alignedat}{2}
# &\text{(Prior)} \quad
# \quad\quad\quad    \vgamma &&\sim Q(\vgamma), \\
# &\text{(Inter-task)} \quad
# \;    \vmu_j | \vx_j, \vgamma &&\sim g(\vmu_j | \vx_j, \vgamma) = \vx_j^{T}\vgamma + \vdelta_{j}, \\
# &\text{(Intra-task)} \quad
# \;    R_{j,t} = Y_{j,t} &&= \sum\nolimits_{a \in [K]} \I(A_{j,t} = a) \mu_{j,a} + \epsilon_{j,t}, 
#       \end{alignedat}
# \end{equation} where $\vdelta_{j} \stackrel{i.i.d.}{\sim} \mathcal{N}(\vzero, \Cov)$, and $\epsilon_{j,t} \stackrel{i.i.d.}{\sim} \mathcal{N}(\vzero, \sigma^{2})$. Under the TS framework, at each round $t$ with task $j$, the agent will sample a $\tilde{\vmu}_{j}$ from its posterior distribution updated according to the hierarchical model, then the action $a$ with the maximum sampled $\tilde{\mu}_{j,a}$ will be pulled.
# 
# %\begin{equation}
# %    \vthe_i | \vx_i, \vgamma \sim f(\vthe_i | \vx_i, \vgamma) = \vx_i^{T}\vgamma + \vdelta_{i},
# %\end{equation} 
# %Additionally, \textbf{MTTS} assumes a prior distribution of $\vgamma$, 
# Essentially, \textbf{MTTS} assumes that the mean reward $\vmu_{j}$ is sampled from model $g$ parameterized by unknown parameter $\vgamma$ and conditional on task feature $\vx_{j}$. Instead of assuming that $\vmu_j$ is fully determined by its features through a deterministic function, \textbf{MTTS} adds an item-specific noise term to account for the inter-task heterogeneity. Simultaneously modeling heterogeneity and sharing information across tasks via $g$, \textbf{MTTS} is able to provide an informative prior distribution to guide the exploration. Appropriately addressing the heterogeneity between tasks, the MTTS has been shown to have a lower regret bound than both the feature-agnostic and feature-determined MAB algorithms \citep{wan2021metadata}.
# 
# However, it is only applicable to MAB tasks, while we consider CBB, which involves more complex reward structures. Besides, existing meta Bandits algorithms are designed to share information across multiple bandit tasks, and none of them can be applied to our problem of one single large-scale bandit task.
# 
# ## Algorithm Details
# 
# ## Key Steps

# In[ ]:




