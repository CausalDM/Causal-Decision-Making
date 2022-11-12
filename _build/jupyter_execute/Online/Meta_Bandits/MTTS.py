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
# \;    \boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma})=\boldsymbol{x}_j^{T}\boldsymbol{\gamma} + \boldsymbol{\delta}_{j}, \\
# &\text{(Intra-task)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&= \mu_{j,a} + \epsilon_{j,t}, 
#       \end{alignedat}
# \end{equation} where $\boldsymbol{\delta}_{j} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$, and $\epsilon_{j,t} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \sigma^{2})$. For simplicity, we assume a Normal prior, which resulted in a Normal posterior with explicit form. Note that, if we replace the inter-task layer to a deterministic model (i.e., $g(\boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma})=\boldsymbol{x}_j^{T}\boldsymbol{\gamma}$), **MTTS** is reduced to an algorithm similar to **AdaTS** with linear bandits and Gaussian rewards discussed in Section 3.2 [2]. In contrast to **MTSS**, the **AdaTS** fail to address the issue of heterogeneous tasks.
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
# where  $logistic(x) \equiv 1 / (1 + exp^{-1}(x))$, $\psi$ is a known parameter, and  $\text{Beta}(\mu, \psi)$ denotes a Beta distribution with mean $\mu$ and precision $\psi$. Still, we assume a Normal prior of $\boldsymbol{\gamma}$. As there is no explicit form of the corresponding posterior, we update the posterior distribution by **Pymc3**.
# 
# Under the TS framework, at each round $t$ with task $j$, the agent will sample a $\tilde{\boldsymbol{\mu}}_{j}$ from its posterior distribution updated according to the hierarchical model, then the action $a$ with the maximum sampled $\tilde{\mu}_{j,a}$ will be pulled. Mathmetically,
# \begin{equation}
#     A_{j,t} = argmax_{a \in \mathcal{A}} \hat{E}(R_{j,t}(a)) = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}.
# \end{equation}
# 
# Essentially, **MTTS** assumes that the mean reward $\boldsymbol{\mu}_{j}$ is sampled from model $g$ parameterized by unknown parameter $\boldsymbol{\gamma}$ and conditional on task feature $\boldsymbol{x}_{j}$. Instead of assuming that $\boldsymbol{\mu}_j$ is fully determined by its features through a deterministic function, **MTTS** adds an item-specific noise term to account for the inter-task heterogeneity. Simultaneously modeling heterogeneity and sharing information across tasks via $g$, **MTTS** is able to provide an informative prior distribution to guide the exploration. Appropriately addressing the heterogeneity between tasks, the MTTS has been shown to have a superior performance in practice[1].
# 
# ## Key Steps
# For $(j,t) = (1,1),(1,2),\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H})$ either by implementing **Pymc3** or by calculating the explicit form of the posterior distribution;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H})$;
# 3. Update $P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$ and sample $\tilde{\boldsymbol{\mu}} \sim P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$;
# 4. Take the action $A_{j,t}$ such that $A_{j,t} = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}$;
# 6. Receive reward $R_{j,t}$.
# 

# ## References
# 
# [1] Wan, R., Ge, L., & Song, R. (2021). Metadata-based multi-task bandits with bayesian hierarchical models. Advances in Neural Information Processing Systems, 34, 29655-29668.
# 
# [2] Basu, S., Kveton, B., Zaheer, M., & Szepesvári, C. (2021). No regrets for learning the prior in bandits. Advances in Neural Information Processing Systems, 34, 28029-28041.
# 

# In[ ]:




