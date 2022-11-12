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
# To learn the prior distribution of $\boldsymbol{\mu}_{j}$, it introduces a meta-parameter $\gamma$ with a meta-prior distribution $Q(\gamma)$. The **Meta-TS** efficiently leverages the knowledge received from different tasks to learn the prior distribution and to guide the exploration of each task by maintaining the meta-posterior distribution of $\gamma$. Theoretically, it is demonstrated to have a regret bound comparable to that of the Thompson sampling method with known prior distribution of $\mu_{j,a}$. Both the 
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
# \end{equation} where $\boldsymbol{\delta}_{j} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$, and $\epsilon_{j,t} \stackrel{i.i.d.}{\sim} \mathcal{N}(\boldsymbol{0}, \sigma^{2})$. The $\boldsymbol{\Sigma}$ and $\sigma$ are both supposed to be known. A Gaussian meta-prior is employed by default with explicit forms of posterior distributions for simplicity. However, d ifferent meta-priors are welcome, with only minor modifications needed, such as using the **Pymc3** to accomplish posterior updating instead if there is no explicit form.
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
# $$\mathcal{P} = \{(\alpha_{i,j})_{i=1}^{K}, (\beta_{i,j})_{i=1}^{K}\}_{j=1}^{L},$$ 
# which contains **L** potential instance priors and assume a categorical distribution over the $\mathcal{P}$ as the meta-prior. See [1] for more information about the corresponding meta-posterior updating.
# 
# **Remark.** While the original system only supported a sequential schedule of interactions (i.e., a new task will not be interacted with until the preceding task is completed), we adjusted the algorithm to accommodate different recommending schedules.
# 
# ## Key Steps
# For $(j,t) = (1,1),(1,2),\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H})$ either by implementing **Pymc3** or by calculating the explicit form of the posterior distribution;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H})$;
# 3. Update $P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$ and sample $\tilde{\boldsymbol{\mu}} \sim P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$;
# 4. Take the action $A_{j,t}$ such that $A_{j,t} = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}$;
# 6. Receive reward $R_{j,t}$.

# ## Demo Code
# 
# TODO： Bernoulli: specify the prior space

# ## References
# 
# [1] Kveton, B., Konobeev, M., Zaheer, M., Hsu, C. W., Mladenov, M., Boutilier, C., & Szepesvari, C. (2021, July). Meta-thompson sampling. In International Conference on Machine Learning (pp. 5884-5893). PMLR.
# 
# [2] Basu, S., Kveton, B., Zaheer, M., & Szepesvári, C. (2021). No regrets for learning the prior in bandits. Advances in Neural Information Processing Systems, 34, 28029-28041.
# 
