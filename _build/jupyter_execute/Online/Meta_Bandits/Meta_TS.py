#!/usr/bin/env python
# coding: utf-8

# # Meta Thompson Sampling
# 
# ## Overview
# - **Advantage**: When task instances are sampled from the same unknown instance prior (i.e., the tasks are similar), it shares information across tasks to efficiently learn the common prior distribution of the mean rewards.
# - **Disadvantage**: When there is a large number of different tasks, the algorithm is not scalable and inefficient.
# - **Application Situation**: Useful when there are multiple **similar** multi-armed bandit tasks, each with the same action space. The reward space can be either binary or continuous.
# 
# ## Main Idea
# The **Meta-TS**[1] assumes that the mean reward for each task $j$ are i.i.d sampled from some distribution parameterized by $\boldsymbol{\gamma})$. Specifically, it assumes that
# \begin{equation}
#   \begin{alignedat}{2}
# &\text{(meta-Prior)} \quad
# \quad\quad\quad    \boldsymbol{\gamma} &&\sim Q(\boldsymbol{\gamma}), \\
# &\text{(Prior)} \quad
# \; \quad\quad\quad   \boldsymbol{\mu}_j | \boldsymbol{x}_j, \boldsymbol{\gamma} &&\sim g(\boldsymbol{\mu}_j | \boldsymbol{\gamma})\\
# &\text{(Reward)} \quad
# \;    R_{j,t}(a) = Y_{j,t}(a) &&\sim f(Y_{j,t}(a)|\mu_{j,a}), 
#       \end{alignedat}
# \end{equation}. 
# 
# **Gaussian**
# 
# **Bernoulli**
# 
# 
# While the original algorithm only support the sequential interaction (i.e, a new task will be interacted with until the previous task is finished), we modified the algorithm slightly to adapt to different recommending schedules. 
# 
# ## Key Steps
# For $(j,t) = (1,1),(1,2),\cdots$:
# 1. Approximate $P(\boldsymbol{\gamma}|\mathcal{H})$ either by implementing **Pymc3** or by calculating the explicit form of the posterior distribution;
# 2. Sample $\tilde{\boldsymbol{\gamma}} \sim P(\boldsymbol{\gamma}|\mathcal{H})$;
# 3. Update $P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$ and sample $\tilde{\boldsymbol{\mu}} \sim P(\boldsymbol{\mu}|\tilde{\boldsymbol{\gamma}},\mathcal{H})$;
# 4. Take the action $A_{j,t}$ such that $A_{j,t} = argmax_{a \in \mathcal{A}} \tilde\mu_{j,a}$;
# 6. Receive reward $R_{j,t}$.

# ## References
# 
# [1] Kveton, B., Konobeev, M., Zaheer, M., Hsu, C. W., Mladenov, M., Boutilier, C., & Szepesvari, C. (2021, July). Meta-thompson sampling. In International Conference on Machine Learning (pp. 5884-5893). PMLR.
# 
# [2] Basu, S., Kveton, B., Zaheer, M., & Szepesvári, C. (2021). No regrets for learning the prior in bandits. Advances in Neural Information Processing Systems, 34, 28029-28041.
# 
