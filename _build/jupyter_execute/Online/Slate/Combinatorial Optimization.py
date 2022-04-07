#!/usr/bin/env python
# coding: utf-8

# # Online Combinatorial Optimization
# Online combinatorial optimization has a wide range of applications [1], including maximum weighted matching, ad allocation, and news page optimization, to name a few. This tutorial focuses on a typical application situation, where all chosen items will generate a separate observation, known as the semi-bandit problem [2]. 
# 
# ## Problem Setting (Combinatorial Semi-Bandit)
# 
# Suppose that, at each round t, we need to recommend at most $K$ items. Formally, the feasible set $\mathcal{A} \subseteq \{A \subseteq [N] : |A|\leq K\}$ consists of subsets that satisfy the size constraint and other application-specific constraints. The agent will sequentially choose a subset $A_t$ from $\mathcal{A}$, and then receive a separate reward $Y_{i, t}$ for each chosen item $i \in A_t$, with the overall reward defined as $R_t = \sum_{i \in A_t} Y_{i,t}$. Mathematically, for each round $t$,
# \begin{equation}
#     \begin{split}
#      Y_{i, t} &\sim \mathcal{P}(\theta_{i}), \forall i \in A_{t},\\
#     R_t &= \sum_{i \in A_t} Y_{i,t}, 
#     \end{split}
# \end{equation} where $\mathcal{P}$ can be different distributions, such as $Bernoulli(\theta_i)$ for binary outcomes, $\mathcal{N}(\theta_i, \sigma_2^2)$ for normally distributed outcomes, etc.
# 
# When the mean reward of each item is known, the optimal action can be obtained from a combinatorial optimization problem, depending on the applications. 
# 
# 
# ## Supported Algorithms
# 
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [CombTS [4]](http://proceedings.mlr.press/v80/wang18a/wang18a.pdf) | | | | 
# | [CombLinTS [5]](http://proceedings.mlr.press/v37/wen15.pdf) | | ✅ | | 
# | [CombUCB1 [2]](http://proceedings.mlr.press/v28/chen13a.pdf) | | | | 
# | [CombLinUCB [5]](http://proceedings.mlr.press/v37/wen15.pdf) | | ✅ | | 
# | [MTSS-Comb [3]](https://arxiv.org/pdf/2202.13227.pdf) | Continuous | ✅ | Scalable, Robust, accounts for inter-item heterogeneity | 

# ## Reference
# 
# [1] Sankararaman, K. A. (2016). Semi-bandit feedback: A survey of results.
# 
# [2] Chen, W., Wang, Y., and Yuan, Y. (2013). Combinatorial multi-armed bandit: General framework and applications. In International conference on machine learning, pages 151–159. PMLR.
# 
# [3] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [4] Wang, S., & Chen, W. (2018, July). Thompson sampling for combinatorial semi-bandits. In International Conference on Machine Learning (pp. 5114-5122). PMLR.
# 
# [5] Wen, Z., Kveton, B., & Ashkan, A. (2015, June). Efficient learning in large-scale combinatorial semi-bandits. In International Conference on Machine Learning (pp. 1113-1122). PMLR.
