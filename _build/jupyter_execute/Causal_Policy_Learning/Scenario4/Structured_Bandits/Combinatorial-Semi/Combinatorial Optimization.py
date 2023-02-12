#!/usr/bin/env python
# coding: utf-8

# # Online Combinatorial Optimization (Combinatorial Semi-Bandit)
# Online combinatorial optimization has a wide range of applications [1], including maximum weighted matching, ad allocation, and news page optimization, to name a few.  It is also common that all chosen items will generate a separate observation, known as the semi-bandit problem [2]. Taking the motivating example of **Online Ad**, suppose we need to decide which group of target customers to show the advertisement to. Combinatorial semi-bandit algorithms aim to find and recommend the optimal group of potential customers to advertisers, with the ultimate goal of optimizing the overall number of customers attracted by the advertisement. This tutorial will concentrate on typical application scenarios such as online Ad.
# 
# ## Problem Setting
# 
# Suppose that, at each round t, we need to recommend at most $K$ items. Formally, the feasible set $\mathcal{A} \subseteq \{A \subseteq [N] : |A|\leq K\}$ consists of subsets that satisfy the size constraint as well as other application-specific constraints. The agent will sequentially choose a subset $A_t$ from $\mathcal{A}$, and then receive a separate reward $Y_{i, t}$ for each chosen item $i \in A_t$, with the overall reward defined as $R_t = \sum_{i \in A_t} Y_{i,t}$. Let $Y_{i, t}(a)$ denote the potential outcome of item $i \in a$ if a slate of items $a$ is played at round $t$. Similarly, $R_t(a)$ denotes the reward that would be received if the slate $a$ is palyed. Mathematically, for each round $t$,
# \begin{equation}
#     \begin{split}
#      Y_{i,t}(a) &\sim \mathcal{P}(\theta_{i}), \forall i \in a,\\
#     R_t(a) &= \sum_{i \in a} Y_{i,t}(a), 
#     \end{split}
# \end{equation} 
# where $\mathcal{P}$ can be different distributions, such as $Bernoulli(\theta_i)$ for binary outcomes, $\mathcal{N}(\theta_i, \sigma_2^2)$ for normally distributed outcomes, etc.
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
# | [MTSS_Comb [3]](https://arxiv.org/pdf/2202.13227.pdf) | Continuous | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
# 
# ## Real Data
# 
# Add a description of the Adult dataset

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
