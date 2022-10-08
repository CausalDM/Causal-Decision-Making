#!/usr/bin/env python
# coding: utf-8

# # Online Learning to Rank (Cascading Bandit)
# Online learning to rank is a popular problem, where we are interested in users' behavior toward an ordered list of items and aim to maximize the click-through rate. For example, considering the motivating example **Recommender Systems**, suppose that there are a great amount of restaurants available to be recommended, Cascading Bandit algorithms aims to find and recommend the optimal ordered list of restaurants whenever a user visits, with the ultimate objective of optimizing the overall user satisfication, which is quantified as the click-through rate. This tutorial focuses on one of the popular choice models in learning to rank--Cascading model [1]. 
# 
# ## Problem Setting
# 
# Suppose that, at each round t, we need to recommend an ordered list of $K$ items. In cascading bandits, $\mathcal{A}$ contains all the subsets of $[N]$ with length $K$, $A_t = (a_t^1, \dots, a_t^K) \in \mathcal{A}$ is a sorted list of items being displayed with any element $a_t^k \in [N]$, $\boldsymbol{Y}_t(a)$ is a vector indicating the potential outcome that will be observed if slate $a$ is recommended, with the $i$th entry equal to $1$ when the $i$th displayed item will be clicked, and $R_t(a)$ is the corresponding potential reward would be received, with $f_r[\boldsymbol{Y}_t(a)] \equiv \sum_{k \in [K]} Y_{k,t}(a)  \in \{0,1\}$, where $Y_{k,t}(a)$ is the $k$th entry of $\boldsymbol{Y}_t(a)$. 
# 
# 
# At each round $t$, the user will examine the $K$ displayed items from top to bottom and stop to click one item once she is attracted (or leave if none of them is attractive). Similar to the definition of the potential reward $R_t(a)$. denote $I_t(a)$ the index of the potential chosen item if it exists, and otherwise, let $I_t(a) = K$. To formally define the model $f$, it is useful to introduce a latent binary variable $E_{k,t}(a)$ to indicate if the $k$th displayed item will be examined by the $t$th user, and a latent variable $W_{k,t}(a)$ to indicate if the $k$th displayed item is attractive to the $t$th user. Therefore, the value of $W_{k,t}(a)$ is only visible when $k \le I_t(a)$. Let $\theta_i$ be the attractiveness of the item $i$. The key probabilistic assumption is that $W_{k, t}(a) \sim Bernoulli(\theta_{a^k}), \forall k \in [K]$. 
# 
# Mathmatically,
# \begin{equation}\label{eqn:model_cascading}
#     \begin{split}
#     W_{k, t}(a) &\sim Bernoulli(\theta_{a^k}), \forall k \in [K], \\
#     Y_{k,t}(a) &= W_{k,t}(a) E_{k,t}(a), \forall k \in [K],\\
#     E_{k,t}(a) &= [1-Y_{k-1}(a)] E_{k-1,t}(a), \forall k \in [K],\\
#     R_t(a) &= \sum_{k \in [K]} Y_{k,t}(a), 
#     \end{split}
# \end{equation} with $E_{1,t}(a) \equiv 1$. 
# 
# When $\boldsymbol{\theta}$ is known, the optimal action can be shown as any permutation of the top $K$ items with the highest attractiveness factors.
# 
# ## Supported Algorithms
# 
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [TS_Cascade [4]](http://proceedings.mlr.press/v89/cheung19a/cheung19a.pdf) | Binary | | | 
# | [CascadeLinTS [5]](https://arxiv.org/pdf/1603.05359.pdf) | Binary | ✅ | | 
# | [MTSS_Cascade [2]](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity 
# 

# ## Reference
# 
# [1] Chuklin, A., Markov, I., and Rijke, M. d. Click models for web search. Synthesis lectures on information concepts, retrieval, and services, 7(3):1–115, 2015.
# 
# [2] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [3] Kveton, B., Szepesvari, C., Wen, Z., & Ashkan, A. (2015, June). Cascading bandits: Learning to rank in the cascade model. In International Conference on Machine Learning (pp. 767-776). PMLR.
# 
# [4] Cheung, W. C., Tan, V., & Zhong, Z. (2019, April). A Thompson sampling algorithm for cascading bandits. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 438-447). PMLR.
# 
# [5] Zong, S., Ni, H., Sung, K., Ke, N. R., Wen, Z., & Kveton, B. (2016). Cascading bandits for large-scale recommendation problems. arXiv preprint arXiv:1603.05359.
