#!/usr/bin/env python
# coding: utf-8

# # Dynamic Assortment Optimization (Multinomial Logit Bandit)
# Assortment optimization [1] is a long-standing problem that aims to solve the most profitable subset of items to offer, especially when substitution effects exist. For example, consider the motivating example **Recommender Systems**. Assume that there are a large number of products available to be recommended, each of which has a price. Multinomial Logit Bandit algorithms aim to find and recommend the optimal list of products whenever a user visits, with the ultimate goal of optimizing the overall income, which depends not only on the probability of at least one product being purchased, but also on the price of the purchased product. The Multinomial Logit (MNL) model [2] is arguably the most popular to characterize the users' response. This tutorial will focus on the bandit version of the MNL model.
# 
# 
# ## Problem Setting
# 
# **Standard Offering:** Suppose that, at each round t, we need to recommend at most $K$ items. In assortment optimization, the agent needs to offer a subset (assortment) $A_t \in \mathcal{A} = \{A\subseteq[N]:|A|\leq K\}$, and the customer will then choose either one of them or no-purchase which is denoted as item $0$. Let $\boldsymbol{Y}_t(a)=(Y_{0,t}(a),\cdots,Y_{N,t}(a))^{T}$ be an indicator vector of length $N+1$, where $Y_{i,t}(a)$ equals to $1$ if the item $i$ would be chosen if the action $a$ is offered. Let $\boldsymbol{\eta} = (\eta_0, \eta_1, \dots,\eta_{N})^{T}$, where $\eta_k$ is the revenue of the item $k$.  Conventionally, $\eta_0 = 0$. The potential revenue that would be collected, if action $a$ is offered in round $t$, is then $R_t(a) = \sum_{i\in a} Y_{i,t}(a) \eta_{i}$. In an MNL bandit, each item $i$ has an utility factor $v_i$, and 
# the choice behaviour is characterized by a multinomial distribution
# \begin{equation}\label{MNLdist}
#     \boldsymbol{Y}_t(a) \sim Multinomial(1, \frac{v_i \mathcal{I}(i \in \{0\}\cup a)}{1 + \sum_{j \in a} v_i}),
# \end{equation} 
# with the convention that $v_0 = 1$. Note that $v_{i}$ will be a determined via a deterministic function of the item-specific parameter $\theta_{i}$.
# 
# **Epoch-type Offering:** Alternatively, since the direct inference under the above MNL model is intractable, an epoch-based algorithmic structure is introduced [4,5,7] for computational efficiency. Specifically, for each epoch $l$, we keep offering the same assortment $A^l$ until the no-purchase appears. Let $\theta_i = (v_i+1)^{-1}$. Denote the potential number of purchases for the item $i \in a$ in each epoch as $Y_{i}^l(a)$, if the slate $a$ is recommended. Under this setup, it is easy to show that 
# \begin{equation}
#     Y_{i}^l(a) \sim Geometric(\theta_i), \forall i \in a,
# \end{equation} 
# and then the corresponding reward of epoch $l$ is $R^l(a) = \sum_{i\in a}Y_{i}^l(a)\eta_{i}$ [4]. 
# 
# No matter what offering type is employed, for each decision round $t$, when $v_i$'s are known, the optimal assortment $A_{t}$ can be determined efficiently through linear programming [4], such that
# \begin{equation}
#     A_{t} = arg max_{a \in \mathcal{A}} E(R_t(a) \mid\boldsymbol{v})=argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}v_{i}}{1+\sum_{j\in a} v_{j}}.
# \end{equation}
# 
# 
# ## Supported Algorithms
# 
# | algorithm | $Y_{i,t}$ | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [TS_MNL [4]](http://proceedings.mlr.press/v65/agrawal17a/agrawal17a.pdf) | Binary | | | 
# | [TS_Contextual_MNL [5]](https://proceedings.neurips.cc/paper/2019/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf) | Binary | ✅ | | 
# | [MTSS_MNL [3]](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
# 
# ## Real Data
# Add a description of the MovieLens Dataset

# ## Reference
# 
# [1] Pentico, D. W. (2008). The assortment problem: A survey. European Journal of Operational Research, 190(2), 295-309.
# 
# [2] Luce, R. D. (2012). Individual choice behavior: A theoretical analysis. Courier Corporation.
# 
# [3] Wan, R., Ge, L., & Song, R. (2022). Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework. arXiv preprint arXiv:2202.13227.
# 
# [4] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2017, June). Thompson sampling for the mnl-bandit. In Conference on Learning Theory (pp. 76-78). PMLR.
# 
# [5] Oh, M. H., & Iyengar, G. (2019). Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32.
# 
# [6] Agrawal, S., Avadhanula, V., Goyal, V., & Zeevi, A. (2019). Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5), 1453-1485.
# 
# [7] Ou, M., Li, N., Zhu, S., & Jin, R. (2018). Multinomial logit bandit with linear utility functions. arXiv preprint arXiv:1805.02971.

# In[ ]:




