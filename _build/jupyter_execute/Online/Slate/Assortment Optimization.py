#!/usr/bin/env python
# coding: utf-8

# # Dynamic Assortment Optimization
# Assortment optimization [1] is a long-standing problem that aims to solve the most profitable subset of items to offer, especially when substitution effects exist. The Multinomial Logit (MNL) model [2] is arguably the most popular to characterize the users' response. This tutorial will focus on the bandit version of the MNL model.
# 
# ## Problem Setting (Multinomial Logit Bandit)
# 
# **Standard Offering:** Suppose that, at each round t, we need to recommend at most $K$ items. In assortment optimization, the agent needs to offer a subset (assortment) $A_t \in \mathcal{A} = \{A\subseteq[N]:|A|\leq K\}$, and the customer will then choose either one of them or no-purchase which is denoted as item $0$. Let $\boldsymbol{Y}_t=(Y_{0,t},\cdots,Y_{N,t})^{T}$ be an indicator vector of length $N+1$, where $Y_{i,t}$ equals to $1$ if the item $i$ is chosen. Let $\boldsymbol{eta} = (\eta_0, \eta_1, \dots,\eta_{N})^{T}$, where $\eta_k$ is the revenue of the item $k$.  Conventionally, $\eta_0 = 0$. The collected revenue in round $t$ is then $R_t = \sum_{i\in A_t} Y_{i,t} \eta_{i}$. In an MNL bandit, each item $i$ has an utility factor $v_i$, and 
# the choice behaviour is characterized by a multinomial distribution
# \begin{equation}\label{MNLdist}
#     \boldsymbol{Y}_t \sim Multinomial(1, \frac{v_i \mathcal{I}(i \in \{0\}\cup A_t)}{1 + \sum_{j \in A_t} v_i}),
# \end{equation} with the convention that $v_0 = 1$. Note that $v_{i}$ will be a determined via a deterministic function of the item-specific parameter $\theta_{i}$.
# 
# **Epoch-type Offering:** Alternatively, since the direct inference under the above MNL model is intractable, an epoch-based algorithmic structure is introduced [4,5,7] for computational efficiency. Specifically, for each epoch $l$, we keep offering the same assortment $A^l$ until the no-purchase appears. Let $v_i = \theta_i^{-1}-1$. Denote the number of purchases for the item $i$ in each epoch as $Y_{i}^l$. Under this setup, it is easy to show that 
# \begin{equation}
#     Y_{i}^l \sim Geometric(\theta_i), \forall i \in A^l,
# \end{equation} and then the corresponding reward of epoch $l$ is $R^l = \sum_{i\in A^{l}}Y_{i}^l\eta_{i}$ [4]. 
# 
# No matter what offering type is offered, for each round $t$, when $v_i$'s are known, the optimal assortment $A_{t}$ can be determined efficiently through linear programming [4], such that
# \begin{equation}
#     A_{t} = argmax_{a \in \mathcal{A}} \frac{\sum_{i\in a}\eta_{i}v_{i}}{1+\sum_{j\in a} v_{j}}.
# \end{equation}
# 
# 
# ## Supported Algorithms
# 
# | algorithm | $Y_{i,t}$ | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [MNL-Thompson-Beta [4]](http://proceedings.mlr.press/v65/agrawal17a/agrawal17a.pdf) | Binary | | | 
# | [TS-Contextual-MNL [5]](https://proceedings.neurips.cc/paper/2019/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf) | Binary | ✅ | | 
# | [UCB-MNL [6]](https://pubsonline.informs.org/doi/pdf/10.1287/opre.2018.1832?casa_token=6aWDZ292SSsAAAAA:KAG0_j813jxeL6PVNI1dcdLv_CHD7oQ6SKinqxcoq0pC2mX5Q2qGgyYvE8esMSXZPlqOanCPOQ) | Binary | | | 
# | [LUMB [7]](https://arxiv.org/pdf/1805.02971.pdf) | Binary | ✅ | | 
# | [MTSS-MNL [3]](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |

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




