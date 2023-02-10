#!/usr/bin/env python
# coding: utf-8

# # Overview
# ## Introduction
# ---
# 
# 
# 
# ## What to expect?
# ---
# The diagram below depicts the overall structure of this book, which is comprised of three primary components: **Causal Structure Learning**, **Causal Policy Learning**, and **Causal Effect Learning**. Specifically, in the chapter [**Causal Structure Learning (CSL)**](#SL), we present state-of-the-art techniques for learning the skeleton of causal relationships among input variables. When a causal structure is known, the second chapter of [**Causal Effect Learning (CEL)**](#ML) introduces approaches making causal inference. Finally, the [**Causal Policy Learning (CPL)**](#PL) chapter introduces diverse policy learners to learn optimal policies and evaluate various policies of interest.
# 
# ![Overall.png](Overall.png)
# 
# Following is a brief summary of the contents of each chapter.
# 
# ## <a name="SL"></a> Causal Structure Learning (CSL)
# ---
# This chapter discusses three classical techniques for learning causal graphs, each with its own merits and downsides.
# 
# | Learners      Type    | Supported Model  | Noise Required for Training |   Complexity     | Scale-Free? | Learners Example |
# |-----------------------|------------------|-----------------------------|------------------|-------------|------------------|
# |    Testing based      |      Models 1    |          Gaussian           |     $O(p^q)$     |     Yes     |         PC       |
# |    Functional based   |   Models 1 & 2   |        non-Gaussian         |     $O(p^3)$     |     Yes     |       LiNGAM     |
# |    Score based        |   Models 1 & 3   |    Gaussian/non-Gaussian    |     $O(p^3)$     |     No      |       NOTEARS    |
# 
# *$p$ is the number of nodes in $\mathcal{G}$, and $q$ is the max number of nodes adjacent to any nodes in $\mathcal{G}$.*
# 
# ## <a name="ML"></a> Causal Effect Learning (CEL)
# ---
# 
# ## <a name="PL"></a> Causal Policy Learning (CPL)
# ---
# This chapter focuses on six common data dependence structures in decision making, including [**I.I.D.**](#Case1), [**Offline Reinforcement Learning**](#Case2), [**Multiple-Stage DTR**](#Case3), [**Adaptive Decision Making with Independent States (ADMIS)**](#Case4), [**Online Reinforcement Learning**](#Case5), and [**All Others**](#Case6). The similarities and differences between four scenarios are summarized as follows.
# 
# ![Causal_DM_Causal_Structure_Table.png](Causal_DM_Causal_Structure_Table.png)
# 
# ### <a name="Case1"></a> Scenario 1: I.I.D
# As the figure illustrated, observations in Scenario 1 are i.i.d. sampled. For each observation, there are three components, $S_i$ is the context information if there is any, $A_i$ is the action taken, and $R_i$ is the reward received. When there is contextual information, the action would be affected by the contextual information, while the final reward would be affected by both the contextual information and the action. A classical class of problems that are widely studied in this context is the **Single-Stage Dynamic Treatment Regime (DTR)**[1]. In this book, we mainly focus on methods for policy evaluation and policy optimization for Single-Stage DTR, with a detailed map in [Appendix A](#SingleDTR)
# 
# ### <a name="Case2"></a> Scenario 2: Offline Reinforcement Learning
# The Scenario 2 is well-known as Markov Decision Process (MDP), whose main characteristic is the Markovian state transition. In particular, while $A_t$ is only affected by $S_t$, both $R_t$ and $S_{t+1}$ would be affected by $(S_t,A_t)$. Given $S_{t}, A_t$, a standard assumption of MDP problems is that $R_t$ and $S_{t+1}$ are independent of previous observations. A list of related learning methods will be introduced, with a map in [Appendix B](#MDP).
# 
# ### <a name="Case3"></a> Scenario 3: Multiple-Stage DTR
# When a history-independent policy is applied, the Scenario 3 takes all the possible causal relationships into account and is well-known as the multiple-stage DTR problem [1]. In this book, we introduce two classical learning methods, including Q-learning and A-learning (See a map in [Appendix C](#MultiDTR))
# 
# ### <a name="Case4"></a> Scenario 4: Adaptive Decision Making with Independent States (ADMIS)
# The Scenario 4 setting is widely examined in the online decision making literature, especially the bandits, where the treatment policy is time-adaptive. Specifically, $H_{t-1}$ includes all the previous observations up to time $t-1$ (include observations at time $t-1$) and is used to update the action policy at time $t$, and therefore affect the action $A_t$. While $S_t$ is i.i.d sampled from the correponding distribution, $R_t$ is influenced by both $A_t$ and $S_t$. Finally, the new observation $(S_t,A_t,R_t)$, in conjunction with all previous observations, would then be formulated as $H_{t+1}$ and affect $A_{t+1}$ only. A structure that lacks contextual information $S_t$ is also very common. In this book, a list of bandits algorithms would be introduced, with a detailed map in [Appendix D](#Bandits).
# 
# ### <a name="Case5"></a> Scenario 5: Online Reinforcement Learning
# Building upon the MDP structure, when an adaptive policy is applied, the Scenario 5 clearly depicts the data-generating process, in which  $S_t$ follows the Markovian state transition and $A_t$ would be affected by all previous observations $H_{t-1}$.
# 
# ### <a name="Case6"></a> Scenario 6: All others
# 
# 
# ## Appendix
# ---
# ### <a name="SingelDTR"></a> A. Scenario 1
# | Algorithm | Treatment Type | Outcome Type | Single Stage? | Multiple Stages? | Infinite Horizon? | Evaluation? | Optimization? | C.I.? | Advantages |
# |:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| |✅|✅| ||
# | [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean) |✅|✅|  |✅|✅| ||
# | [OWL](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2012.695674?casa_token=bwkVvffpyFcAAAAA:hlN58Fbz59blLj5npZFjEQD-HkPeMevEN5pWWLu_vuIVxPWl5aYShgCVHUVeODAfj6Pr8DpzGFlPZ1E) | Discrete | Continuous (Mean) |✅|❗BOWL| |✅|❗TODO| ||
# | [Quatile-OTR](https://doi.org/10.1080/01621459.2017.1330204) | Discrete | Continuous (Quantiles) |✅✏️|  |  |✅✏️|✅✏️| ||
# | [Deep Jump Learner](https://proceedings.neurips.cc/paper/2021/file/816b112c6105b3ebd537828a39af4818-Paper.pdf) | Continuous | Continuous/Discrete |✅|  |  |✅| ✅| | Flexible to implement & Fast to Converge|
# | Kernel-Based Learner | | ||  |  ||| ||
# | Outcome Learning | | ||  |  ||| ||
# 
# <div>
# <img src="Scenario1.png" align="center" width="500"/>
# </div>
# 
# ### <a name="MDP"></a> B. Scenario 2
# <div>
# <img src="Scenario2.png" align="center"  width="400"/>
# </div>
# 
# ### <a name="MultiDTR"></a> C. Scenario 3
# | Algorithm | Treatment Type | Outcome Type | Evaluation? | Optimization? | C.I.? | Advantages |
# |:-|:-:|:-:|:-:|:-:|:-:|:-:|
# | [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| ||
# | [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean)  |✅|✅| ||
# 
# <div>
# <img src="Scenario3.png" align="center"  width="300"/>
# </div>
# 
# ### <a name="Bandits"></a> D. Scenario 4
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | **Multi-Armed Bandits** || | |
# | [$\epsilon$-greedy]() | Binary/Gaussian | |Simple|
# | [TS](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/notes_slides/sampling/TS_Tutorial.pdf) | Binary/Guaasian | | |
# | [UCB1](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf) | Binary/Gaussian | | |
# | **Contextual Bandits** || | |
# | LinTS | [GLM](http://proceedings.mlr.press/v108/kveton20a/kveton20a.pdf)/[Gaussian](http://proceedings.mlr.press/v28/agrawal13.pdf) | ✅ | |
# | LinUCB | [GLM](http://proceedings.mlr.press/v70/li17c/li17c.pdf)/[Guassian](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758?casa_token=CJjeIziLmjEAAAAA:CkRvgHQNqpy10rzcUP5kx31NWJmgSldd6zx8wZxskZYCoCc8v7EDIw3t3Gk1_6mfurqQTqRZ7fVA) | ✅ | |
# | **Meta Bandits** || | |
# |Meta-TS|| | |
# |MTSS|| | |
# | **Structured Bandits** || | |
# | *Learning to Rank* || | |
# | [TS-Cascade](http://proceedings.mlr.press/v89/cheung19a/cheung19a.pdf) | Binary | | |
# | [CascadeLinTS](https://arxiv.org/pdf/1603.05359.pdf) | Binary | ✅ | |
# | [MTSS-Cascade](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
# | *Combinatorial Optimization* || | |
# | [CombTS](http://proceedings.mlr.press/v80/wang18a/wang18a.pdf) | | | |
# | [CombLinTS](http://proceedings.mlr.press/v37/wen15.pdf) | | ✅ | |
# | [MTSS-Comb](https://arxiv.org/pdf/2202.13227.pdf) | Continuous | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
# | *Assortment Optimization* || | |
# | [MNL-Thompson-Beta](http://proceedings.mlr.press/v65/agrawal17a/agrawal17a.pdf) | Binary | | |
# | [TS-Contextual-MNL](https://proceedings.neurips.cc/paper/2019/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf) | Binary | ✅ | |
# | [MTSS-MNL](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
# 
# <img src="Scenario4.png" align="center"  width="900"/>
# 
# ## Reference
# [1] Tsiatis, A. A., Davidian, M., Holloway, S. T., & Laber, E. B. (2019). Dynamic treatment regimes: Statistical methods for precision medicine. Chapman and Hall/CRC.
# 

# In[ ]:




