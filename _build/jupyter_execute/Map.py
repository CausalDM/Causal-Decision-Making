#!/usr/bin/env python
# coding: utf-8

# # Overview
# ## What we included?
# ---
# The diagram below depicts the overall structure of this book, which is comprised of three primary components: **Causal Structure Learning**, **Causal Policy Learning**, and **Causal Machine Learning**. Specifically, in the chapter [**Causal Structure Learning (CSL)**](#SL), we present state-of-the-art techniques for learning the skeleton of causal relationships among input variables. When a causal structure is known, the second chapter of [**Causal Machine Learning (CML)**](#ML) introduces approaches making causal inference. Finally, the [**Causal Policy Learning (CPL)**](#PL) chapter introduces diverse policy learners to learn optimal policies and evaluate various policies of interest.
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
# ## <a name="ML"></a> Causal Machine Learning (CML)
# ---
# 
# ## <a name="PL"></a> Causal Policy Learning (CPL)
# ---
# This chapter focuses on four common data dependence structures in decision making, including [**I.I.D.**](#Case1), [**Adaptive Decision Making with Independent States (ADMIS)**](#Case2), [**Adaptive Decision Making with State Transition (ADMST)**](#Case3), and [**All Others**](#Case4). 
# 
# ### <a name="Case1"></a> Scenario 1: I.I.D
# ![Case1.png](Case1.png)
# 
# As the figure illustrated, observations in Scenario 1 are i.i.d. sampled. For each observation, there are three components, $S_i$ is the context information if there is any, $A_i$ is the action taken, and $R_i$ is the reward received. When there is contextual information, the action would be affected by the contextual information, while the final reward would be affected by both the contextual information and the action. There are two types of problems are widely studied in this context: the **Single-Stage Dynamic Treatment Regime (DTR)** and the **Offline Bandits**[1]. In this book, we mainly focus on the Single-Stage DTR by providing methods for policy evaluation and policy optimization, with a detailed map in [Appendix A](#SingleDTR)
# 
# ### <a name="Case2"></a> Scenario 2: Adaptive Decision Making with Independent States (ADMIS)
# ![Case2.png](Case2.png)
# The Case2 setting is widely examined in the online decision making literature, especially the bandits, where the treatment policy is time-adaptive. Specifically, $H_t$ includes all the previous observations up to time $t$ and is used to update the policy at time $t$ (i.e., $\pi_t$). At time $t$, only the action $A_t$ would be affected by the history-dependent policy $\pi_t$. While $S_t$ is i.i.d sampled from the correponding distribution, $R_t$ is influenced by both $A_t$ and $S_t$. Finally, the new observation $(S_t,A_t,R_t)$, in conjunction with all previous observations, would then  be formulated as $H_{t+1}$ and affect $\pi_{t+1}$. A structure that lacks contextual information $S_t$ is also very common. In this book, a list of bandits algorithms would be introduced, with a detailed map in [Appendix B](#Bandits).
# 
# ### <a name="Case3"></a> Scenario 3: Adaptive Decision Making with State Transition (ADMST)
# ![Case3.png](Case3.png)
# This structure is the well-known Markov Decision Process (MDP). Typically, MDP problems will involve $N$ i.i.d. trajectories. Here, we index the trajectory as $i$ (i.e., $A_{i,t}$ is the treatment received for trajectory $i$ at time $t$). For each trajectory, the causal relationship between the observation at time $t$ and the observation at $t+1$ is illustrated in Fig. 4. Specifically, the state transition is explicitly depicted. While $A_t$ is only affected by $S_t$, while both $R_t$ and $S_{t+1}$ would be affected by $(S_t,A_t)$. Given $S_{t}, A_t$, both $R_t$ and $S_{t+1}$ are independent of previous observations up to time $t$, which is the standard assumption of MDP problems. List of related learning methods would be introduced, with a map in [Appendix C](#MDP).
# 
# ### <a name="Case4"></a> Scenario 4: All Others
# ![Case4.png](Case4.png)
# 
# Taking all the possible causal relationship into account, this scenario considers all the data dependence struacture that are not included in the previous three classical scenarios, including **Multi-Stage DTR**, **Partially Observable MDP (POMDP)**, **Confounded MDP**, ......
# 
# 
# ## Appendix
# ---
# ### <a name="SingelDTR"></a> A. Scenario 1
# ![Scenario1.png](Scenario1.png)
# 
# ### <a name="Bandits"></a> B. Scenario 2
# ![Online.png](Online.png)
# 
# ### <a name="MDP"></a> C. Scenario 3
# ![Scenario3.png](Scenario3.png)
# 
# ### D. Scenario 4
# 
# | Algorithm | Treatment Type | Outcome Type | Evaluation? | Optimization? | C.I.? | Advantages |
# |:-|:-:|:-:|:-:|:-:|:-:|:-:|
# | [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| ||
# | [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean)  |✅|✅| ||
# 
# 
# ## Reference
# [1] Dudík, M., Langford, J., & Li, L. (2011). [Doubly robust policy evaluation and learning](https://arxiv.org/pdf/1103.4601.pdf). arXiv preprint arXiv:1103.4601.
# 
# 
# 
# 
# 
# ## Supported Offline Algorithms
# ---
# | Algorithm | Treatment Type | Outcome Type | Single Stage? | Multiple Stages? | Infinite Horizon? | Evaluation? | Optimization? | C.I.? | Advantages |
# |:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| |✅|✅| ||
# | [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean) |✅|✅|  |✅|✅| ||
# | [OWL](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2012.695674?casa_token=bwkVvffpyFcAAAAA:hlN58Fbz59blLj5npZFjEQD-HkPeMevEN5pWWLu_vuIVxPWl5aYShgCVHUVeODAfj6Pr8DpzGFlPZ1E) | Discrete | Continuous (Mean) |✅|❗BOWL| |✅|❗TODO| ||
# | [Quatile-OTR](https://doi.org/10.1080/01621459.2017.1330204) | Discrete | Continuous (Quantiles) |✅✏️|  |  |✅✏️|✅✏️| ||
# | [Deep Jump Learner](https://proceedings.neurips.cc/paper/2021/file/816b112c6105b3ebd537828a39af4818-Paper.pdf) | Continuous | Continuous/Discrete |✅|  |  |✅| ✅| | Flexible to implement & Fast to Converge|
# |**TODO** | | ||  |  ||| ||
# | Policy Search| | ||  |  ||| ||
# | Concordance-assisted learning| | ||  |  ||| ||
# | Entropy learning | | ||  |  ||| ||
# | **Continuous Action Space (Main Page)** | | ||  |  ||| ||
# | Kernel-Based Learner | | ||  |  ||| ||
# | Outcome Learning | | ||  |  ||| ||
# | **Miscellaneous (Main Page)** | | ||  |  ||| ||
# | Time-to-Event Data | | ||  |  ||| ||
# | Adaptively Collected Data | | ||  |  ||| ||
# 
# 
# ## Supported Online Algorithms
# | algorithm | Reward | with features? | Advantage | Progress|
# |:-|:-:|:-:|:-:|:-:|
# | **Single-Item Recommendation** || | |95% (proofreading)|
# | [$\epsilon$-greedy]() | Binary/Gaussian | |Simple| 100%|
# | [TS](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/notes_slides/sampling/TS_Tutorial.pdf) | Binary/Guaasian | | |100%|
# | [LinTS](http://proceedings.mlr.press/v28/agrawal13.pdf) | Gaussian | ✅ | | 90% notebook|
# | [GLMTS](http://proceedings.mlr.press/v108/kveton20a/kveton20a.pdf) | GLM | ✅ | | 0% |
# | [UCB1](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf) | Binary/Gaussian | | |  100%|
# | [LinUCB](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758?casa_token=CJjeIziLmjEAAAAA:CkRvgHQNqpy10rzcUP5kx31NWJmgSldd6zx8wZxskZYCoCc8v7EDIw3t3Gk1_6mfurqQTqRZ7fVA) | Guassian | ✅ | | 50% (code ready)|
# | [UCB-GLM](http://proceedings.mlr.press/v70/li17c/li17c.pdf) | GLM | ✅ | | 0%|
# |*Multi-Task*|| | |0% (main page)|
# |Meta-TS|| | |50% (code almost ready)|
# |MTSS|| | |50% (code almost ready)|
# | **Slate Recommendation** || | |95% (proofreading)|
# | *Online Learning to Rank* || | |95% (proofreading)|
# | [TS-Cascade](http://proceedings.mlr.press/v89/cheung19a/cheung19a.pdf) | Binary | | | 40% code+notebook|
# | [CascadeLinTS](https://arxiv.org/pdf/1603.05359.pdf) | Binary | ✅ | |  40% code+notebook|
# | [MTSS-Cascade](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity | 95% (proofreading)|
# | *Online Combinatorial Optimization* || | |95% (proofreading)|
# | [CombTS](http://proceedings.mlr.press/v80/wang18a/wang18a.pdf) | | | | 50% code ready|
# | [CombLinTS](http://proceedings.mlr.press/v37/wen15.pdf) | | ✅ | | 50% code ready
# | [MTSS-Comb](https://arxiv.org/pdf/2202.13227.pdf) | Continuous | ✅ | Scalable, Robust, accounts for inter-item heterogeneity | 95% (proofreading)|
# | *Dynamic Assortment Optimization* || | |95% (proofreading)|
# | [MNL-Thompson-Beta](http://proceedings.mlr.press/v65/agrawal17a/agrawal17a.pdf) | Binary | | |  40% code+notebook|
# | [TS-Contextual-MNL](https://proceedings.neurips.cc/paper/2019/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf) | Binary | ✅ | |  40% code+notebook|
# | [MTSS-MNL](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |95% (proofreading)|
# | [UCB-MNL [6]](https://pubsonline.informs.org/doi/pdf/10.1287/opre.2018.1832?casa_token=6aWDZ292SSsAAAAA:KAG0_j813jxeL6PVNI1dcdLv_CHD7oQ6SKinqxcoq0pC2mX5Q2qGgyYvE8esMSXZPlqOanCPOQ) | Binary | | |80% (notebook)| 
# | [LUMB [7]](https://arxiv.org/pdf/1805.02971.pdf) | Binary | ✅ | | 50% code ready|
# | **TODO** || | ||
# | environment with real data || | |50% MAB is ready|
# | overall simulation || | ||

# In[ ]:




