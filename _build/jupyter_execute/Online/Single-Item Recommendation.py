#!/usr/bin/env python
# coding: utf-8

# # Single-Item Recommendation
# 
# The bandit problems have received increasing attention recently and have been widely applied to areas such as clinical trials [1], finance [2], and recommendation systems [3], among others. The most classical version of it is the multi-armed bandit (MAB) [4], where an agent will sequentially select an item (arm) from a few and then receive a random reward for the item selected. Since the reward distributions are unknown in most real applications, the central task of a MAB algorithm is to learn the distributions from feedback received and find the optimal item to maximize the cumulative rewards or, equivalently, to minimize the cumulative regret. This chapter focuses on the MAB problems by illustrating a group of classical algorithms to tackle the well-known exploration-exploitation trade-off.
# 
# ## Problem Setting
# Let $T$ be the total number of rounds, and $K$ be the number of arms (actions to be selected). The agent would choose one arm at each round $t = 1, \dots, T$. Then the agent will receive the corresponding stochastic reward $R_t$ from the environment. Denote the expected reward for each arm $i$ as $r_{i}$. Since, in most real applications, such a reward distribution is always unknown, the agent needs to learn the reward distribution from feedback received. Overall, the objective is to find a bandit algorithm to maximize the cumulative Reward $\sum_{t=1}^{T}R_{t}$.
# 
# MAB has been extensively studied and widely applied to different areas, including healthcare, recommender system, and finance, to name a few. See [4] for a detailed review of MAB and [5] for a survey of practical applications. Among them, the ultimate goal of a learning algorithm is always to strike a good balance between exploration (try an unfamiliar action to learn more information) and exploitation (take the action that has the highest estimated reward so far) so as to maximize the cumulative reward. In the following, we will briefly illustrate three popular and classical categories of algorithms to handle the exploration-exploitation trade-off: i) $\epsilon$-greedy, ii) Upper Confidence Bound (UCB), and iii) Thompson Sampling (TS). 

# ## Claasical Methods
# ### $\epsilon$-Greedy
# An intuitive algorithm to incorporate the exploration and exploitation is $\epsilon$-Greedy, which is simple and widely used [6]. Specifically, at each round $t$, we will select a random action with probability $\epsilon$, and select an action with the highest estimated mean reward based on the history so far with probability $1-\epsilon$. Here the parameter $\epsilon$ is pre-specified. A more adaptive variant is $\epsilon_{t}$-greedy, where the probability of taking a random action is defined as a decreasing function of $t$. Auer et al. [7] showed that $\epsilon_{t}$-greedy performs well in practice with $\epsilon_{t}$ decreases to 0 at a rate of $\frac{1}{t}$.
# 
# #### Supported Algorithms
# 
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [$\epsilon$-greedy]() | Binary/Gaussian | |Simple| 
# 
# 
# ### Thompson Sampling
# Thompson Sampling, also known as posterior sampling, solves the exploration-exploitation dilemma by selecting an action according to its posterior distribution [8].  At each round $t$, the agent sample the rewards from the corresponding posterior distributions and then select the action with the highest sampled reward greedily. It has been shown that, when the true reward distribution is known, a TS algorithm with the true reward distribution as the prior is nearly optimal [9]. However, such a distribution is always unknown in practice. Therefore, one of the major objectives of TS-based algorithms is to find an informative prior to guide the exploration.
# 
# #### Supported Algorithms
# 
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [TS [8]](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/notes_slides/sampling/TS_Tutorial.pdf) | Binary/Guaasian | | | 
# | [LinTS [13]](http://proceedings.mlr.press/v28/agrawal13.pdf) | Gaussian | ✅ | | 
# | [GLMTS [12]](http://proceedings.mlr.press/v108/kveton20a/kveton20a.pdf) | GLM | ✅ | | 
# 
# ### Upper Confidence Bounds 
# As the name suggested, the UCB algorithm estimates the upper confidence bound $U_{i}^{t}$ of the mean rewards based on the observations and then choose the action has the highest estimates. The class of UCB-based algorithms is firstly introduced by Auer et al. [7]. Generally, at each round $t$, $U_{i}^{t}$ is calculated as the sum of the estimated reward (exploitation) and the estimated confidence radius (exploration) of item $i$ based on $\mathcal{H}_{t}$. Then, $A_{t}$ is selected as 
# \begin{equation}
#     A_t = argmax_{a \in \mathcal{A}} E(R_t \mid a,\{ U_{i}^{t}\}_{i=1}^{N}).
# \end{equation} As an example, \textbf{UCB1} \citep{auer2002finite} estimates the confidence radius as $\sqrt{\frac{2log(t)}{\text{\# item $i$ played so far}}}$. Doing so, either the item with a large average reward or the item with limited exploration will be selected.
# 
# #### Supported Algorithms
# 
# | algorithm | Reward | with features? | Advantage |
# |:-|:-:|:-:|:-:|
# | [UCB1 [7]](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf) | Binary/Gaussian | | | 
# | [LinUCB [10]](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758?casa_token=CJjeIziLmjEAAAAA:CkRvgHQNqpy10rzcUP5kx31NWJmgSldd6zx8wZxskZYCoCc8v7EDIw3t3Gk1_6mfurqQTqRZ7fVA) | Guassian | ✅ | | 
# | [UCB-GLM [11]](http://proceedings.mlr.press/v70/li17c/li17c.pdf) | GLM | ✅ | | 
# 
# 
# 

# ## Reference
# [1] Durand, A., Achilleos, C., Iacovides, D., Strati, K., Mitsis, G. D., and Pineau, J. (2018). Contextual bandits for adapting treatment in a mouse model of de novo carcinogenesis. In Machine learning for healthcare conference, pages 67–82. PMLR.
# 
# [2] Shen, W., Wang, J., Jiang, Y.-G., and Zha, H. (2015). Portfolio choices with orthogonal bandit learning. In Twenty-fourth international joint conference on artificial intelligence.
# 
# [3] Zhou, Q., Zhang, X., Xu, J., and Liang, B. (2017). Large-scale bandit approaches for recommender systems. In International Conference on Neural Information Processing, pages 811–821. Springer.
# 
# [4] Slivkins, A. (2019). Introduction to multi-armed bandits. arXiv preprint arXiv:1904.07272.
# 
# [5] Bouneffouf, D. and Rish, I. (2019). A survey on practical applications of multi-armed and contextual bandits. arXiv preprint arXiv:1904.10040.
# 
# [6] Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press
# 
# [7] Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256.
# 
# [8] Russo, D., Van Roy, B., Kazerouni, A., Osband, I., and Wen, Z. (2017). A tutorial on thompson sampling. arXiv preprint arXiv:1707.0203
# 
# [9] Lattimore, T. and Szepesv´ari, C. (2020). Bandit algorithms. Cambridge University Press.
# 
# [10] Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661–670
# 
# [11] Li, L., Lu, Y., and Zhou, D. (2017). Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning, pages 2071–2080. PMLR.
# 
# [12] Kveton, B., Zaheer, M., Szepesvari, C., Li, L., Ghavamzadeh, M., and Boutilier, C. (2020). Randomized exploration in generalized linear bandits. In International Conference on Artificial Intelligence and Statistics, pages 2066–2076. PMLR.
# 
# [13] Agrawal, S. and Goyal, N. (2013). Thompson sampling for contextual bandits with linear payoffs. In International conference on machine learning, pages 127–135. PMLR.
# 
