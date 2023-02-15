#!/usr/bin/env python
# coding: utf-8

# # HTE Estimation 
# In the previous section, we've introduced the estimation of average treatment effect, where we aims to estimate the difference of potential outcomes by executing action $A=1$ v.s. $A=0$. That is, 
# \begin{equation*}
# \text{ATE}=\mathbb{E}[R(1)-R(0)].
# \end{equation*}
# 
# In this section, we will focus on the estimation of heterogeneous treatment effect (HTE), which is also one of the main focuses in causal inference.
# 
# 
# 
# ## Main Idea
# Let's first consider the single stage setup, where the observed data can be written as a state-action-reward triplet $\{S_i,A_i,R_i\}_{i=1}^n$ with a total of $n$ trajectories. Heterogeneous treatment effect, as we can imagine from its terminology, aims to measure the heterogeneity of the treatment effect for different subjects. Specifically, we define HTE as $\tau(s)$, where
# \begin{equation*}
# \tau(s)=\mathbb{E}[R(1)-R(0)|S=s],
# \end{equation*}
# 
# where $S=s$ denotes the state information of a subject. 
# 
# The estimation of HTE is widely used in a lot of real cases such as precision medicine, advertising, recommendation systems, etc. For example, in adversiting system, the company would like to know the impact (such as annual income) of exposing an ad to a group of customers. In this case, $S$ contains all of the information of a specific customer, $A$ denotes the status of ads exposure ($A=1$ means exposed and $A=0$ means not), and $R$ denotes the reward one can observe when assigned to policy $A$. 
# 
# Suppose the ad is a picture of a dress that can lead the customers to a detail page on a shopping website. In this case, females are more likely to be interested to click the picture and look at the detail page of a dress, resulting in a higher conversion rate than males. The difference of customers preference in clothes can be regarded as the heterogeneity of the treatment effect. By looking at the HTE for each customer, we can clearly estimate the reward of ads exposure from a granular level. 
# 
# Another related concept is conditional averge treatment effect, which is defined as
# \begin{equation*}
# \text{CATE}=\mathbb{E}[R(1)-R(0)|Z],
# \end{equation*}
# 
# where $Z$ is a collection of states with some specific characsteristics. For example, if the company is interested in the treatment effect of exposing the dress to female customers, $Z$ can be defined as ``female", and the problem can be addressed under the structure CATE estimation.
# 
# 
# 
# ## Different approaches in single-stage HTE estimation
# In the next few subsections, we will briefly summarize some state-of-the-art approaches in estimating the heterogeneous treatment effect. These methods contains
# 
# *   Meta learners [1]: S-learner, T-learner, X-learner.
# *   R-learner [2], DR-learner [4], Lp-R-learner [4].
# *   Other methods: Generalized random forest (GRF) [8], Dragonnet [9], etc.
# 
# Aside from the above methods that will be introduced in details in this section, there are several review papers which summarize some commonly-used approaches in literature, some of which are also detailed in the following subsections here. For more details please refer to [1, 10], etc.
# 
# 
# ## The advantages of different approaches
# 
# * **S-learner**: the easiest apporach to implement, general enough to incorporate any regression methods, from linear regression to very complicated neural networks. It is worth noting that S-learner regards the treatment variable $A$ the same as all other covariates $S$. This may cause S-learner to be less sensible if the regression method is not well chosen.
# 
# * **T-learner**: Different from S-learner, T-learner performs quite well when the underlying treatment effect is complicated, and there is no common trend in treatment group and control group that can be cancelled out when calculating HTE.
# 
# * **X-learner**: tends to perform particularly well when one of the treatment groups is much larger than the other, or when the separate parts of the X-learner can exploit the structural properties of the reward and treatment effect functions.
# 
# * **R-learner**: easy to implement,  can be adapted to any loss-minimization method such as penalized regression, deep neural networks, or boosting. Moreover, it acheives a quasi-oracle error bound under penalized kernel regression.
# 
# 
# * **DR-learner**: the idea is in line with the doubly robust procedure, which can be guaranteed to achieve the oracle efficiency under mild assumptions. The overall performance of DR-learner is always very impressive.
# 
# 
# * **Lp-R-learner**: an extension of R-learner, which has faster convergence rates than DR-learner in the non-oracle regime. However, it might be an computationally intensive to apply local polynomial regressions with a large degree. There is a tradeoff between theoretical properties and computational time.
# 
# 
# * **GRF**: a general approach to handle nonparametric estimation with random forests. It inherits the advantages of tree-based methods, and provides a very easy-to-understand, well-implemented procedure to estimate heterogeneous treatment effect on large datasets. In my experience, GRF is able to handle large, massive and complicated data structures and provides very reasonable results.
# 
# 
# * **Dragon Net**: able to adapt the design and training of
# the neural networks to improve the quality of the final estimate of the treatment effect. Existing studies show that Dragonnet has a great potential to outperform existing methods.
# 

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
# 2. Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319, 2021.
# 
# 3. Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society, pages 931–954, 1988.
# 
# 4. Edward H Kennedy. Optimal doubly robust estimation of heterogeneous causal effects. arXiv preprint arXiv:2004.14497, 2020
# 
# 5. M. J. van der Laan. Statistical inference for variable importance. The International Journal of Biostatistics, 2(1), 2006.
# 
# 6. S. Lee, R. Okui, and Y.-J. Whang. Doubly robust uniform confidence band for the conditional average treatment effect function. Journal of Applied Econometrics, 32(7):1207–1225, 2017.
# 
# 7. D. J. Foster and V. Syrgkanis. Orthogonal statistical learning. arXiv preprint arXiv:1901.09036, 2019.
# 
# 8. Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.
# 
# 9. Claudia Shi, David Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment effects. Advances in neural information processing systems, 32, 2019.
# 
# 10. Alicia Curth and Mihaela van der Schaar. Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. In International Conference on Artificial Intelligence and Statistics, pages
# 1810–1818. PMLR, 2021.
# 

# In[ ]:




