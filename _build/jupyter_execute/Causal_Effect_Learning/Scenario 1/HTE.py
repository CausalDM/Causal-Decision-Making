#!/usr/bin/env python
# coding: utf-8

# # Heterogeneous Treatment Effect Estimation (Single Stage)
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
# *   Other methods: Generalized random forest [8], Dragonnet, etc.
# 
# Aside from the above methods that will be introduced in details in this section, there are several review papers which summarize some commonly-used approaches in literature, some of which are also detailed in the following subsections here. For more details please refer to [1], etc.
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
