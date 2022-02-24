#!/usr/bin/env python
# coding: utf-8

# ## Motivating Examples
# 
# ### Personalized Incentives
# User growth and engagement are critical in a fast-changing market. Marketing campaigns in internet companies offer quantifiable incentives to encourage users to engage or use new products. The treatment has positive effects on desired business growth, also lead to a surplus in operation cost. The increasing cost impels internet companies to carry out more refined strategies in user acquisition, user retention, and etc. Specifically, the associated costs of massive-scale promotion campaigns must be balanced by incremental business value, with a sustainable return-on-investment (ROI). We are required to predict, for each user, the change in business value caused by different incentive actions, in order to maximize the ROI of market campaigns. This problem is known as uplift modeling, or heterogeneous treatment effect estimation, which has received more and more attention in the causal inference literature.
# 
# ### Ad Targeting & Bidding
# John Wanamaker once phrased 'Half the money I spend on advertising is wasted, the trouble is I don't know which half.'
# It indicates that we are wasting our advertisement on users with high intention to convert naturally. Today's digital technique enables us to estimate the conversion lift of each user via randomized controlled studies. We randomly select users to form two groups, one can be intervened via our ads and the other cannot. Based on the collected data, we can estimate the difference of conversion between ad intervention and no ad intervention. We can then target users with high converion lift, or even increase our bidding price to win the impression for those users.
# 
# 
# ### Multi-touch Attribution
# Users may interact with the same advertisement (possibly with different styles) many times through different channels. 
# Multi-touch attribution, which allows distributing the credit to all related advertisements based on their corresponding contributions, has recently become an important research topic in digital advertising. Rules based on simple intuition have been used in practice for a long time. With the ever enhanced capability to tracking advertisement and users’ interaction with the advertisement, data-driven multi-touch attribution models, which attempt to infer the contribution from user interaction data, become increasingly more popular recently. This problem can be easily formalized to the multi-stage dynamic treatment regime framework. 
# 
# 
# ## Causal Inference 101
# Classical statistical inference considers associational relationships of different variables in a population of interest. However, in scientific research, we are often interested in causal relationships. Does action $T$ cause $Y$. In this section, we will focus on establishing and estimating causal relationships between actions or interventions and subsequent response, under the potential outcome framework advocated by Neyman, Rubin, Robins and others.
# 
# We denote the treatment (intervention, action) by the variable $T$, which can be a vector of discrete, continuous random variables. For simplicity, we consider the simplest case where $T$ is a binary indicator, i.e. $T \in \{0, 1\}$ to denote two different interventions, say, for example, ad intervention versus no ad intervention for acquiring new users for new app. We will denote by $Y$ the response of interest, say, convert (download the new app) or not. The ad of the new app may be randomly assigned to a sample of users in a randomized controlled experiment, or the users may be intervened by the ad according to baseline features in an observational study. The data that are available can be summarized as $Z_i = (X_i, T_i, Y_i), i=1, \cdots, n$, where for the $i-$ th sample, $X_i$ denotes the covariates that have been collected on the individual prior to the intervention, $T_i$ is the treatment received, $Y_i$ denotes the response. We are interested in establishing a causal relationship between $T$ and $Y$. 
# 
# ### Potential Outcome
# Under the potential outcome framework, we will assume, for each treatment, that there exists a potential outcome $Y^*(t)$ which will represent the response of a randomly chosen individual in our population if they are given treatment $T=t$. Hence, when there are two treatments, $Y^*(0)$ and $Y^*(1)$ will represent the responses of a randomly chosen individual from the population if they are given treatment $T=0$ and $T=1$, respectively. $Y^*(0)$ and $Y^*(1)$ are sometimes referred to as counterfactual random variables because, in reality, they both cannot be observed and correspond to a hypothetical value if contrary to the fact the individual actually received a treatment different than the one considered. Using the notion of potential outcomes, we can define the causal treatment effect at the individual level by $\{Y^*(1) - Y^*(0)\}$. Because only one of $Y^*(0)$ and $Y^*(1)$ can be observed for any one individual, it is not possible to measure or estimate the subject-specific causal treatment effect.
# 
# ### Assumptions
# We now consider the necessary assumptions that allow us to deduce the distribution of $Y*(0)$ and $Y^*(1)$ from the distribution of $(X, T, Y)$.
# 
# - SUTVA (stable unit treatment value assumption) 
# $$Y_i = Y_i^*(1) T_i + Y_i^*(0) (1-T_i), i = 1, \cdots, n.$$
# That is, the actual response that is observed for the $i$-th individual in our sample, $Y_i$, who received treatment $T_i$, is the same as the potential outcome for that assigned treatment regardless of the experimental conditions used to assign treatment. This assumption also indicates that there are not any interference between subjects in the population, that is, the observed response for any individual is not affected by the responses of other individuals 
# - No Unmeasured Confounders (Strong Ignorability) 
# $$\{Y^*(0), Y^*(1)\} \perp\!\!\!\perp T|X$$
# We cannot use the data in an observational study to either confirm or refute this assumption. However, if we believe that $X$ contained all the relevant information about the treatment assignment, then it will be reasonable to make the above assumption. 
# - Positivity Assumption 
# $$0 < P(T=1|X=x) < 1$$
# This assumption ensures that for any given individual in treatment group $1$, we are able to find a similar individual in treatment group $0$, and vice versa.
# 
# ### Average Causal Effect
# $\text{ATE} = E[Y^*(1) - Y^*(0)]$
# There are two general approaches to deriving estimators for the average causal treatment effect from observational data: outcome regression modeling and the use of propensity score.
# 
# #### Outcome Regression Models
# The use of outcome regression modeling to estimate the average causal treatment effect from observational data under the assumption of no unmeasured confouners and SUTVA comes from the following consideration:
# \begin{align}
# E[Y^*(1)] &= E_x [E\{Y^*(1)|X\}] \\
# &= E_x[E\{Y^*(1)|T=1, X\}] \quad \text{No Unmeasured Confounders}\\
# &= E_x[E\{Y|T=1, X\}] \quad \text{SUTVA}
# \end{align}
# Similarly, we can show that $E[Y^*(0)] = E_x[E\{Y|T=0, X\}].$
# Hence, the average causal treatment effect 
# $$\text{ATE} = E_x[E\{Y|T=1, X\}] - E_x[E\{Y|T=0, X\}]$$
# 
# We can posit a model for $E[Y|T, X] = \mu(X, T;\gamma)$, and estimate the parameter $\gamma$ via MLE or least square estimation. Then,
# $$\hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \{\mu(X_i, 1;\hat{\gamma}) - \mu(X_i, 0; \hat{\gamma})\}$$
# 
# #### Propensity Score
# Another class of estimators for the average causal treatment effect uses propensity score, defined as 
# $$\pi(X) = P(T=1|X)$$
# in the case where there are two treatments. It refers to the propensity of getting one of the treatments as a function of the covariates. One of the attractive features of the propensity score, that makes it useful for causal inference follows from the fact that given the `SUTVA` and `strong ignorability` assumptions, then
# $$\{Y^*(1), Y^*(0)\} \perp\!\!\!\perp T|\pi(X)$$
# It may be difficult to build an outcome regression model for $E[Y|T, X]$ if $X$ is high dimensional, however, based on the above result, we can simply build models for $E[Y|T, \pi(X)]$.
# 
# ##### Stratification
# Rubin suggested the following to estimate ATE. Stratify the observations into groups based on the value of the estimated propensity scores $\hat{\pi}(X_i), i=1, \cdots, n$. That is, choose cutoff values,
# $$0 = C_0 < C_1 < \cdots < C_K = 1,$$
# then individual $i$ belongs to group $k$ if 
# $$C_{k-1} < \hat{\pi}(X_i) \le C_k, k=1, \cdots, K,$$
# then
# $$\text{ATE} = \sum_{k=1}^K (\bar{Y}_{1k} - \bar{Y}_{0k}) n_k /n$$
# where $\hat{Y}_{1k}, \hat{Y}_{0k}$ are the sample average responses among individuals in the $k-$th group receiving treatments $1$ and $0$ respectively, and $n_k$ is the number of individuals from both treatments in the $k-$th group.
# 
# ##### Inverse Propensity Score Weighting
# Another class of estimators with a more theoretical justification coming from semiparametric theory uses the inverse propensity score estimator and augmented inverse propensity score estimator for the average causal treatment effect.
# 
# The motivation of IPW is as follows. Since the propensity (probability) of receiving treatment $1$ is $\pi(X)$ for an individual with baseline covariates $X$, then every individual with covariates $X_i$ that was observed to receive treatment 1 in our sample is weighted by $1/\pi(X_i)$ so that their response not only represent themselves but also other individuals who did not receive treatment 1. More formally, we will show that $\frac{TY}{\pi(X)}$ is an unbiased estimator of $E[Y^*(1)]$. This follows that, given the `positivity` assumption, 
# $$E\left[\frac{TY}{\pi(X)}\right] = E\left[\frac{TY^*(1)}{\pi(X)}\right] = E\left[E\left\{\frac{TY^*(1)}{\pi(X)}|X\right\}\right] = E\left[E\left\{\frac{T}{\pi(X)}|X\right\}Y^*(1)\right] = E[Y^*(1)].$$
# 
# Similarly, we have 
# $$E[Y^*(0)] = E\left[\frac{(1-T)Y}{1-\pi(X)} \right],$$
# Consequently, then another unbiased estimator for the average causal treatment effect would be 
# $$\hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \left\{\frac{T_iY_i}{\pi(X_i)}  - \frac{(1-T_i)Y_i}{1-\pi(X_i)}  \right\}$$
# 
# #### Doubly Robust Estimators
# We are able to combine outcome regression estimator and the propensity score weighted estimator to obtain another so-called doubly robust estimator, as follows
# $$\hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \left\{\mu(X_i,1;\hat{\gamma})+\frac{T_i(Y_i - \mu(X_i,1;\hat{\gamma}))}{\hat{\pi}(X_i)} - \mu(X_i,0;\hat{\gamma}) - \frac{(1-T_i)(Y_i-\mu(X_i,0;\hat{\gamma}))}{1-\hat{\pi}(X_i)} \right\}$$
