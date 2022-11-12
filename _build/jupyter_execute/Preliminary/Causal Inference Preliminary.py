#!/usr/bin/env python
# coding: utf-8

# ## Average Treatment Effect
# Classical statistical inference considers associational relationships of different variables in a population of interest. However, in scientific research, we are often interested in causal relationships. Does action $T$ cause $Y$. In this section, we will focus on establishing and estimating causal relationships between actions or interventions and subsequent response, under the potential outcome framework advocated by Neyman, Rubin, Robins and others.
# 
# We denote the treatment (intervention, action) by the variable $T$, which can be a vector of discrete, continuous random variables. For simplicity, we consider the simplest case where $T$ is a binary indicator, i.e. $T \in \{0, 1\}$ to denote two different interventions, say, for example, ad intervention versus no ad intervention for acquiring new users for new app. We will denote by $Y$ the response of interest, say, convert (download the new app) or not. The ad of the new app may be randomly assigned to a sample of users in a randomized controlled experiment, or the users may be intervened by the ad according to baseline features in an observational study. The data that are available can be summarized as $Z_i = (X_i, T_i, Y_i), i=1, \cdots, n$, where for the $i-$ th sample, $X_i$ denotes the covariates that have been collected on the individual prior to the intervention, $T_i$ is the treatment received, $Y_i$ denotes the response. We are interested in establishing a causal relationship between $T$ and $Y$. 
# 
# ### Potential Outcome
# Under the potential outcome framework, we will assume, for each treatment, that there exists a potential outcome $Y^*(t)$ which will represent the response of a randomly chosen individual in our population if they are given treatment $T=t$. Hence, when there are two treatments, $Y^*(0)$ and $Y^*(1)$ will represent the responses of a randomly chosen individual from the population if they are given treatment $T=0$ and $T=1$, respectively. $Y^*(0)$ and $Y^*(1)$ are sometimes referred to as counterfactual random variables because, in reality, they both cannot be observed and correspond to a hypothetical value if contrary to the fact the individual actually received a treatment different than the one considered. Using the notion of potential outcomes, we can define the causal treatment effect at the individual level by $\{Y^*(1) - Y^*(0)\}$. Because only one of $Y^*(0)$ and $Y^*(1)$ can be observed for any one individual, it is not possible to measure or estimate the subject-specific causal treatment effect.
# 
# ### Do-Operator
# The potential outcome can be equivelently described using the do-operator advocated by Pearl, Spirtes, and others. Specifically, we have $Y^*(t) = Y\{do(T=t)\}$ as the response of a randomly chosen individual from the population if they are given treatment $T=t$, where $do(T=t)$ is a mathematical operator to simulate physical interventions that hold $T$ constant as $t$ while keeping the rest of the model unchanged.
# 
# 
# ### Assumptions
# We now consider the necessary assumptions that allow us to deduce the distribution of $Y^*(0)$ and $Y^*(1)$ from the distribution of $(X, T, Y)$.
# 
# - SUTVA (stable unit treatment value assumption) 
# \begin{align}
# Y_i = Y_i^*(1) T_i + Y_i^*(0) (1-T_i), i = 1, \cdots, n.
# \end{align}
# That is, the actual response that is observed for the $i$-th individual in our sample, $Y_i$, who received treatment $T_i$, is the same as the potential outcome for that assigned treatment regardless of the experimental conditions used to assign treatment. This assumption also indicates that there are not any interference between subjects in the population, that is, the observed response for any individual is not affected by the responses of other individuals 
# - No Unmeasured Confounders (Strong Ignorability) 
# \begin{align}
# \{Y^*(0), Y^*(1)\} \perp\!\!\!\perp T|X
# \end{align}
# We cannot use the data in an observational study to either confirm or refute this assumption. However, if we believe that $X$ contained all the relevant information about the treatment assignment, then it will be reasonable to make the above assumption. 
# - Positivity Assumption 
# \begin{align}
# 0 < P(T=1|X=x) < 1
# \end{align}
# This assumption ensures that for any given individual in treatment group $1$, we are able to find a similar individual in treatment group $0$, and vice versa.
# 
# ### Average Causal Effect
# $\text{ATE} = E[Y^*(1) - Y^*(0)] = E[ Y|do(T=1)] -  E[ Y|do(T=0)]$
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
# \begin{align}
# \text{ATE} = E_x[E\{Y|T=1, X\}] - E_x[E\{Y|T=0, X\}]
# \end{align}
# 
# We can posit a model for $E[Y|T, X] = \mu(X, T;\gamma)$, and estimate the parameter $\gamma$ via MLE or least square estimation. Then,
# \begin{align}
# \hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \{\mu(X_i, 1;\hat{\gamma}) - \mu(X_i, 0; \hat{\gamma})\}
# \end{align}
# 
# #### Propensity Score
# Another class of estimators for the average causal treatment effect uses propensity score, defined as 
# \begin{align}
# \pi(X) = P(T=1|X)
# \end{align}
# in the case where there are two treatments. It refers to the propensity of getting one of the treatments as a function of the covariates. One of the attractive features of the propensity score, that makes it useful for causal inference follows from the fact that given the `SUTVA` and `strong ignorability` assumptions, then
# \begin{align}
# \{Y^*(1), Y^*(0)\} \perp\!\!\!\perp T|\pi(X)
# \end{align}
# It may be difficult to build an outcome regression model for $E[Y|T, X]$ if $X$ is high dimensional, however, based on the above result, we can simply build models for $E[Y|T, \pi(X)]$.
# 
# ##### Stratification
# Rubin suggested the following to estimate ATE. Stratify the observations into groups based on the value of the estimated propensity scores $\hat{\pi}(X_i), i=1, \cdots, n$. That is, choose cutoff values,
# \begin{align}
# 0 = C_0 < C_1 < \cdots < C_K = 1,
# \end{align}
# then individual $i$ belongs to group $k$ if 
# \begin{align}
# C_{k-1} < \hat{\pi}(X_i) \le C_k, k=1, \cdots, K,
# \end{align}
# then
# \begin{align}
# \text{ATE} = \sum_{k=1}^K (\bar{Y}_{1k} - \bar{Y}_{0k}) n_k /n
# \end{align}
# where $\hat{Y}_{1k}, \hat{Y}_{0k}$ are the sample average responses among individuals in the $k-$th group receiving treatments $1$ and $0$ respectively, and $n_k$ is the number of individuals from both treatments in the $k-$th group.
# 
# ##### Inverse Propensity Score Weighting
# Another class of estimators with a more theoretical justification coming from semiparametric theory uses the inverse propensity score estimator and augmented inverse propensity score estimator for the average causal treatment effect.
# 
# The motivation of IPW is as follows. Since the propensity (probability) of receiving treatment $1$ is $\pi(X)$ for an individual with baseline covariates $X$, then every individual with covariates $X_i$ that was observed to receive treatment 1 in our sample is weighted by $1/\pi(X_i)$ so that their response not only represent themselves but also other individuals who did not receive treatment 1. More formally, we will show that $\frac{TY}{\pi(X)}$ is an unbiased estimator of $E[Y^*(1)]$. This follows that, given the `positivity` assumption, 
# \begin{align}
# E\left[\frac{TY}{\pi(X)}\right] = E\left[\frac{TY^*(1)}{\pi(X)}\right] = E\left[E\left\{\frac{TY^*(1)}{\pi(X)}|X\right\}\right] = E\left[E\left\{\frac{T}{\pi(X)}|X\right\}Y^*(1)\right] = E[Y^*(1)].
# \end{align}
# 
# Similarly, we have 
# \begin{align}
# E[Y^*(0)] = E\left[\frac{(1-T)Y}{1-\pi(X)} \right],
# \end{align}
# Consequently, then another unbiased estimator for the average causal treatment effect would be 
# \begin{align}
# \hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \left\{\frac{T_iY_i}{\pi(X_i)}  - \frac{(1-T_i)Y_i}{1-\pi(X_i)}  \right\}
# \end{align}
# 
# #### Doubly Robust Estimators
# We are able to combine outcome regression estimator and the propensity score weighted estimator to obtain another so-called doubly robust estimator, as follows
# \begin{align}
# \hat{\text{ATE}} = n^{-1}\sum_{i=1}^n \left\{\mu(X_i,1;\hat{\gamma})+\frac{T_i(Y_i - \mu(X_i,1;\hat{\gamma}))}{\hat{\pi}(X_i)} - \mu(X_i,0;\hat{\gamma}) - \frac{(1-T_i)(Y_i-\mu(X_i,0;\hat{\gamma}))}{1-\hat{\pi}(X_i)} \right\}
# \end{align}

# In[ ]:




