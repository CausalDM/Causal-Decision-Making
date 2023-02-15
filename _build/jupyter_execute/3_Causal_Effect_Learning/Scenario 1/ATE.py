#!/usr/bin/env python
# coding: utf-8

# # ATE Estimation 
# Average treatment effect (ATE), is a measure used to compare treatments (or interventions) in randomized experiments, evaluation of policy interventions, and medical trials. As we've introduced in the preliminary section, it aims to estimate the difference of some reward function in between treatment and control. Under the potential outcome's framework, or the notation of do-operator in RL-based literature, our main purpose lies in estimating and inferring on 
# \begin{equation}
# \text{ATE} = E[R^*(1) - R^*(0)] = E[ R|do(A=1)] -  E[ R|do(A=0)].
# \end{equation}
# 
# 
# ## Assumptions
# Under three common assumptions in causal inference, we can identify, or consistently estimate the potential outcome $\{R^*(0),R^*(1)\}$ from the observed data. These assumptions are 1) Consistency, 2) No unmeasured confounders (NUC), and 3) Positivity assumption.
# 
# - **SUTVA (Stable Unit Treatment Value Assumption, or Consistency Assumption)** 
# \begin{align}
# R_i = R_i^*(1) A_i + R_i^*(0) (1-A_i), i = 1, \cdots, n.
# \end{align}
# That is, the actual response that is observed for the $i$-th individual in our sample, $R_i$, who received treatment $A_i$, is the same as the potential outcome for that assigned treatment regardless of the experimental conditions used to assign treatment. This assumption also indicates that there is no interference between subjects in the population, that is, the observed response for any individual is not affected by the responses of other individuals.
# 
# - **NUC (No Unmeasured Confounders Assumption, or Strong Ignorability)**
# \begin{align}
# \{R^*(0), R^*(1)\} \perp\!\!\!\perp A|S
# \end{align}
# We cannot use the data in an observational study to either confirm or refute this assumption. However, if we believe that $X$ contained all the relevant information about the treatment assignment, then it will be reasonable to make the above assumption. 
# - **Positivity Assumption**
# \begin{align}
# 0 < P(A=1|S=s) < 1
# \end{align}
# This assumption ensures that for any given individual in treatment group $1$, we are able to find a similar individual in treatment group $0$, and vice versa.
# 
# We remark that these three assumptions are commonly imposed in causal inference and individualized treatment regimes literature [1]. Moreover, NUC and Positivity assumptions are automatically satisfied in randomized studies where the behavior policy is usually a strictly positive function independent of $s$. 
# 
# 
# ## Identification
# Based on the three assumptions above, we can re-write ATE as a function of the observed data. The details are shown below:
# \begin{equation}
# \begin{aligned}
# E[Y^*(1)] &= E_x [E\{Y^*(1)|X\}] \\
# &= E_x[E\{Y^*(1)|T=1, X\}] \quad \text{(NUC)}\\
# &= E_x[E\{Y|T=1, X\}] \quad\quad \text{(SUTVA)}
# \end{aligned}
# \end{equation}
# Similarly, we can show that $E[Y^*(0)] = E_x[E\{Y|T=0, X\}].$ Hence, the average causal treatment effect can be written as
# \begin{equation}
# \begin{aligned}
# \text{ATE} = E_x[E\{Y|T=1, X\}] - E_x[E\{Y|T=0, X\}],
# \end{aligned}
# \end{equation}
# where the RHS we get rid of the potential outcome's notations and thus can be identified from purely the observed data. 
# 
# 
# 
# 

# ## Estimation
# In this section, we will introduce three categories of estimators that have been widely used in ATE estimation: direct method (DM), importance sampling estimator (IS), and doubly robust estimator (DR).
# 
# 

# ### 1. Direct Method
# 
# The first approach to estimate the average treatment effect is through direct model fitting based on the identification result, which is also known as the outcome regression model. 
# 
# Specifically, we can posit a model for $E[R|A, S] = \mu(S, A;\gamma)$, and estimate the parameter $\gamma$ via least square or any other machine learning-based tools for model fitting. Then, The DM estimator of ATE is given by
# \begin{equation}
# \begin{aligned}
# \widehat{\text{ATE}}_{\text{DM}} = \frac{1}{n}\sum_{i=1}^n \{\mu(X_i, 1;\hat{\gamma}) - \mu(X_i, 0; \hat{\gamma})\}
# \end{aligned}
# \end{equation}

# In[1]:


# import data


# ### 2. Importance Sampling Estimator
# The second type of estimators is called importance sampling (IS) estimator, or inverse propensity score (IPW) and augmented inverse propensity score (AIPW) in causal inference literature.
# 
# Before we proceed, let's define the propensity score as below:
# \begin{equation}
# \begin{aligned}
# \pi(S) = P(A=1|S).
# \end{aligned}
# \end{equation}
# In the case where there are only two treatments, it refers to the propensity of getting one of the treatments as a function of the covariates. One of the attractive features of the propensity score is that given the `SUTVA` and `strong ignorability` assumptions, we have
# \begin{align}
# \{R^*(1), R^*(0)\} \perp\!\!\!\perp A|\pi(S)
# \end{align}
# In some cases when it is difficult to fit an outcome regression model for $E[R|A, S]$ and $S$ may be high dimensional, we can alternatively fit models for $E[R|A, \pi(S)]$ based on the result above.
# 
# The motivation of IPW is as follows. Since the propensity (probability) of receiving treatment $1$ is $\pi(S)$ for an individual with baseline covariates $S$, then every individual with covariates $X_i$ that was observed to receive treatment 1 in our sample is weighted by $1/\pi(S_i)$ so that their response not only represent themselves but also other individuals who did not receive treatment 1. More formally, we will show that $\frac{AR}{\pi(S)}$ is an unbiased estimator of $E[R^*(1)]$. This follows that, given the `positivity` assumption,
# \begin{equation} 
# \begin{aligned}
# E\left[\frac{AR}{\pi(S)}\right] = E\left[\frac{AR^*(1)}{\pi(S)}\right] = E\left[E\bigg\{\frac{AR^*(1)}{\pi(S)}\Big|S\bigg\}\right] = E\left[E\bigg\{\frac{A}{\pi(S)}\Big| S\bigg\}R^*(1)\right] = E[R^*(1)].
# \end{aligned}
# \end{equation}
# 
# Similarly, by flipping the role of treatment ($A=1$) and control ($A=0$), we have 
# \begin{equation}
# \begin{aligned}
# E[R^*(0)] = E\left[\frac{(1-A)R}{1-\pi(S)} \right].
# \end{aligned}
# \end{equation}
# Consequently, the IS (or IPW) estimator for the estimation of ATE is given by
# \begin{equation}
# \begin{aligned}
# \widehat{\text{ATE}}_{\text{IS}} =\frac{1}{n}\sum_{i=1}^n \left\{\frac{T_iY_i}{\pi(X_i)}  - \frac{(1-T_i)Y_i}{1-\pi(X_i)}  \right\}.
# \end{aligned}
# \end{equation}
# 

# In[2]:


# import data


# ### 3. Doubly Robust Estimator
# The third type of estimator is the doubly robust estimator. Basically, DR combines DM and IS estimators, which is more robust to model misspecifications. 
# \begin{equation}
# \begin{aligned}
# \widehat{\text{ATE}}_{\text{DR}} = \frac{1}{n}\sum_{i=1}^n \left\{\mu(S_i,1;\hat{\gamma})- \mu(S_i,0;\hat{\gamma})+\frac{A_i(R_i - \mu(S_i,1;\hat{\gamma}))}{\hat{\pi}(S_i)}  - \frac{(1-A_i)(R_i-\mu(S_i,0;\hat{\gamma}))}{1-\hat{\pi}(S_i)} \right\}.
# \end{aligned}
# \end{equation}
# To be more specific, the first two terms on the RHS forms a direct estimator and the last two terms serve as an augmentation to correct the bias rising from outcome regression models. When either the outcome regression models or the propensity scores are correctly specified, $\widehat{\text{ATE}}_{\text{DR}}$ can be proved to be consistent.
# 
# Under some mild entropy conditions or sample splitting, DR estimator is also a semi-parametrically efficient estimator when the convergence rate of both $\hat{\mu}$ and $\hat{\pi}$ are at least $o(n^{-1/4})$. Details can be found in Chernozhukov et al. 2018 [2].

# In[3]:


# import data


# ## References
# 1. Zhang, B., A. A. Tsiatis, E. B. Laber, and M. Davidian (2013). Robust estimation of optimal dynamic treatment regimes for sequential treatment decisions. Biometrika 100 (3), 681–694.
# 2. Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins (2018). Double/debiased machine learning for treatment and structural parameters.

# In[ ]:




