#!/usr/bin/env python
# coding: utf-8

# ## Causal Inference Preliminary
# Classical statistical inference considers associational relationships of different variables in a population of interest. However, in scientific research, we are often interested in causal relationships. Does action $A$ cause $R$. In this section, we will focus on establishing and estimating causal relationships between actions or interventions and subsequent response, under the potential outcome framework advocated by Neyman, Rubin, Robins and others.
# 
# We denote the treatment (intervention, action) by the variable $A$, which can be a vector of discrete, continuous random variables. For simplicity, we consider the simplest case where $A$ is a binary indicator, i.e. $A \in \{0, 1\}$ to denote two different interventions, say, for example, ad intervention versus no ad intervention for acquiring new users for new app. We will denote by $Y$ the response of interest, say, convert (download the new app) or not. The ad of the new app may be randomly assigned to a sample of users in a randomized controlled experiment, or the users may be intervened by the ad according to baseline features in an observational study. The data that are available can be summarized as $Z_i = (S_i, A_i, R_i), i=1, \cdots, n$, where for the $i-$ th sample, $S_i$ denotes the covariates that have been collected on the individual prior to the intervention, $A_i$ is the treatment received, $R_i$ denotes the response. We are interested in establishing a causal relationship between $A$ and $R$. 
# 
# ### Potential Outcome
# Under the potential outcome framework, we will assume, for each treatment, that there exists a potential outcome $R^*(t)$ which will represent the response of a randomly chosen individual in our population if they are given treatment $A=t$. Hence, when there are two treatments, $R^*(0)$ and $R^*(1)$ will represent the responses of a randomly chosen individual from the population if they are given treatment $A=0$ and $A=1$, respectively. $R^*(0)$ and $R^*(1)$ are sometimes referred to as counterfactual random variables because, in reality, they both cannot be observed and correspond to a hypothetical value if contrary to the fact the individual actually received a treatment different than the one considered. Using the notion of potential outcomes, we can define the causal treatment effect at the individual level by $\{R^*(1) - R^*(0)\}$. Because only one of $R^*(0)$ and $R^*(1)$ can be observed for any one individual, it is not possible to measure or estimate the subject-specific causal treatment effect.
# 
# ### Do-Operator
# The potential outcome can be equivalently described using the do-operator advocated by Pearl, Spirtes, and others. Specifically, we have $R^*(t) = R\{do(A=a)\}$ as the response of a randomly chosen individual from the population if they are given treatment $A=a$, where $do(A=a)$ is a mathematical operator to simulate physical interventions that hold $A$ constant as $a$ while keeping the rest of the model unchanged.
# 
# 
# ### Assumptions
# We now consider the necessary assumptions that allow us to deduce the distribution of $R^*(0)$ and $R^*(1)$ from the distribution of $(S, A, R)$.
# 
# - SUTVA (stable unit treatment value assumption) 
# \begin{align}
# R_i = R_i^*(1) A_i + R_i^*(0) (1-A_i), i = 1, \cdots, n.
# \end{align}
# That is, the actual response that is observed for the $i$-th individual in our sample, $R_i$, who received treatment $A_i$, is the same as the potential outcome for that assigned treatment regardless of the experimental conditions used to assign treatment. This assumption also indicates that there are not any interference between subjects in the population, that is, the observed response for any individual is not affected by the responses of other individuals 
# - No Unmeasured Confounders (Strong Ignorability) 
# \begin{align}
# \{R^*(0), R^*(1)\} \perp\!\!\!\perp A|S
# \end{align}
# We cannot use the data in an observational study to either confirm or refute this assumption. However, if we believe that $X$ contained all the relevant information about the treatment assignment, then it will be reasonable to make the above assumption. 
# - Positivity Assumption 
# \begin{align}
# 0 < P(A=1|S=s) < 1
# \end{align}
# This assumption ensures that for any given individual in treatment group $1$, we are able to find a similar individual in treatment group $0$, and vice versa.
# 
# ### Average Treatment Effect
# $\text{ATE} = E[R^*(1) - R^*(0)] = E[ R|do(A=1)] -  E[ R|do(A=0)]$
# 
# There are three general approaches to derive estimators for the average treatment effect from observational data: outcome regression modeling, the use of propensity score, and doubly roust procedure. We will introduce them in Causal Effect Learning (CEL) in details.
# 
# 
# ### Heterogeneous Treatment Effect
# $\text{HTE} = E[R^*(1) - R^*(0)|S=s] = E[ R|do(A=1),S=s] -  E[ R|do(A=0),S=s]$
# 
# Different from ATE, HTE aims to capture the heterogeniety of treatment effect caused by the difference of subjects' state information. There are quite a few approaches to deal with the estimation of HTE. We will introduce them in Causal Effect Learning (CEL) in details.
# 
# 
# 
