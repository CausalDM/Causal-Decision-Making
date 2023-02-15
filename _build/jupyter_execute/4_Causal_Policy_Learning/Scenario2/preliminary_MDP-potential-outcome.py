#!/usr/bin/env python
# coding: utf-8

# # Preliminary: Off-policy Evaluation and Optimization in Markov Decision Processes
# In this section, we introduce the formulation of the Markov Decision Process and a few related concepts which will be used repeatedly in this chapter. 
# We will proceed under the potential outcome framework, which provides a unique causal perspectiy, is different from the conventional notations [2], and is largely based on [3]. 
# Some of the assumptions to be discussed (such as the sequential randomization assumption) are imposed implicitly in the RL literature. 
# By writting these assumptions out, we aim to provide a more formal theoretical ground as well as to connect RL to the causal inference literautre. 

# ### Markov Decision Process under a Potential Outcome Framework 
# As the underlying data generation model for RL, we consider an infinite-horizon discounted Markov Decision Process (MDP) [1]. 
# For any $t\ge 0$, let $\bar{a}_t=(a_0,a_1,\cdots,a_t)^\top\in \mathcal{A}^{t+1}$ denote a treatment history vector up to time $t$. Let $\mathbb{S} \subset \mathbb{R}^d$ denote the support of state variables and $S_0$ denote the initial state variable. 
# For any $(\bar{a}_{t-1},\bar{a}_{t})$, let $S_{t}^*(\bar{a}_{t-1})$ and $Y_t^*(\bar{a}_t)$ be the counterfactual state and counterfactual outcome, respectively,  that would occur at time $t$ had the agent followed the treatment history $\bar{a}_{t}$. 
# The set of potential outcomes up to time $t$ is given by
# \begin{eqnarray*}
# 	W_t^*(\bar{a}_t)=\{S_0,Y_0^*(a_0),S_1^*(a_0),\cdots,S_{t}^*(\bar{a}_{t-1}),Y_t^*(\bar{a}_t)\}.
# \end{eqnarray*}
# Let $W^*=\cup_{t\ge 0,\bar{a}_t\in \{0,1\}^{t+1}} W_t^*(\bar{a}_t)$ be the set of all potential outcomes.
# 
# A deterministic policy $\pi$ is a time-homogeneous function that maps the space of state variables to the set of available actions. 
# Following $\pi$, the agent will assign actions according to $\pi$ at each time.  We use $S_t^*(\pi)$ and $Y_t^*(\pi)$ to denote the associated potential state and outcome that would occur at time $t$ had the agent followed $\pi$. 
# 
# The goodness of  a policy $\pi$ is measured by its (state) value function, 
# \begin{eqnarray*}
# 	V^{\pi}(s)=\sum_{t\ge 0} \gamma^t \mathbb{E} \{Y_t^*(\pi)|S_0=s\},
# \end{eqnarray*}
# where $0<\gamma<1$ is a discount factor that reflects the trade-off between immediate and future outcomes. The value function measures the discounted cumulative outcome that the agent would receive had they followed $\pi$. Note that our definition of the value function is slightly different from those in the existing literature [2]. Specifically, $V(\pi;s)$ is defined through potential outcomes rather than the observed data. 
# 
# Similarly, we define the Q function by
# \begin{eqnarray*}
# 	Q^{\pi}(a,s)=\sum_{t\ge 0} \gamma^t \mathbb{E} \{Y_t^*(\pi)|S_0=s, A_0 = a\}. 
# \end{eqnarray*}
# 
# The optimal policy is defined as $\pi^* = \arg \max_{\pi} V^\pi(s), \forall s \in \mathcal{S}$. 
# 
# 
# The following two assumptions are central to and also unique in the reinforcement learning setting.
# 
# **(MA) Markov assumption**:  there exists a Markov transition kernel $\mathcal{P}$ such that  for any $t\ge 0$, $\bar{a}_{t}\in \{0,1\}^{t+1}$ and $\mathcal{S}\subseteq \mathbb{R}^d$, we have 
# $\mathbb{P}\{S_{t+1}^*(\bar{a}_{t})\in \mathcal{S}|W_t^*(\bar{a}_t)\}=\mathcal{P}(\mathcal{S};a_t,S_t^*(\bar{a}_{t-1})).$
# 
# **(CMIA) Conditional mean independence assumption**: there exists a function $r$ such that  for any $t\ge 0, \bar{a}_{t}\in \{0,1\}^{t+1}$, we have 
# $\mathbb{E} \{Y_t^*(\bar{a}_t)|S_t^*(\bar{a}_{t-1}),W_{t-1}^*(\bar{a}_{t-1})\}=r(a_t,S_t^*(\bar{a}_{t-1}))$.
# 
# They assume (i) the process is statioanry, and (ii) the state variables shall be chosen to include those that serve as important mediators between past treatments and current outcomes. 
# These two conditions are central to the empirical validity of most RL algorithms. 
# Specifically, under these two conditions, one can show that there exists an optimal time-homogenous stationary policy whose value is no worse than any history-dependent policy [1]. 

# ## Off-policy Evaluation and Optimization
# 
# In the off-policy setting, the observed data consists of $n$ i.i.d. trajectories $\{(S_{i,t},A_{i,t},R_{i,t},S_{i,t+1})\}_{0\le t<T_i,1\le i\le n}$, where $T_i$ denotes the length of the $i$th trajectory. Without loss of generality, we assume $T_1=\cdots=T_n=T$ and the immediate rewards are uniformly bounded. 
# The dataset is collected by following a stationary policy $b$, known as the *behavior policy*. 
# 
# **Off-Policy Evaluation(OPE).** The goal of OPE is to estimate the value of a given *target policy* $\pi$ with respect to the initial state distribution $\mathbb{G}$, defined as 
# \begin{eqnarray}\label{eqn:def_value}
# 	\eta^{\pi} =  \mathbb{E}_{s \sim \mathbb{G}} V^{\pi}(s). 
# \end{eqnarray} 
# By definition, we directly have $\eta^{\pi} = \mathbb{E}_{s \sim \mathbb{G}, a \sim \pi(\cdot|s)} Q^{\pi}(a, s)$. 
# 
# In addition to a point estimator, many applications would benefit from having a CI for $\eta^{\pi}$. 
# We refer to an interval $[\hat{\eta}^{\pi}_l, \hat{\eta}^{\pi}_u]$ as an $(1-\alpha)$-CI for $\eta^{\pi}$ if and only if $P(\hat{\eta}^{\pi}_l \le \eta^{\pi} \le \hat{\eta}^{\pi}_u) \ge 1 - \alpha$, for any $\alpha \in (0, 1)$.  
# 
# **Off-Policy Optimization(OPO).** The goal of OPO is to solve the optimal policy $\pi^*$, or in other words, to learn a policy $\hat{\pi}$ so as to minimize the regret $\eta^{\pi^*} - \eta^{\hat{\pi}}$. 
# 
# 
# ## Causal Identifiability 
# In general, the set $W^*$ cannot be observed, whereas at time $t$, we observe the state-action-outcome triplet $(S_t,A_t,Y_t)$. 
# For any $t\ge 0$, let $\bar{A}_t=(A_0,A_1,\cdots,A_t)^\top$ denote the observed treatment history. 
# Similar to our discussions in previous chapters, 
# the off-policy evaluation/optimization tasks requires certain assumptions to ensure the causal identifiability. 
# The two critical assumptions are: 
# 
# **(CA) Consistency assumption**: $S_{t+1}=S_{t+1}^*(\bar{A}_{t})$ and $Y_t=Y_t^*(\bar{A}_t)$ for all $t\ge 0$, almost surely.
# 
# **(SRA) Sequential randomization assumption**: $A_t\perp W^*| S_{t}, \{S_j,A_j,Y_j\}_{0\le j<t}$.
# 
# The CA requires that the observed state and outcome correspond to the potential state and outcome whose treatments are assigned according to the observed treatment history. 
# It generalizes SUTVA to our setting, allowing the potential outcomes to depend on past treatments. 
# The SRA implies that  there are no unmeasured confounders and it automatically holds in online randomized experiments (or when all trajectories are collected by following policies that depend on the same set of state variables), in which the treatment assignment mechanism is pre-specified. 
# In SRA, we allow $A_t$ to depend on the observed data history $S_{t}, \{S_j,A_j,Y_j\}_{0\le j<t}$ and thus, the treatments can be adaptively chosen.  
# 
# 
# In addition, these two conditions guarantee that MA and CMIA hold on the observed dataset as well.
# \begin{eqnarray}\label{eqn:Markovobserve}
# 	P(S_{t+1}\in \mathcal{S}|A_t,S_t,\{S_j,A_j,Y_j\}_{0\le j<t})&=&\mathcal{P}(\mathcal{S};A_t,S_t),\\\label{eqn:robserve}
# 	\mathbb{E}(Y_t|A_t,S_t,\{S_j,A_j,Y_j\}_{0\le j<t})&=&r(A_t,S_t).
# \end{eqnarray}
# As such, $\mathcal{P}$ corresponds to the transition function that defines the next state distribution conditional on the current state-action pair and $r$ corresponds to the conditional expectation of the immediate reward as a function of the state-action pair. 
# In this chapter, we may use both the potential outcomes and the observed variables interchangeably. 

# ## Reference
# 
# [1] Puterman M L. Markov decision processes: discrete stochastic dynamic programming[M]. John Wiley & Sons, 2014.
# 
# [2] Sutton R S, Barto A G. Reinforcement learning: An introduction[M]. MIT press, 2018.
# 
# [3] Shi C, Wang X, Luo S, et al. A Reinforcement Learning Framework for Time-Dependent Causal Effects Evaluation in A/B Testing[J]. arXiv preprint arXiv:2002.01711, 2020.
