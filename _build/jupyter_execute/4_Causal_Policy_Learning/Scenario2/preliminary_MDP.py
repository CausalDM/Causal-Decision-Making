#!/usr/bin/env python
# coding: utf-8

# # Preliminary: Off-policy Evaluation and Optimization in Markov Decision Processes
# 
# 
# ## Markov Decision Process
# As the underlying data generation model for RL, 
# we consider an infinite-horizon discounted Markov Decision Process MDP [1] defined by a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. Here, $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, 
# $\mathcal{P}$ is the transition kernel with $\mathcal{P}(s'| s, a)$ giving the mass function (or probability density) of entering state $s'$ by taking action $a$ in state $s$, 
# $\mathcal{R}$ is the reward kernel with $\mathcal{R}(r|s, a)$ denoting the mass function (or probability density) of receiving a reward $r$ after taking action $a$ in state $s$, 
# and $\gamma \in (0, 1)$ is a discounted factor that balances the immediate and future rewards. 
# To simplify the presentation, we assume the spaces are discrete throughout this report. 
# Meanwhile, most discussions continue to hold with continuous spaces as well. 
# Following a given policy $\pi$, when the current state is $s$, 
# the agent will select action $a$ with probability $\pi(a|s)$. 
# 
# 
# Let $\{(S_t,A_t,R_t)\}_{t\ge 0}$ denote a trajectory generated from the MDP model where $(S_t,A_t,R_t)$ denotes the state-action-reward triplet at time $t$. 
# A trajectory following policy $\pi$ is generated as follows. 
# The agent starts from a state $S_0$ sampled from the initial state distribution,  denoted by $\mathbb{G}$. 
# At each time point $t \ge 0$, the agent will select an action $A_{t}$ following $\pi(\cdot|S_{t})$, and then receive a random reward $R_{t}$ following  $\mathcal{R}(\cdot|S_{t}, A_{t})$, and finally moves to the next state  $S_{t+1}$ following $\mathcal{P}(\cdot; S_{t}, A_{t})$. 

# ## Value functions
# 
# 
# 
# The state value function (referred to as the V-function) and state-action value function (referred to as the Q-function) for a policy $\pi$ are given as follows:
# \begin{align}
# 	V^\pi(s)&=\mathbb{E}^{\pi} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|S_{0}=s),\\
# 	Q^\pi(a,s)&= \mathbb{E}^{\pi} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|A_{0}=a,S_{0}=s) \label{eqn:Q},
# \end{align}
# where the expectation $\mathbb{E}^{\pi}$ is defined by assuming the system follows the policy $\pi$. 
# The observed discounted cumulative reward for a trajectory $\{(S_t,A_t,R_t)\}_{t\ge 0}$ is $\sum_{t=0}^{\infty} \gamma^t R_t$. 
# The optimal policy is defined as $\pi^* = \arg \max_{\pi} V^\pi(s), \forall s \in \mathcal{S}$. 
# 
# Finally, we are ready to introduce the well-known Bellman equation [2]. Note that the definition of the Q function implies that 
# \begin{align*}
#     Q^\pi(a,s)
#     = \mathbb{E}^{\pi} (R_{0} + \sum_{t=1}^{+\infty} \gamma^t R_{t}|A_{0}=a,S_{0}=s)
#     &= \mathbb{E}^{\pi} (R_{0} + \gamma \sum_{t=0}^{+\infty} \gamma^t R_{t+1}|A_{0}=a,S_{0}=s)\\
#     &=  \mathbb{E}^\pi \Big(R_t + \gamma Q^{\pi}(A_{t + 1}, S_{t+1})  | A_t = a, S_t = s \Big), 
# \end{align*}
# where the last equality follows from the stationarity of the MDP. 
# Motivated by this fact, 
# the Bellman equation for the Q-function is defined as
# \begin{equation}\label{eqn:bellman_Q}
#     Q(a, s) = \mathbb{E}^\pi \Big(R_t + \gamma Q(A_{t + 1}, S_{t+1})  | A_t = a, S_t = s \Big), 
# \end{equation}
# for which $Q^\pi$ is the unique solution [2]. 
# The Bellman equation relates different components of the MDP via a recursive form and is the foundation for lots of RL algorithms. 
# 
# Similarly, we have the Bellman optimality equation, which characterizes the optimal policy $\pi^*$ and is commonly used in policy optimization. 
# Specifically, its value function $Q^*$ is the unique solution of 
# \begin{equation}
#     Q(a, s) = \mathbb{E} \Big(R_t + \gamma \arg \max_{a'} Q(a, S_{t+1})  | A_t = a, S_t = s \Big).  \;\;\;\;\; \text{(2)} 
# \end{equation}
# 
# 
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

# ## Reference
# 
# [1] Puterman M L. Markov decision processes: discrete stochastic dynamic programming[M]. John Wiley & Sons, 2014.
# 
# [2] Sutton R S, Barto A G. Reinforcement learning: An introduction[M]. MIT press, 2018.

# 
