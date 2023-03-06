#!/usr/bin/env python
# coding: utf-8

# (section:online_RL)=
# # Ooline Policy Learning and Evaluation in Markovian Environments 
# 
# This chapter focuses on the online policy learning and evaluation problem in an Markov Decision Process (MDP), which is the most well-known and typically default setup of Reinforcement Learning (RL). 
# From the causal perspective, the data dependency structure is the same with that in [offline RL](section:OPE_OPO_preliminary), with the major difference in that the data collection policy is now data-dependent and the objective is sometimes shifted from finding the optimal policy to maximizing the cumulative rewards. 
# As this is a vast area with a huge literature, we do not aim to repeat the disucssions hear. Instead, we will focus on connecting online RL to the other parts of this paper. 
# We refer interested readers to {cite:t}`sutton2018reinforcement` for more materials.
# 
# ## Model
# 
# We first recap the infinite-horizon discounted MDP model that we introduced in [offline RL](section:OPE_OPO_preliminary). 
# For any $t\ge 0$, let $\bar{a}_t=(a_0,a_1,\cdots,a_t)^\top\in \mathcal{A}^{t+1}$ denote a treatment history vector up to time $t$. 
# Let $\mathbb{S} \subset \mathbb{R}^d$ denote the support of state variables and $S_0$ denote the initial state variable. 
# For any $(\bar{a}_{t-1},\bar{a}_{t})$, let $S_{t}^*(\bar{a}_{t-1})$ and $Y_t^*(\bar{a}_t)$ be the counterfactual state and counterfactual outcome, respectively,  that would occur at time $t$ had the agent followed the treatment history $\bar{a}_{t}$. 
# The set of potential outcomes up to time $t$ is given by
# \begin{eqnarray*}
# 	W_t^*(\bar{a}_t)=\{S_0,Y_0^*(a_0),S_1^*(a_0),\cdots,S_{t}^*(\bar{a}_{t-1}),Y_t^*(\bar{a}_t)\}.
# \end{eqnarray*}
# Let $W^*=\cup_{t\ge 0,\bar{a}_t\in \{0,1\}^{t+1}} W_t^*(\bar{a}_t)$ be the set of all potential outcomes.
# 
# The goodness of  a policy $\pi$ is measured by its value functions, 
# \begin{eqnarray*}
#     V^{\pi}(s)=\sum_{t\ge 0} \gamma^t \mathbb{E} \{Y_t^*(\pi)|S_0=s\}, \;\; 	Q^{\pi}(a,s)=\sum_{t\ge 0} \gamma^t \mathbb{E} \{Y_t^*(\pi)|S_0=s, A_0 = a\}. 
# \end{eqnarray*}
# 
# We need two critical assumptions for the MDP model. 
# 
# **(MA) Markov assumption**:  there exists a Markov transition kernel $\mathcal{P}$ such that  for any $t\ge 0$, $\bar{a}_{t}\in \{0,1\}^{t+1}$ and $\mathcal{S}\subseteq \mathbb{R}^d$, we have 
# $\mathbb{P}\{S_{t+1}^*(\bar{a}_{t})\in \mathcal{S}|W_t^*(\bar{a}_t)\}=\mathcal{P}(\mathcal{S};a_t,S_t^*(\bar{a}_{t-1})).$
# 
# **(CMIA) Conditional mean independence assumption**: there exists a function $r$ such that  for any $t\ge 0, \bar{a}_{t}\in \{0,1\}^{t+1}$, we have 
# $\mathbb{E} \{Y_t^*(\bar{a}_t)|S_t^*(\bar{a}_{t-1}),W_{t-1}^*(\bar{a}_{t-1})\}=r(a_t,S_t^*(\bar{a}_{t-1}))$.
# 
# 
# ## Policy Evaluation
# 
# To either purely evaluate a policy or improve over it, we need to understand its performance (ideally at every state-action tuple), which corresponds to the policy value function estimation and evaluation problem. 
# We introduce two main appraoches in the section. 
# 
# **Monte Carlo (MC).** In an online environment, the most straightforward approach is to just sample trajectories and use the average observed cumulative reward from sub-trajectories that satisfy our conditions as the estimator. 
# Due to the sampling nature, this approach is typically referred to as Monte Carlo {cite:p}`singh1996reinforcement`. 
# For example, to estimate the value $V^{\pi}(s)$ for a given state $s$, we can sample $N$ trajectories following $\pi$, then find time points where we visit state $s$, and finally use the returns from then on to construct an average as our value estimate. 
# 
# 
# 
# **Temporal-Difference (TD) Learning.** 
# One limitation of MC is that one has to wait until the end of a trajectory to collect a data point, which makes it less online and incremental. 
# An alternative is to leverage the Bellman equation and the dynamic optimization structure, as we have utilized in [Paradigm 2](section:FQE). 
# The is known as the Temporal-Difference (TD) Learning {cite:p}`sutton1988learning`. 
# The name is from the fact that it involves the estimate at time point $t$ and $t+1). 
# We first recall the Bellman equation:
# 
# \begin{equation}\label{eqn:bellman_Q}
#     Q^\pi(a, s) = \mathbb{E}^\pi \Big(R_t + \gamma Q^\pi(A_{t + 1}, S_{t+1})  | A_t = a, S_t = s \Big). 
# \end{equation}
# 
# Therefore, suppose we currently have an Q-function estimate $\hat{Q}^{\pi}$. 
# Then, after collecting a trasition tuple $(s, a, r, s')$, we can then update the estimate of $\hat{Q}^{\pi}(s, a)$ as 
# \begin{equation}
#     \hat{Q}^\pi(a, s) + \alpha \Big[r + \gamma \hat{Q}^\pi(\pi(s'), s')  - \hat{Q}^\pi(a, s) \Big], 
# \end{equation}
# where $\alpha$ is a learning rate. 
# 
# 
# 
# **Statistical inference.** As discussed in [Paradigm 4](section:Direct Online Policy Evaluator), statistical inference with adaptively collected data is challenging. 
# To address that issue, {cite:t}`shi2020statistical` leverages a carefully designed data splitting schema to provide valid asymptotic distribution (and hence the confidence interval). 
# 
# 
# 
# 
# ## Policy Optimization
# 
# The online policy optimization problem with MDP is the most well-known RL problem. 
# There are three major approaches: policy-based (policy gradient), value-based (approximate DP), and actor critic. 
# We will focus on illustrate their main idea and connection to other topics in the book. 
# 
# ### Policy gradient
# The policy-based algorithms (e.g., REINFORCE {cite:t}`williams1992simple`, TRPO {cite:t}`schulman2015trust`, PPO {cite:t}`schulman2017proximal`) directly learn a policy function $\pi$ by applying gradient descent to optimize its value. 
# In its simplist form, the value estimation is obtained via MC, i.e., sampling trajectories following a policy. 
# However, the gradient descient is not straightforward, as it requires the value estimation of any policies around the current one to compute the gradient, which is not feasible. 
# Fortunately, we have the following *policy gradient theorem*. 
# 
# \begin{align}
# \bigtriangledown_{\theta}\, J(\theta)
# &= 
# \mathbb{E}_{\tau \sim \pi(\cdot;\theta)}
# \big\{
# G(\tau) \times
# \big[
# \sum_{t=0}^T 
# \bigtriangledown_{\theta}\, 
# \log \pi(A_t| S_t; \theta)
# \big]
# \big\}, 
# \label{eqn:REINFORCE}
# \end{align}
# where $\tau$ represents a trajectory and $\theta$ parameterizes the policy. 
# 
# 
# 
# ### Value-based (Approximate DP)
# 
# The second appraoch is closely related to the Q-function-based appraoch discussed in Paradigm 1 and 2, and in particular, the [FQI](section:FQI) algorithm. 
# 
# Recall the Bellman optimality equations: $Q^*$ is the unique solution of 
# \begin{equation}
#     Q(a, s) = \mathbb{E} \Big(R_t + \gamma \arg \max_{a'} Q(a, S_{t+1})  | A_t = a, S_t = s \Big).  \;\;\;\;\; \text{(1)}. 
# \end{equation}
# Since the right-hand side of (1) is a contraction mapping on $Q$ and its fixed point is $Q^*$, we can iteratively solve the following problem to update the estimation of $Q^*$: 
# 
# \begin{eqnarray}
# 	\widehat{Q}^{{\ell}}=\arg \min_{Q} 
# 	\sum_{(s,a,r,s') \sim D}
# 	\Big\{
# 	\gamma \max_{a'} \widehat{Q}^{\ell-1}(a', s') 
#     +r- Q(a, s)
# \Big\}^2.  \;\;\;\;\; \text{(2)}. 
# \end{eqnarray}
# 
# One major difference lies in how the optimization above is done. 
# In the offline setting (e.g., [FQI](section:FQI)), $D$ is the fixed batch dataset and the optimization is solved fully. 
# In the original online Q-learning {cite:p}`watkins1992q` algorithm, we only run one gradient descent step in the optimization. 
# There are many variants to improve this idea in different practical ways. 
# Besides, a unique feature of the online setting is the ability to collect new data (i.e., exploration), and how to efficiently explore is another new problem compared with the offline setup. 
# For example, DQN {cite:p}`mnih2015human` maintains a replay buffer and apply epsilon-greedy for exploration, and Double DQN {cite:p}`van2016deep`) uses two different Q-estimators in the RHS of (2) to solve the over estimation issue.
# 
# ### Actor Critic
# 
# One limitation of the policy gradient approach is the efficiency, since it is heavy to sample new trajectories from scratch to evaluate the current policy and hence has high variance. 
# A natural idea is: 
# if we have a value estimator, then it can be used for policy evaluation as well. 
# Such a motivation is well grounded by the following relationship (there are more extensions): 
# 
# \begin{align}
# \bigtriangledown_{\theta}\, J(\theta)
# &= 
# \mathbb{E}_{\tau \sim \pi(\cdot;\theta)}
# \big\{
# G(\tau) \times
# \big[
# \sum_{t=0}^T 
# \bigtriangledown_{\theta}\, 
# \log \pi(A_t| S_t; \theta)
# \big]
# \big\}\\
# &= 
# \mathbb{E}_{\tau \sim \pi(\cdot;\theta)}
# \big\{
# \sum_{t=0}^T 
# Q_t^{\theta}(S_t, A_t)
# \bigtriangledown_{\theta}\,
# \log \pi(A_t| S_t; \theta)
# % Q_0^{\theta}(s,a) \bigtriangledown_{\theta}\, \pi_\theta(s)
# \big\}. 
# \end{align}
# 
# Therefore, an actor-critic maintains both a policy function estimator (the actor) to select actions, and an value function estimator (the critic) that evaluates the current policy and guides the direction to apply gradient descent. 
# It hences combines the idea of the first two approaches in this section to improve effiicency. 
# From the other direction, it shares similar ideas with the direct policy search appraoch that we disucssed in Paradigm 1. 
# Popular actor-critic algorithms include A2C {cite:t}`mnih2016asynchronous`, SAC {cite:t}`haarnoja2018soft`, A3C {cite:t}`mnih2016asynchronous`)

# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
