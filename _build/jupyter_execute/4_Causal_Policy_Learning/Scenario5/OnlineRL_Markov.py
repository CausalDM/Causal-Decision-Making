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
# 1. MC
# 
# 2. TD
# 
# 3. SAVE
# 
# ## Policy Optimization
# 
# 1. DQN

# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
