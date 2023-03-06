#!/usr/bin/env python
# coding: utf-8

# (section:Direct Online Policy Evaluator)=
# # Direct Online Policy Evaluator
# 
# Evaluating the performance of an ongoing policy plays a vital role in many areas such as medicine and economics, to provide crucial instruction on the early-stop of the online experiment and timely feedback from the environment. Policy evaluation in online learning thus attracts increasing attention by inferring the mean outcome
# of the optimal policy (i.e., the value) in real-time. We introduce the **direct online policy evaluator** to infer the value under the estimated optimal policy in online learning. 
# 
# ***Application Situations***: 
#     
# 1. Dependent data generated in the online environment;
# 2. Unknown optimal policy or known fixed policy;
# 3. Contain exploration and exploitation trade-off;
# 4. The underlying reward function can be easily modeled. 
# 
# ***Advantage of the Evaluator***:
# 
# 1. Provide a Wald-type confidence interval on the value estimator;
# 2. Flexible to implement in practice. 
# 
# 
# ## Main Idea
# 
# ### Framework
# 
# We focus on online policy evaluation for contextual linear bandits with deterministic policy for illustration. We focus on online policy optimization under three commonly used bandit algorithms for exposition, including Upper Confidence Bound (UCB), Thompson Sampling (TS), and $\epsilon$-Greedy (EG) methods [[[link to be added]]], and propose **direct online policy evaluator** to infer the mean outcome of the optimal online policy. Given a context $\boldsymbol{x}_t$, we choose an action $a_t$ based on a policy  $\pi(\cdot)$ which is defined as a deterministic function that maps the context space to the action space as $\pi: \mathcal{X} \to \mathcal{A}$. We then receive a reward  $r_t$ for all $t \in  \mathcal{T} $. Denote the history observations previous to time step $t$ as $\mathcal{H}_{t-1}=\{\boldsymbol{x}_i,a_i,r_i\}_{1\leq i\leq t-1}$. Define the potential reward $R^*(A=a)$ as the reward we would observe given the action as $A=a$. Then the value (Dudík
# et al., 2011) of a given policy $\pi(\cdot)$ is 
# \begin{equation*}
#  V(\pi)\equiv \mathbb{E}_{\boldsymbol{X}  \sim P_{\mathcal{X}}}\left[\mathbb{E}\{R^*(A=\pi(\boldsymbol{X} ))\}\right] =(SUVTA/Ignoribility)=\mathbb{E}_{\boldsymbol{X}  \sim P_{\mathcal{X}}}\left[\mathbb{E}\{R|\boldsymbol{X}, A =\pi(\boldsymbol{X} )\}\right] =\mathbb{E}_{\boldsymbol{X}  \sim P_{\mathcal{X}}}\left[\mu\{ \boldsymbol{X} ,  \pi(\boldsymbol{X} )\}\right] .
# \end{equation*}
# We define the optimal policy as  $\pi^*(\boldsymbol{x}) \equiv argmax_{a\in \mathcal{A}} \mu(\boldsymbol{x },a),   \forall \boldsymbol{x} \in \mathcal{X}$, that finds the optimal action based on the conditional mean outcome function given a context $\boldsymbol{x}$. 
# Thus, the optimal value can be defined as  $V^*\equiv V(\pi^*)=\mathbb{E}_{\boldsymbol{X}  \sim P_{\mathcal{X}}}\left[\mu\{ \boldsymbol{X} ,  \pi^*(\boldsymbol{X} )\}\right]$. In the rest, to simplify the exposition, we focus on two actions, i.e., $\mathcal{A}=\{0,1\}$.  Then the optimal policy is given by
# \begin{equation*}
# \pi^*(\boldsymbol{x}) \equiv argmax_{a\in \mathcal{A}} \mu(\boldsymbol{x},a) = \mathbb{I}\{\mu(\boldsymbol{x},1)>\mu(\boldsymbol{x},0)\}, \quad \forall \boldsymbol{x} \in \mathcal{X}.
# \end{equation*}  
# Our goal is to infer the value under the optimal policy $\pi^* $ from the online data generated sequentially by some bandit algorithms. Since the optimal policy is unknown, we estimate the optimal policy from the online data as $\widehat{\pi}_t$ by $\mathcal{H}_{t}$.  
# 
# ### Direct Online Policy Evaluator 
# 
# Following the direct value estimator in Dudík et al., (2011), we propose the direct mean outcome estimator as the value estimator under the optimal policy as
#  \begin{equation}\label{dr_est}
# \widehat{V}_T={1\over T }\sum_{t=1}^T \widehat{\mu}_{t-1}\{\boldsymbol{x}_t ,\widehat{\pi}_t(\boldsymbol{x}_t ) \},
# \end{equation}
# where $T$ is the current/termination time, and $\widehat{\mu}_{t}\{\boldsymbol{x}_t ,\widehat{\pi}_t(\boldsymbol{x}_t ) \}$ is the estimated conditional mean function given the current context and action.  
# 
# 
# ## Demo Code
# In the following, we exhibit how to apply the direct online policy evaluator on real data under UCB, TS, and EG, respectively.

# ### 1. Policy Evaluation under UCB

# ### 2. Policy Evaluation under TS

# ### 3. Policy Evaluation under EG

# ## References
# 
# [1] Cai, H., Shen, Y., & Song, R. (2021). Doubly Robust Interval Estimation for Optimal Policy Evaluation in Online Learning. arXiv preprint arXiv:2110.15501.
# 
# [2] Chen, H., Lu, W., & Song, R. (2021). Statistical inference for online decision making: In a contextual bandit setting. Journal of the American Statistical Association, 116(533), 240-255.
# 
# [3] Chen, H., Lu, W., & Song, R. (2021). Statistical inference for online decision making via stochastic gradient descent. Journal of the American Statistical Association, 116(534), 708-719.
# 
# [4] Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601.

# In[ ]:




