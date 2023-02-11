#!/usr/bin/env python
# coding: utf-8

# # Q-Learning (Multiple Stages)
# 
# ## Main Idea
# Early in 2000, as a classic method of Reinforcement Learning, Q-learning was adapted to decision-making problems[1] and kept evolving with various extensions, such as penalized Q-learning [2]. Q-learning with finite decision points is mainly a regression modeling problem based on positing regression models for outcome at each decision point. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received at the end of the final decision point. In other words, by training a model with the observed data, we hope to find an optimal policy to predict the optimal action for each individual to maximize rewards. For example, considering the motivating example **Personalized Incentives**, Q-learning aims to find the best policy to assign different incentives ($A$) to different users to optimize the return-on-investment ($R$). Overall, Q-learning is practical and easy to understand, as it allows straightforward implementation of diverse established regression methods. 
# 
# 
# Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., A,B,C,D), and the outcome of interest R is **continuous** and **non-negative**, where the larger the $R$ the better.
# 
# ## Algorithm Details
# For multistage cases, we apply a backward iterative approach, which means that we start from the final decision point T and work our way backward to the initial decision point. At the final step $T$, it is again a standard regression modeling problem that is the same as what we did for the single decision point case. Particularly, we posit a model $Q_{T}(h_{T},a_{T})$ for the expectation of potential outcome $R(\bar{a}_T)$, and then the optimal policy at step $T$ is derived as $\text{arg max}_{\pi_{T}}Q_{T}(h_{T},\pi_{T}(h_{T}))$. For the decision point $T-1$ till the decision point $1$, a new term is introduced, which is the pseudo-outcome $\tilde{R}_{t}$.
#     \begin{align}
#     \tilde{R}_{t} = \text{max}_{\pi_{t}}\hat{Q}_{t}(h_{t},\pi_{t}(h_{t}),\hat{\beta}_{t})
#     \end{align}
#     By doing so, the pseudo-outcome taking the **delayed effect** into account to help explore the optimal policy. Then, for each decision point $t<T$, with the $\tilde{R}_{t+1}$ calculated, we repeat the regression modeling step for $\tilde{R}_{t+1}$. After obtaining the fitted model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$, the optimal policy is obtained as $\arg \max_{\pi_{t}}Q_{t}(h_{t},\pi_{t}(h_{t}))$.
# 
# 
# ## Key Steps
# **Policy Learning:**
# 1. At the final decision point $t=T$, fitted a model $Q_{T}(h_{T},a_{T},\beta_{T})$;
# 2. For each individual $i$, calculated the pseudo-outcome $\tilde{R}_{Ti}=\text{max}_{\pi}\hat{Q}_{T}(h_{Ti},\pi(h_{Ti}),\hat{\beta}_{T})$, and the optimal action $d^{opt}_{T}(s_{i})=\text{arg max}_{a}\hat{Q}_{T}(h_{Ti},a,\hat{\beta}_{T})$;
# 3. For decision point $t = T-1,\cdots, 1$,
#     1. fitted a model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$ for the pseudo-outcome $\tilde{R}_{t+1}$
#     2. For each individual $i$, calculated the pseudo-outcome $\tilde{R}_{ti}=\text{max}_{\pi}\hat{Q}_{t}(h_{ti},\pi(h_{ti}),\hat{\beta}_{t})$, and the optimal action $d^{opt}_{t}(s_{i})=\text{arg max}_{a}\hat{Q}_{t}(h_{ti},a,\hat{\beta}_{t})$;
#     
# **Policy Evaluation:**    
# We use the backward iteration as what we did in policy learning. However, here for each round, the pseudo outcome is not the maximum of Q values. Instead, the pseudo outcome at decision point t is defined as below:
# \begin{align}
# \tilde{R}_{ti} = \hat{Q}_{t}(h_{ti},d_{t}(h_{ti}),\hat{\beta}_{t}),
# \end{align} where $d$ is the fixed regime that we want to evaluate.
# The estimated value of the policy is then the average of $\tilde{R}_{1i}$.
# 
# **Note** we also provide an option for bootstrapping. Particularly, for a given policy, we utilize bootstrap resampling to get the estimated value of the regime and the corresponding estimated standard error.
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# TODO: feasible set
from causaldm.learners import QLearning
from causaldm.test import shared_simulation
import numpy as np


# In[2]:


#prepare the dataset (dataset from the DTR book)
import pandas as pd
#Important!! reset the index is required
dataMDP = pd.read_csv("dataMDP_feasible.txt", sep=',')#.reset_index(drop=True) 
R = dataMDP['Y']
S = dataMDP[['CD4_0','CD4_6','CD4_12']]
A = dataMDP[['A1','A2','A3']]


# In[3]:


# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
# If want to include all the variable in S and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]

# train the policy
QLearn.train(S, A, R, model_info, T=3)


# In[4]:


#4. recommend action
opt_d = QLearn.recommend_action(S).value_counts()
#5. get the estimated value of the optimal regime
V_hat = QLearn.predict_value(S)
print("fitted model Q0:",QLearn.fitted_model[0].params)
print("fitted model Q1:",QLearn.fitted_model[1].params)
print("fitted model Q2:",QLearn.fitted_model[2].params)
print("opt regime:",opt_d)
print("opt value:",V_hat)


# In[5]:


QLearn.recommend_action(S).value_counts()


# In[6]:


# Optional: we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_0*A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_0*A3+CD4_6*A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]
QLearn.train(S, A, R, model_info, T=3, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


# ### 2. Policy Evaluation

# In[7]:


#specify the fixed regime to be tested
# For example, regime d = 1 for all subjects at all decision points\
N=len(S)
# !! IMPORTANT: INDEX SHOULD BE THE SAME AS THAT OF THE S,R,A
regime = pd.DataFrame({'A1':[1]*N,
                      'A2':[1]*N,
                      'A3':[1]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]
QLearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True)
QLearn.predict_value(S)


# In[8]:


# bootstrap average and the std of estimate value
QLearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


# 💥 Placeholder for C.I.

# ## References
# 1. Murphy, S. A. (2005). A generalization error for Q-learning.
# 2. Song, R., Wang, W., Zeng, D., & Kosorok, M. R. (2015). Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3), 901.

# !! Already tested for accuracy using the data provided in DTR book
