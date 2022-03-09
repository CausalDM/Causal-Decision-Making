#!/usr/bin/env python
# coding: utf-8

# # Q-Learning (Multiple Stages)

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## Main Idea
# 
# Q-learning is a classic method of Reinforcement Learning. Early in 2000, it was adapted to decision-making problems[1] and kept evolving with various extensions, such as penalized Q-learning [2]. In the following, we would start from a simple case having only one decision point and then introduce the multistage case with multiple decision points. Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., A,B,C,D), and the outcome of interest Y is **continuous** and **non-negative**, where the larger the $Y$ the better. 
# 

# ## Multiple Decision Points
# 
# - **Application Situation**: Suppose we have a dataset containning observations from $N$ individuals. For each individual $i$, the observed data is structured as follows
#     \begin{align}
#     (X_{1i},A_{1i},\cdots,X_{Ti},A_{Ti},Y), i=1,\cdots, N.
#     \end{align} 
#     Let $h_{ti}=\{X_{1i},A_{1i},\cdots,X_{ti}\})$ includes all the information observed till step t. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received at the end of the final decision point $T$. In other words, by training a model with the observed dataset, we want to find an optimal policy that can help us determine the optimal sequence of actions for each individual to optimize the reward.
# 
# - **Basic Logic**: For multistage cases, we apply a backward iterative approach, which means that we start from the final decision point T and work our way backward to the initial decision point. At the final step $T$, it is again a standard regression modeling problem that is the same as what we did for the single decision point case. Particularly, we posit a model $Q_{T}(h_{T},a_{T})$ for the observed outcome $Y$, and then the optimal policy at step $T$ is derived as $\text{arg max}_{\pi_{T}}Q_{T}(h_{T},\pi_{T}(h_{T}))$. For the decision point $T-1$ till the decision point $1$, a new term is introduced, which is the pseudo-outcome $\tilde{Y}_{t}$.
#     \begin{align}
#     \tilde{Y}_{t} = \text{max}_{\pi_{t}}\hat{Q}_{t}(h_{t},\pi_{t}(h_{t}),\hat{\beta}_{t})
#     \end{align}
#     By doing so, the pseudo-outcome taking the **delayed effect** into account to help explore the optimal policy. Then, for each decision point $t<T$, with the $\tilde{Y}_{t+1}$ calculated, we repeat the regression modeling step for $\tilde{Y}_{t+1}$. After obtaining the fitted model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$, the optimal policy is obtained as $\text{arg max}_{\pi_{t}}Q_{t}(h_{t},\pi_{t}(h_{t}))$.
# 
# - **Key Steps**: 
#     1. At the final decision point $t=T$, fitted a model $\hat{Q}_{T}(h_{T},a_{T},\hat{\beta}_{T})$ for the observed outcome $Y$;
#     2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{Ti}=\text{max}_{\pi}\hat{Q}_{T}(h_{Ti},\pi(h_{Ti}),\hat{\beta}_{T})$, and the optimal action $a_{Ti}=\text{arg max}_{a}\hat{Q}_{T}(h_{Ti},a,\hat{\beta}_{T})$;
#     3. For decision point $t = T-1,\cdots, 1$,
#         1. fitted a model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$ for the pseudo-outcome $\tilde{Y}_{t+1}$
#         2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{ti}=\text{max}_{\pi}\hat{Q}_{t}(h_{ti},\pi(h_{ti}),\hat{\beta}_{t})$, and the optimal action $a_{ti}=\text{arg max}_{a}\hat{Q}_{t}(h_{ti},a,\hat{\beta}_{t})$;

# ### 1. Optimal Decision

# #### import package

# In[1]:


# TODO: feasible set
from causaldm.learners import QLearning
from causaldm.test import shared_simulation
import numpy as np


# #### prepare the dataset

# In[2]:


#prepare the dataset (dataset from the DTR book)
import pandas as pd
#Important!! reset the index is required
dataMDP = pd.read_csv("dataMDP_feasible.txt", sep=',')#.reset_index(drop=True) 
Y = dataMDP['Y']
X = dataMDP[['CD4_0','CD4_6','CD4_12']]
A = dataMDP[['A1','A2','A3']]


# #### train policy

# In[3]:


# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
# If want to include all the variable in X and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]

# train the policy
QLearn.train(X, A, Y, model_info, T=3)


# #### get the recommend actions and optimal value

# In[4]:


# recommend action
opt_d = QLearn.recommend().head()
# get the estimated value of the optimal regime
V_hat = QLearn.estimate_value()
print("opt regime:",opt_d)
print("opt value:",V_hat)


# In[5]:


QLearn.recommend().sum(axis=0)


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
QLearn.train(X, A, Y, model_info, T=3, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()
print('Value_hat:',value_avg,'Value_std:',value_std)


# ### 2. Policy Evaluation

# We use the backward iteration as before. However, here for each round, the pseudo outcome is not the maximum of Q values. Instead, the pseudo outcome at decision point t is defined as below:
# 
# \begin{align}
# \tilde{Y}_{t} = \hat{Q}_{t}(h_{t},d_{t}(h_{t}),\hat{\beta}_{t})
# \end{align}, where $d$ is the fixed regime that we want to evaluate.
# 

# In[10]:


#specify the fixed regime to be tested
# For example, regime d = 1 for all subjects at all decision points\
N=len(X)
# !! IMPORTANT: INDEX SHOULD BE THE SAME AS THAT OF THE X,Y,A
regime = pd.DataFrame({'A1':[1]*N,
                      'A2':[1]*N,
                      'A3':[1]*N}).set_index(X.index)
#evaluate the regime
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]
QLearn.train(X, A, Y, model_info, T=3, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()


# In[11]:


# bootstrap average and the std of estimate value
print('Value_hat:',value_avg,'Value_std:',value_std)


# In[12]:


# Otional: just estimate the value
QLearn.train(X, A, Y, model_info, T=3, regime = regime, evaluate = True)
QLearn.estimate_value()


# 💥 Placeholder for C.I.

# ## References
# 1. Murphy, S. A. (2005). A generalization error for Q-learning.
# 2. Song, R., Wang, W., Zeng, D., & Kosorok, M. R. (2015). Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3), 901.

# !! Already tested for accuracy using the data provided in DTR book

# In[ ]:




