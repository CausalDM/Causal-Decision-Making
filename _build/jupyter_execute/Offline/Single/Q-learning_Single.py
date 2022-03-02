#!/usr/bin/env python
# coding: utf-8

# # Q-Learning (Single Stage)

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

# ## Single Decision Point
# 
# - **Application Situation**: Suppose we have a dataset containing observations from $N$ individuals. For each individual $i$, we have $\{\mathbf{X}_{i},A_{i},Y_{i}\}$, $i=1,\cdots,N$. $\mathbf{X}_{i}$ includes the feature information, $A_{i}$ is the action taken, and $Y_{i}$ is the observed reward received. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received. In other words, by training a model with the observed dataset, we want to find an optimal policy that can help us determine the optimal action for each individual to optimize the reward.
# 
# - **Basic Logic**: Q-learning with a single decision point is mainly a regression modeling problem, as the major component is to find the relationship between $Y$ and $\{X,A\}$. Let's first define a Q-function, such that
# \begin{align}
#     Q(x,a) = E(Y|X=x, A=a).
# \end{align} Then, to find the optimal policy is equivalent to solve
# \begin{align}
#     \text{arg max}_{\pi}Q(x_{i},\pi(x_{i})).
# \end{align} 
# 
# - **Key Steps**:
#     1. Fitted a model $\hat{Q}(x,a,\hat{\beta})$, which can be solved directly by existing approaches (i.e., OLS, .etc),
#     2. For each individual find the optimal action $a_{i}$ such that $a_{i} = \text{arg max}_{a}\hat{Q}(x_{i},a,\hat{\beta})$.

# ### 1. Optimal Decision

# #### import the learner

# In[1]:


# A demo with code on how to use the package
from causaldm.learners import QLearning
import pandas as pd
import numpy as np
from sklift.datasets import fetch_hillstrom


# #### prepare the dataset (dataset from the DTR book)

# In[2]:


# continuous Y
data, target, treatment = fetch_hillstrom(target_col='spend', return_X_y_t=True)
# use pd.concat to join the new columns with your original dataframe
data = pd.concat([data,pd.get_dummies(data['zip_code'], prefix='zip_code')],axis=1)
data = pd.concat([data,pd.get_dummies(data['channel'], prefix='channel')],axis=1)
# now drop the original 'country' column (you don't need it anymore)
data.drop(['zip_code'],axis=1, inplace=True)
data.drop(['channel'],axis=1, inplace=True)
data.drop(['history_segment'],axis=1, inplace=True)
data.drop(['zip_code_Rural'],axis=1, inplace=True) # Rural as the reference group
data.drop(['channel_Multichannel'],axis=1, inplace=True) # Multichannel as the reference group 

Y = target
Y.name = 'Y'
X = data# add an intercept column
A = treatment
A.name = 'A'
#get the subset which has Y>0 == n=578
X = X[Y>0]
A = A[Y>0]
Y = Y[Y>0]
X.columns


# In[3]:


X


# #### train model

# In[4]:


# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
# If want to include all the variable in X and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~C(A)*(recency+history+mens+womens+newbie+zip_code_Surburban+zip_code_Urban+channel_Phone+channel_Web)", #default is add an intercept!!!
              'action_space':{'A':['Womens E-Mail', 'No E-Mail', 'Mens E-Mail']}}]
# train the policy
QLearn.train(X, A, Y, model_info, T=1)
# Fitted Model
QLearn.fitted_model[0].params


# #### get the optimal regime and the optimal value

# In[5]:


# recommend action
opt_d = QLearn.recommend().head()
# get the estimated value of the optimal regime
V_hat = QLearn.estimate_value()
print("opt regime:",opt_d)
print("opt value:",V_hat)


# In[6]:


# Optional: we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~C(A)*(recency+history+mens+womens+newbie+zip_code_Surburban+zip_code_Urban+channel_Phone+channel_Web)", #default is add an intercept!!!
              'action_space':{'A':['Womens E-Mail', 'No E-Mail', 'Mens E-Mail']}}]
QLearn.train(X, A, Y, model_info, T=1, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()
print('Value_hat:',value_avg,'Value_std:',value_std)


# ### 2. Policy Evaluation

# For a given policy, we utilze the boostrap resampling to get the estimated value of the regime and the corresponding estimated standard error. Basically, for each round of bootstrap, we resample a dataset of the same size as the original dataset with replacement, fitted the Q function based on the sampled dataset, and estimated the value of a given regime using the estimated Q function. 

# In[9]:


#specify the fixed regime to be tested
# For example, regime d = 'No E-Mail' for all subjects
N=len(X)
# !! IMPORTANT： index shold be the same as that of the X
regime = pd.DataFrame({'A':['No E-Mail']*N}).set_index(X.index)
#evaluate the regime
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~C(A)*(recency+history+mens+womens+newbie+zip_code_Surburban+zip_code_Urban+channel_Phone+channel_Web)", #default is add an intercept!!!
              'action_space':{'A':['Womens E-Mail', 'No E-Mail', 'Mens E-Mail']}}]
QLearn.train(X, A, Y, model_info, T=1, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()


# In[10]:


# bootstrap average and the std of estimate value
print('Value_hat:',value_avg,'Value_std:',value_std)


# In[11]:


# Otional: just estimate the value
QLearn.train(X, A, Y, model_info, T=1, regime = regime, evaluate = True)
QLearn.estimate_value()


# ### TODO: 
#     1. estimate the standard error for the binary case with sandwich formula;
#     2. inference for the estimated optimal regime: projected confidence interval? m-out-of-n CI?....

# ## References
# 1. Murphy, S. A. (2005). A generalization error for Q-learning.
# 2. Song, R., Wang, W., Zeng, D., & Kosorok, M. R. (2015). Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3), 901.

# ## !!Functions are already tested with the data and results provided in the DTR book
