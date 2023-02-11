#!/usr/bin/env python
# coding: utf-8

# # Q-Learning (Single Stage)
# 
# ## Main Idea
# Early in 2000, as a classic method of Reinforcement Learning, Q-learning was adapted to decision-making problems[1] and kept evolving with various extensions, such as penalized Q-learning [2]. Q-learning with finite decision points is mainly a regression modeling problem based on positing regression models for outcome at each decision point. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received. In other words, by training a model with the observed data, we hope to find an optimal policy to predict the optimal action for each individual to maximize rewards. For example, considering the motivating example **Personalized Incentives**, Q-learning aims to find the best policy to assign different incentives ($A$) to different users to optimize the return-on-investment ($R$). Overall, Q-learning is practical and easy to understand, as it allows straightforward implementation of diverse established regression methods. 
# 
# 
# Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., A,B,C,D), and the outcome of interest R is **continuous** and **non-negative**, where the larger the $R$ the better.
# 
# 
# 
# ## Algorithm Details
# Q-learning with a single decision point is mainly a regression modeling problem, as the major component is to find the relationship between the expectation of potential reward $R(a)$ and $\{s,a\}$. Let's first define a Q-function, such that
# \begin{align}
#     Q(s,a) = E(R(a)|S=s).
# \end{align} Then, to find the optimal policy is equivalent to solve
# \begin{align}
#     \text{arg max}_{\pi}Q(s_{i},\pi(s_{i})).
# \end{align} 
# 
# ## Key Steps
# **Policy Learning:**
# 1. Fitted a model $\hat{Q}(s,a,\hat{\beta})$, which can be solved directly by existing approaches (i.e., OLS, .etc),
# 2. For each individual find the optimal action $d^{opt}(s_{i})$ such that $d^{opt}(s_{i}) = \text{arg max}_{a}\hat{Q}(s_{i},a,\hat{\beta})$.
# 
# **Policy Evaluation:**    
# 1. Fitted the Q function $\hat{Q}(s,a,\hat{\beta})$, based on the sampled dataset
# 2. Estimated the value of a given regime $d$ (i.e., $V(d)$) using the estimated Q function, such that, $\hat{E}(R_{i}[d(s_{i})]) = \hat{Q}(s_{i},d(s_{i}),\hat{\beta})$, and $\hat{V}(d) = \frac{1}{N}\sum_{i=1}^{N}\hat{E}(R_{i}[d(s_{i})])$.
# 
# **Note** we also provide an option for bootstrapping. Particularly, for a given policy, we utilize bootstrap resampling to get the estimated value of the regime and the corresponding estimated standard error. For each round of bootstrapping, we first resample a dataset of the same size as the original dataset, then fit the Q function based on the sampled dataset, and finally estimate the value of a given regime based on the estimated Q function. 
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# import learner
from causaldm._util_causaldm import *
from causaldm.learners import QLearning


# In[ ]:


# get the data
S,A,R = get_data(target_col = 'spend', binary_trt = False)


# In[ ]:


#1. specify the model you would like to use
# If want to include all the variable in S and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~C(A)*(recency+history)", #default is add an intercept!!!
              'action_space':{'A':[0,1,2]}}]


# By specifing the model_info, we assume a regression model that:
# \begin{align}
# Q(s,a,\beta) &= \beta_{00}+\beta_{01}*recency+\beta_{02}*history\\
# &+I(a=1)*\{\beta_{10}+\beta_{11}*recency+\beta_{12}*history\} \\
# &+I(a=2)*\{\beta_{20}+\beta_{21}*recency+\beta_{22}*history\} 
# \end{align}

# In[ ]:


#2. initialize the learner
QLearn = QLearning.QLearning()
#3. train the policy
QLearn.train(S, A, R, model_info, T=1)


# In[ ]:


#4. recommend action
opt_d = QLearn.recommend_action(S).value_counts()
#5. get the estimated value of the optimal regime
V_hat = QLearn.predict_value(S)
print("fitted model:",QLearn.fitted_model[0].params)
print("opt regime:",opt_d)
print("opt value:",V_hat)


# **Interpretation:** the fitted model is 
# \begin{align}
# Q(s,a,\beta) &= 94.20+4.53*recency+0.0005*history\\
# &+I(a=1)*\{23.24-4.15*recency+0.0076*history\} \\
# &+I(a=2)*\{20.61-4.84*recency+0.0004history\}. 
# \end{align}
# Therefore, the estimated optimal regime is:
# 1. We would recommend $A=0$ (No E-mail) if $23.24-4.15*recency+0.0076*history<0$ and $20.61-4.84*recency+0.0004history<0$
# 2. Else, we would recommend $A=1$ (Womens E-mail) if $23.24-4.15*recency+0.0076*history>20.61-4.84*recency+0.0004history$
# 3. Else, we would recommend $A=2$ (Mens E-Mail).
# 
# The estimated value for the estimated optimal regime is 126.49.

# In[ ]:


# Optional: 
#we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~C(A)*(recency+history)", #default is add an intercept!!!
              'action_space':{'A':[0,1,2]}}]
QLearn.train(S, A, R, model_info, T=1, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


# **Interpretation:** Based on the boostrap with 200 replicates, the estimated optimal value is 132.31 with a standard error of 7.37.

# ### 2. Policy Evaluation

# In[ ]:


#1. specify the fixed regime to be tested (For example, regime d = 'No E-Mail' for all subjects)
# !! IMPORTANT： index shold be the same as that of the S
N=len(S)
regime = pd.DataFrame({'A':[0]*N}).set_index(S.index)
#2. evaluate the regime
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~C(A)*(recency+history)", #default is add an intercept!!!
              'action_space':{'A':[0,1,2]}}]
QLearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True)
QLearn.predict_value(S)


# **Interpretation:** the estimated value of the regime that always sends no emails ($A=0$) is 116.41, under the specified model.

# In[ ]:


# Optional: Boostrap
QLearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.predict_value_boots(S)
# bootstrap average and the std of estimate value
print('Value_hat:',value_avg,'Value_std:',value_std)


# **Interpretation:** the bootstrapped estimated value of the regime that always sends no emails is 115.96 with a bootstrapped standard error 10.50, under the specified model.

# ## References
# 1. Murphy, S. A. (2005). A generalization error for Q-learning.
# 2. Song, R., Wang, W., Zeng, D., & Kosorok, M. R. (2015). Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3), 901.

# !!Functions are already tested with the data and results provided in the DTR book
# 
# TODO: 
#     1. estimate the standard error for the binary case with sandwich formula;
#     2. inference for the estimated optimal regime: projected confidence interval? m-out-of-n CI?....

# In[ ]:




