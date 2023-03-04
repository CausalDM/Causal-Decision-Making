#!/usr/bin/env python
# coding: utf-8

# ## CPL: 2-Stage DTR

# In[1]:


import pandas as pd
DTR_data = pd.read_csv('mimic3_DTR.csv')


# In[2]:


DTR_data.head()


# In[3]:


import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
from causaldm.learners import QLearning


# ## Binary IV_Input

# As an example, we use the **Q-learning** algorithm to optimize and evaluate policy. We begin by training the Q models on the observed data.  By assuming linear regression models and using the code in the following block, we have that A = B.
# \begin{align}
# Q_1(s,a_1) = &.7725+.0016*\textrm{Glucose}_1-.0015*\textrm{paO2}_1+.0003*\textrm{PaO2_FiO2}_1-.0089*\textrm{SOFA}_1+\\
#                     &I(a_1=1)*\{-.1308-.0021*\textrm{Glucose}_1-.0054*\textrm{paO2}_1+.0021*\textrm{PaO2_FiO2}_1+.0973*\textrm{SOFA}_1\},\\
# Q_2(s,a_2) = &2.4362+.0008*\textrm{Glucose}_1-.0008*\textrm{paO2}_1+.0004*\textrm{PaO2_FiO2}_1+.0363*\textrm{SOFA}_1+\\
#                     &.0022*\textrm{Glucose}_2-.0144*\textrm{paO2}_2-.0003*\textrm{PaO2_FiO2}_2-.1306*\textrm{SOFA}_2+\\
#                     &I(a_2=1)*\{-1.5796-.0044*\textrm{Glucose}_2+.0145*\textrm{paO2}_2+.0007*\textrm{PaO2_FiO2}_2+.0649*\textrm{SOFA}_2\}.
# \end{align}
# 
# 

# In[4]:


DTR_data.rename(columns = {'Died_within_48H':'R',
                            'Glucose_1':'S1_1', 'Glucose_2':'S1_2',
                            'paO2_1':'S2_1', 'paO2_2':'S2_2',
                            'PaO2_FiO2_1':'S3_1', 'PaO2_FiO2_2':'S3_2',
                            'SOFA_1':'S4_1', 'SOFA_2':'S4_2',
                            'IV_Input_1':'A1','IV_Input_2':'A2'}, inplace = True)
R = DTR_data['R'] #lower the better
S = DTR_data[['S1_1','S1_2','S2_1','S2_2','S3_1','S3_2','S4_1','S4_2']]
A = DTR_data[['A1','A2']]
# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
model_info = [{"model": "R~S1_1+S2_1+S3_1+S4_1+A1+S1_1*A1+S2_1*A1+S3_1*A1+S4_1*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "R~S1_1+S2_1+S3_1+S4_1+S1_2+S2_2+S3_2+S4_2+A2+S1_2*A2+S2_2*A2+S3_2*A2+S4_2*A2",
              'action_space':{'A2':[0,1]}}]
# train the policy
QLearn.train(S, A, R, model_info, T=2, mimic3_clip = True)
# get the summary of the fitted Q models using the following code
#print("fitted model Q0:",QLearn.fitted_model[0].summary())
#print("fitted model Q1:",QLearn.fitted_model[1].summary())


# **Policy Optimization:** Based on the fitted Q models, an optimal treatment regime can be defined as
# - At stage 1:
#     1. We would recommend $A=0$ (IV_Input = 0) if $-.0021*\textrm{Glucose}_1-.0054*\textrm{paO2}_1+.0021*\textrm{PaO2_FiO2}_1+.0973*\textrm{SOFA}_1<.1308$
#     2. Else, we would recommend $A=1$ (IV_Input = 1).
# - At stage 2:
#     1. We would recommend $A=0$ (IV_Input = 0) if $-.0044*\textrm{Glucose}_2+.0145*\textrm{paO2}_2+.0007*\textrm{PaO2_FiO2}_2+.0649*\textrm{SOFA}_2<1.5796$
#     2. Else, we would recommend $A=1$ (IV_Input = 1).
#     
# In the block that follows, we apply the estimated optimal regime to individuals in the observed data and estimate the corresponding policy value. Following the estimated optimal policy, 
# - 20 patients are recommended to be treated with IV input at both stages, 
# - nine patients are recommended not to be treated at both stages, 
# - 19 patients are recommended to be treated with IV input at the first stage and not to be treated at the second stage,
# - seven patients are recommended not to be treated at the first stage but to be treated with IV input at the second stage.
# 
# The estimated value of the estimated optimal policy is **.9357**.

# In[5]:


#4. recommend action
opt_d = QLearn.recommend_action(S).value_counts()
#5. get the estimated value of the optimal regime
V_hat = QLearn.predict_value(S)
print("opt_d:",opt_d)
print("opt value:",V_hat)


# **Policy Evaluation:** Now, we are interested in evaluating a fixed policy that never gives IV input to patients. Under the specified model, the estimated value of the fixed policy that always does not give IV input is .4107.

# In[6]:


#specify the fixed regime to be tested
# For example, regime d = 1 for all subjects at all decision points\
N=len(S)
# !! IMPORTANT: INDEX SHOULD BE THE SAME AS THAT OF THE S,R,A
regime = pd.DataFrame({'A1':[0]*N,
                      'A2':[0]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
QLearn.train(S, A, R, model_info, T=2, regime = regime, evaluate = True, mimic3_clip = True)
QLearn.predict_value(S)


# In[ ]:





# In[ ]:




