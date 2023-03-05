#!/usr/bin/env python
# coding: utf-8

# ## CPL: 3-Stage DTR

# In this notebook, we analyze the MIMIC iii data with 3-stages. At each stage, there are four confounders are measured before taking the treatment, including Glucose, paO2, PaO2_FiO2, and SOFA. IV_Input is the treatment, which is represented as a binary variable, with 0 indicating no treatment and 1 indicating that an IV input is being used. The final outcome is Died_within_48H, with -1 indicating that the patient died within 48 hours and 1 indicating that the patient is alive.

# In[1]:


import pandas as pd
DTR_data = pd.read_csv('mimic3_DTR_3stage.csv')
DTR_data.head()


# In[2]:


import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
from causaldm.learners import QLearning


# ## 3-Stage DTR Policy Evaluation

# As an example, we use the **Q-learning** algorithm to evaluate policies based on the observed data, with the linear regression models defined as the following:
# \begin{align}
# Q_1(s,a_1,\boldsymbol{\beta}) = &\beta_{00}+\beta_{01}*\textrm{Glucose}_1+\beta_{02}*\textrm{paO2}_1+\beta_{03}*\textrm{PaO2_FiO2}_1+\beta_{04}*\textrm{SOFA}_1+\\
#                     &I(a_1=1)*\{\beta_{10}+\beta_{11}*\textrm{Glucose}_1+\beta_{12}*\textrm{paO2}_1+\beta_{13}*\textrm{PaO2_FiO2}_1+\beta_{14}*\textrm{SOFA}_1\},\\
# Q_2(s,a_2,\boldsymbol{\mu}) = &\mu_{00}+\mu_{01}*\textrm{Glucose}_1+\mu_{02}*\textrm{paO2}_1+\mu_{03}*\textrm{PaO2_FiO2}_1+\mu_{04}*\textrm{SOFA}_1+\\
#                     &\mu_{05}*\textrm{Glucose}_2+\mu_{06}*\textrm{paO2}_2+\mu_{07}*\textrm{PaO2_FiO2}_2+\mu_{08}*\textrm{SOFA}_2+\\
#                     &I(a_2=1)*\{\mu_{10}+\mu_{11}*\textrm{Glucose}_2+\mu_{12}*\textrm{paO2}_2+\mu_{13}*\textrm{PaO2_FiO2}_2+\mu_{14}*\textrm{SOFA}_2\},\\
# Q_3(s,a_3,\boldsymbol{\theta}) = &\theta_{00}+\theta_{01}*\textrm{Glucose}_1+\theta_{02}*\textrm{paO2}_1+\theta_{03}*\textrm{PaO2_FiO2}_1+\theta_{04}*\textrm{SOFA}_1+\\
#                     &\theta_{05}*\textrm{Glucose}_2+\theta_{06}*\textrm{paO2}_2+\theta_{07}*\textrm{PaO2_FiO2}_2+\theta_{08}*\textrm{SOFA}_2+\\
#                     &\theta_{09}*\textrm{Glucose}_3+\theta_{010}*\textrm{paO2}_3+\theta_{011}*\textrm{PaO2_FiO2}_3+\theta_{012}*\textrm{SOFA}_3+\\
#                     &I(a_2=1)*\{\theta_{10}+\theta_{11}*\textrm{Glucose}_3+\theta_{12}*\textrm{paO2}_3+\theta_{13}*\textrm{PaO2_FiO2}_3+\theta_{14}*\textrm{SOFA}_3\}
# \end{align}
# 
# Using the code below, we evaluated two target polices (regimes). The first one is a fixed treatement regime that applies no treatment at all stages (Policy1), with an estimated value of .9211. Another is a fixed treatment regime that applies treatment at all stages (Policy2), with an estimated value of .5645. Therefore, the treatment effect of Policy2 comparing to Policy1 is -.3565, implying that receiving IV input increase the mortality rate.

# In[3]:


DTR_data.rename(columns = {'Died_within_48H':'R',
                            'Glucose_1':'S1_1', 'Glucose_2':'S1_2','Glucose_3':'S1_3',
                            'paO2_1':'S2_1', 'paO2_2':'S2_2','paO2_3':'S2_3',
                            'PaO2_FiO2_1':'S3_1', 'PaO2_FiO2_2':'S3_2','PaO2_FiO2_3':'S3_3',
                            'SOFA_1':'S4_1', 'SOFA_2':'S4_2', 'SOFA_3':'S4_3',
                            'IV_Input_1':'A1','IV_Input_2':'A2', 'IV_Input_3':'A3'}, inplace = True)
R = DTR_data['R'] #lower the better
S = DTR_data[['S1_1','S1_2','S1_3','S2_1','S2_2','S2_3','S3_1','S3_2','S3_3','S4_1','S4_2','S4_3']]
A = DTR_data[['A1','A2', 'A3']]
# specify the model you would like to use
model_info = [{"model": "R~S1_1+S2_1+S3_1+S4_1+A1+S1_1*A1+S2_1*A1+S3_1*A1+S4_1*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "R~S1_1+S2_1+S3_1+S4_1+S1_2+S2_2+S3_2+S4_2+A2+S1_2*A2+S2_2*A2+S3_2*A2+S4_2*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "R~S1_1+S2_1+S3_1+S4_1+S1_2+S2_2+S3_2+S4_2+S1_3+S2_3+S3_3+S4_3+A3+S1_3*A3+S2_3*A3+S3_3*A3+S4_3*A3",
              'action_space':{'A3':[0,1]}}]


# In[4]:


# Evaluating the policy with no treatment
N=len(S)
regime = pd.DataFrame({'A1':[0]*N,
                      'A2':[0]*N,
                     'A3':[0]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
QLearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True, mimic3_clip = True)
QLearn.predict_value(S)


# In[5]:


# Evaluating the policy that gives IV input at both stages
N=len(S)
regime = pd.DataFrame({'A1':[1]*N,
                      'A2':[1]*N,
                     'A3':[1]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
QLearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True, mimic3_clip = True)
QLearn.predict_value(S)


# In[6]:


0.92108006173226-0.5645453516018072


# ## Policy Optimization

# Further, to find an optimal policy maximizing the expected value, we use the **Q-learning** algorithm again to do policy optimization. Using the regression model we specified above and the code in the following block, the estimated optimal policy is summarized as the following regime.
# 
# - At stage 1:
#     1. We would recommend $A=0$ (IV_Input = 0) if $-.0001*\textrm{Glucose}_1-.0018*\textrm{paO2}_1+.0011*\textrm{PaO2_FiO2}_1+.0405*\textrm{SOFA}_1<.2529$
#     2. Else, we would recommend $A=1$ (IV_Input = 1).
# - At stage 2:
#     1. We would recommend $A=0$ (IV_Input = 0) if $-.0001*\textrm{Glucose}_2+.0020*\textrm{paO2}_2+.0001*\textrm{PaO2_FiO2}_2+.0395*\textrm{SOFA}_2<.3912$
#     2. Else, we would recommend $A=1$ (IV_Input = 1).
# - At stage 3:
#     1. We would recommend $A=0$ (IV_Input = 0) if $-.0024*\textrm{Glucose}_2-.0107*\textrm{paO2}_2+.0040*\textrm{PaO2_FiO2}_2-.0416*\textrm{SOFA}_2<-.0181$
#     2. Else, we would recommend $A=1$ (IV_Input = 1).
#     
# Appling the estimated optimal regime to individuals in the observed data, we summarize the regime pattern for each patients in the following table:
# 
# | # patients | IV_Input 1 | IV_Input 2 | IV_Input 3 |
# |------------|------------|------------|------------|
# | 16         | 1          | 1          | 0          |
# | 15         | 0          | 0          | 0          |
# | 8          | 1          | 0          | 0          |
# | 8          | 0          | 1          | 0          |
# | 3          | 0          | 1          | 1          |
# | 2          | 1          | 0          | 1          |
# | 2          | 1          | 1          | 1          |
# 
# The estimated value of the estimated optimal policy is **.9997**.

# In[7]:


# initialize the learner
QLearn = QLearning.QLearning()
# train the policy
QLearn.train(S, A, R, model_info, T=3, mimic3_clip = True)
# get the summary of the fitted Q models using the following code
#print("fitted model Q0:",QLearn.fitted_model[0].summary())
#print("fitted model Q1:",QLearn.fitted_model[1].summary())
#4. recommend action
opt_d = QLearn.recommend_action(S).value_counts()
#5. get the estimated value of the optimal regime
V_hat = QLearn.predict_value(S)
print("opt_d:",opt_d)
print("opt value:",V_hat)


# In[ ]:




