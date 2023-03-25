#!/usr/bin/env python
# coding: utf-8

# ## MIMIC III (Single-Stage)
# 
# In this notebook, we conducted analysis on the MIMIC III data with a single stage. We first analyzed the mediation effect and then evaluate the policy of interest and calculated the optimal policy. As informed by the causal structure learning, here we consider Glucose and PaO2_FiO2 as confounders/states, IV_Input as the action, SOFA as the mediator. 

# In[1]:


import pandas as pd
import numpy as np
single_data = pd.read_csv('mimic3_single_stage.csv')
single_data.iloc[np.where(single_data['IV Input']<=1.5)[0],3]=0 # change the discrete action to binary
single_data.iloc[np.where(single_data['IV Input']>1.5)[0],3]=1 # change the discrete action to binary
single_data.head(6)


# In[2]:


state = np.array(single_data[['Glucose','PaO2_FiO2']])
action = np.array(single_data[['IV Input']])
mediator = np.array(single_data[['SOFA']])
reward = np.array(single_data[['Died within 48H']])
single_dataset = {'state':state,'action':action,'mediator':mediator,'reward':reward}


# # CEL: Single-Stage Mediation Analysis

# Under the single-stage setting, we are interested in analyzing the treatment effect on the final outcome Died_within_48H observed at the end of the study by comparing the target treatment regime that provides IV input for all patients and the control treatment regime that does not provide any treatment. Using the direct estimator proposed in [1], IPW estimator proposed in [2], and robust estimator proposed in [3], we examine the natural direct and indirect effects of the target treatment regime based on observational data. With the code in the following blocks, the estimated effect components are summarized in the following:
# 
# |                  |   DE   | IE     | TE     |
# |------------------|:------:|--------|--------|
# | Direct Estimator | -.3465 |  .0097 | -.3368 |
# | IPW              | -.4585 | 0      | -.4584 |
# | Robust           | -.3643 | -.0482 | -.4126 |
# 
# Specifically, when compared to no treatment, always giving IV input has a negative impact on the survival rate, among which the effect directly from actions to the final outcome dominates.

# In[3]:


import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM/DTR/MRL')
from scipy.special import expit
from evaluator_baseline import evaluator
from probLearner import PMLearner, RewardLearner, PALearner


# In[4]:


# Control Policy
def control_policy(state = None, dim_state=None, action=None, get_a = False):
    # fixed policy with fixed action 0
    if get_a:
        action_value = np.array([0])
    else:
        state = np.copy(state).reshape(-1,dim_state)
        NT = state.shape[0]
        if action is None:
            action_value = np.array([0]*NT)
        else:
            action = np.copy(action).flatten()
            if len(action) == 1 and NT>1:
                action = action * np.ones(NT)
            action_value = 1-action
    return action_value
def target_policy(state, dim_state = 1, action=None):
    state = np.copy(state).reshape((-1, dim_state))
    NT = state.shape[0]
    pa = 1 * np.ones(NT)
    if action is None:
        if NT == 1:
            pa = pa[0]
            prob_arr = np.array([1-pa, pa])
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            raise ValueError('No random for matrix input')
    else:
        action = np.copy(action).flatten()
        action_value = pa * action + (1-pa) * (1-action)
    return action_value


# In[5]:


problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
est_obj1 = evaluator(single_dataset, PMLearner, RewardLearner, PALearner,
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 2, dim_mediator = 1, 
                     expectation_MCMC_iter = 50,
                     seed = 10)

est_obj1.estimate_DE_ME_SE()
est_value1 = est_obj1.est_DEMESE


# In[18]:


est_value1


# ## CPL: Single-Stage Policy Evaluation

# In[11]:


os.chdir('/nas/longleaf/home/lge/CausalDM')
from causaldm.learners import QLearning


# As an example, we use the **Q-learning** algorithm to evaluate policies based on the observed data, with the linear regression models defined as the following:
# \begin{align}
# Q(s,a,\boldsymbol{\beta}) = &\beta_{00}+\beta_{01}*\textrm{Glucose}+\beta_{02}*\textrm{PaO2_FiO2}\\
#                     &I(a_1=1)*\{\beta_{10}+\beta_{11}*\textrm{Glucose}+\beta_{12}*\textrm{PaO2_FiO2}\},
# \end{align}
# 
# Using the code below, we evaluated two target polices (regimes). The first one is a fixed treatement regime that applies no treatment (Policy1), with an estimated value of .9050. Another is a fixed treatment regime that applies treatment all the time (Policy2), with an estimated value of .5085. Therefore, the treatment effect of Policy2 comparing to Policy1 is -.3965, implying that receiving IV input increase the mortality rate.

# In[12]:


single_data.rename(columns = {'Died within 48H':'R', 'Glucose':'S1', 'PaO2_FiO2':'S2', 'IV Input':'A'}, inplace = True)
R = single_data['R'] #lower the better
S = single_data[['S1','S2']]
A = single_data[['A']]
# specify the model you would like to use
model_info = [{"model": "R~S1+S2+A+S1*A+S2*A",
              'action_space':{'A':[0,1]}}]


# In[13]:


# Evaluating the policy with no treatment
N=len(S)
regime = pd.DataFrame({'A':[0]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
QLearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True, mimic3_clip = True)
QLearn.predict_value(S)


# In[14]:


# Evaluating the policy that gives IV input at both stages
N=len(S)
regime = pd.DataFrame({'A':[1]*N}).set_index(S.index)
#evaluate the regime
QLearn = QLearning.QLearning()
QLearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True, mimic3_clip = True)
QLearn.predict_value(S)


# ## CPL: Single-Stage Policy Optimization

# Further, to find an optimal policy maximizing the expected value, we use the **Q-learning** algorithm again to do policy optimization. Using the regression model we specified above and the code in the following block, the estimated optimal policy is summarized as the following regime.
# 
# 1. We would recommend $A=0$ (IV_Input = 0) if $.0014*\textrm{Glucose}+.0015*\textrm{PaO2_FiO2}<1.106$
# 2. Else, we would recommend $A=1$ (IV_Input = 1).
#     
# Appling the estimated optimal regime to individuals in the observed data, we summarize the regime pattern for each patients in the following table:
# 
# | # patients | IV_Input | 
# |------------|----------|
# | 56         | 0        |
# | 1          | 1        |
# The estimated value of the estimated optimal policy is **.9050**.

# In[16]:


# initialize the learner
QLearn = QLearning.QLearning()
# train the policy
QLearn.train(S, A, R, model_info, T=1, mimic3_clip = True)
# get the summary of the fitted Q models using the following code
#print("fitted model Q0:",QLearn.fitted_model[0].summary())
#print("fitted model Q1:",QLearn.fitted_model[1].summary())
#4. recommend action
opt_d = QLearn.recommend_action(S).value_counts()
#5. get the estimated value of the optimal regime
V_hat = QLearn.predict_value(S)
print("opt_d:",opt_d)
print("opt value:",V_hat)


# ## Reference
# 
# [1]Robins, J. M. and Greenland, S. Identifiability and exchangeability for direct and indirect effects. Epidemiology, pp. 143–155, 1992.
# 
# [2]Hong, G. (2010). Ratio of mediator probability weighting for estimating natural direct and indirect effects. In Proceedings of the American Statistical Association, biometrics section (pp. 2401-2415).
# 
# [3] Tchetgen, E. J. T., & Shpitser, I. (2012). Semiparametric theory for causal mediation analysis: efficiency bounds, multiple robustness, and sensitivity analysis. Annals of statistics, 40(3), 1816.

# In[ ]:




