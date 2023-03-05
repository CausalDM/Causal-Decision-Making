#!/usr/bin/env python
# coding: utf-8

# # Multi-Stage Mediation Analysis

# In this notebook, we conducted mediation analysis on the MIMIC III data. Two cases are considered. We first analyzed the mediation effect with 2-stages and then analyzed the mediation effect under the setting with infinite horizon. As informed by the causal structure learning, here we consider Glucose, paO2, and PaO2_FiO2 as confounders/states, IV_Input as the action, SOFA (after being processed with a 1-step lag) as the mediator. 

# In[1]:


import pandas as pd
import pickle
import numpy as np


# ## 3-Stage Longitudinal Mediation Analysis

# Under the 3-stage setting, we are interested in analyzing the treatment effect on the final outcome Died_within_48H observed at the end of the study by comparing the target treatment regime that provides IV input at all three stages and the control treatment regime that does not provide any treatment. Using the Q-learning based estimator proposed in [1], we examine the natural direct and indirect effects of the target treatment regime based on observational data. With the code in the following blocks, the estimated effect components are summarized in the following:
# 
# | NDE   | NIE  | TE    |
# |-------|------|-------|
# | -.857 | .513 | -.344 |
# 
# Specifically, when compared to no treatment, always giving IV input has a negative impact on the survival rate with an effect size of.344, among which the effect directly from actions to the final outcome is -.857 and the indirect effect of actions to the outcome passing through mediators is .513
# 

# In[2]:


import pandas as pd
file = open('mimic3_MDTR_data_dict_3stage.pickle', 'rb')
mimic3_MDTR = pickle.load(file)
MDTR_data = pd.read_csv('mimic3_MDTR_3stage.csv')
MDTR_data.head()


# In[4]:


import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM')
from causaldm.learners import Mediated_QLearning
state, action, mediator, reward = mimic3_MDTR.values()
MediatedQLearn = Mediated_QLearning.Mediated_QLearning()
N=len(state)
regime_control = pd.DataFrame({'IV_Input_1':[0]*N,'IV_Input_2':[0]*N, 'IV_Input_3':[0]*N}).set_index(state.index)
regime_target = pd.DataFrame({'IV_Input_1':[1]*N,'IV_Input_2':[1]*N, 'IV_Input_3':[1]*N}).set_index(state.index)
MediatedQLearn.train(state, action, mediator, reward, T=3, dim_state = 3, dim_mediator = 1, 
                     regime_target = regime_target, regime_control = regime_control)
NIE, NDE = MediatedQLearn.est_NDE_NIE()
NIE, NDE, NIE+NDE


# ## Mediation Analysis with Infinite Horizon

# We processed the MIMIC III data similarly to literature on reinforcement learning by setting the reward of each stage prior to the final stage to 0, and the reward of the final stage to the observed value of Died within 48H. In this section, we analyze the average treatment effect (ATE) of a target policy that provides IV input all of the time compared to a control policy that provides no IV input at all. Using the multiply-robust estimator proposed in [2], we decomposed the ATE into four components, including immediate dierct effect (IDE), immediate mediator effect (IME), delayed direct effect (DDE), and delayed mediator effect (DME), and estimated each of the effect component. The estimation results are summarized in the table below.
# 
# | IDE           | IME | DDE           | DME           | ATE           |
# |---------------|-----|---------------|---------------|---------------|
# | -.0919(.0273) | .0001(.0000)   | -.0165(.0093) | -.0026(.0083) | -.1056(.0278) |
# 
# Specifically, the ATE of the target policy is significantly negative, with an effect size of .1056. Diving deep, we find that the DME is insignificant, whereas the IDE, IME, and DDE are all statistically significant. Further, taking the effect size into account, we can conclude that the majority of the average treatment effect is directly due to the actions derived from the target treatment policy, while the part of the effect that can be attributed to the mediators is negligible.

# In[5]:


import os
os.getcwd()
os.chdir('/nas/longleaf/home/lge/CausalDM/DTR/MRL')
import numpy as np
from scipy.special import expit
from evaluator_Linear import evaluator
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import  RatioLinearLearner as RatioLearner
from qLearner_Linear import Qlearner
os.chdir('/nas/longleaf/home/lge/CausalDM/DTR/Mediation Analysis')
MRL_df = pd.read_csv('mimic3_MRL_df.csv')
MRL_df[MRL_df.icustayid==31005]


# In[9]:


file = open('mimic3_MRL_data_dict.pickle', 'rb')
mimic3_MRL = pickle.load(file)
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


# In[10]:


#Fixed hyper-parameter--no need to modify
expectation_MCMC_iter = 50
expectation_MCMC_iter_Q3 = expectation_MCMC_iter_Q_diff = 50
truncate = 50
problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
dim_state=3; dim_mediator = 1
ratio_ndim = 10
d = 2
L = 5
t_depend_target = False
t_dependent_Q = False
scaler = 'Identity'


# In[11]:


est_obj1 = evaluator(mimic3_MRL, Qlearner, PMLearner, RewardLearner, PALearner, RatioLearner,
                     problearner_parameters = problearner_parameters,
                     ratio_ndim = ratio_ndim, truncate = truncate, l2penalty = 10**(-4),
                     t_depend_target = t_depend_target,
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = dim_state, dim_mediator = dim_mediator, 
                     Q_settings = {'scaler': scaler,'product_tensor': False, 'beta': 3/7, 
                                   'include_intercept': False, 'expectation_MCMC_iter_Q3': expectation_MCMC_iter_Q3, 
                                   'expectation_MCMC_iter_Q_diff':expectation_MCMC_iter_Q_diff, 
                                   'penalty': 10**(-4),'d': d, 'min_L': L, "t_dependent_Q": t_dependent_Q},
                     expectation_MCMC_iter = expectation_MCMC_iter,
                     seed = 10)

est_obj1.estimate_DE_ME_SE()
est_value1 = est_obj1.est_DEMESE
se_value1 = est_obj1.se_DEMESE


# In[12]:


#The following are the estimations of our interest

#1. estimation used the proposed triply robust estimator
IDE_MR, IME_MR, DDE_MR, DME_MR = est_value1[:4]

ATE = est_value1[16]

#6. SE of each estimator
se_IDE_MR, se_IME_MR, se_DDE_MR, se_DME_MR = se_value1[:4]

se_ATE = se_value1[-1]


# In[13]:


IDE_MR, IME_MR, DDE_MR, DME_MR, ATE


# In[14]:


se_IDE_MR, se_IME_MR, se_DDE_MR, se_DME_MR, se_ATE


# In[15]:


IDE_MR/se_IDE_MR, IME_MR/se_IME_MR, DDE_MR/se_DDE_MR, DME_MR/se_DME_MR, ATE/se_ATE


# ## Reference
# 
# [1] Zheng, W., & van der Laan, M. (2017). Longitudinal mediation analysis with time-varying mediators and exposures, with application to survival outcomes. Journal of causal inference, 5(2).
# 
# [2] Ge, L., Wang, J., Shi, C., Wu, Z., & Song, R. (2023). A Reinforcement Learning Framework for Dynamic Mediation Analysis. arXiv preprint arXiv:2301.13348.
