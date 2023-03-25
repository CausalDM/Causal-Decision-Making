#!/usr/bin/env python
# coding: utf-8

# # MIMIC III (Infinite Horizon)
# 
# In this notebook, we conducted analysis on the MIMIC III data with infinite horizon. We first analyzed the mediation effect and then evaluate the policy of interest and calculated the optimal policy. As informed by the causal structure learning, here we consider Glucose and PaO2_FiO2 as confounders/states, IV_Input as the action, SOFA as the mediator. 

# ## CEL: Mediation Analysis with Infinite Horizon

# We processed the MIMIC III data similarly to literature on reinforcement learning by setting the reward of each stage prior to the final stage to 0, and the reward of the final stage to the observed value of Died within 48H. In this section, we analyze the average treatment effect (ATE) of a target policy that provides IV input all of the time compared to a control policy that provides no IV input at all. Using the multiply-robust estimator proposed in [1], we decomposed the ATE into four components, including immediate dierct effect (IDE), immediate mediator effect (IME), delayed direct effect (DDE), and delayed mediator effect (DME), and estimated each of the effect component. The estimation results are summarized in the table below.
# 
# | IDE           | IME | DDE           | DME           | ATE           |
# |---------------|-----|---------------|---------------|---------------|
# | -.1137(.0273) | -.0187(.0067)   | -.0247(.0153) | .0182(.0119) | -.1390(.0297) |
# 
# Specifically, the ATE of the target policy is significantly negative, with an effect size of .1390. Diving deep, we find that the DME and DDE are insignificant, whereas the IDE and IME are all statistically significant. Further, taking the effect size into account, we can conclude that the majority of the average treatment effect is directly due to the actions derived from the target treatment policy, while the part of the effect that can be attributed to the mediators is negligible.

# In[1]:


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
MRL_df = pd.read_csv('mimic3_MRL_df_V2.csv')
MRL_df[MRL_df.icustayid==1006]


# In[12]:


file = open('mimic3_MRL_data_dict_V2.pickle', 'rb')
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


# In[13]:


#Fixed hyper-parameter--no need to modify
expectation_MCMC_iter = 50
expectation_MCMC_iter_Q3 = expectation_MCMC_iter_Q_diff = 50
truncate = 50
problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
dim_state=2; dim_mediator = 1
ratio_ndim = 10
d = 2
L = 5
t_depend_target = False
t_dependent_Q = False
scaler = 'Identity'


# In[14]:


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


# In[15]:


#The following are the estimations of our interest

#1. estimation used the proposed triply robust estimator
IDE_MR, IME_MR, DDE_MR, DME_MR = est_value1[:4]

ATE = est_value1[16]

#6. SE of each estimator
se_IDE_MR, se_IME_MR, se_DDE_MR, se_DME_MR = se_value1[:4]

se_ATE = se_value1[-1]


# In[16]:


IDE_MR, IME_MR, DDE_MR, DME_MR, ATE


# In[17]:


se_IDE_MR, se_IME_MR, se_DDE_MR, se_DME_MR, se_ATE


# In[18]:


IDE_MR/se_IDE_MR, IME_MR/se_IME_MR, DDE_MR/se_DDE_MR, DME_MR/se_DME_MR, ATE/se_ATE


# ## Reference
# 
# [1] Ge, L., Wang, J., Shi, C., Wu, Z., & Song, R. (2023). A Reinforcement Learning Framework for Dynamic Mediation Analysis. arXiv preprint arXiv:2301.13348.

# In[ ]:




