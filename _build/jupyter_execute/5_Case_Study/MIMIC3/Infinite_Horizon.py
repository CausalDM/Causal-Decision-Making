#!/usr/bin/env python
# coding: utf-8

# # MIMIC III (Infinite Horizon)
# 
# In this notebook, we conducted analysis on the MIMIC III data with infinite horizon. We first analyzed the mediation effect and then evaluate the policy of interest and calculated the optimal policy. As informed by the causal structure learning, here we consider Glucose and PaO2_FiO2 as confounders/states, IV_Input as the action, SOFA as the mediator. 

# In[1]:


import os
os.chdir('D:/Github/CausalDM')
import pandas as pd
import numpy as np
import pandas as pd
import pickle
file = open('./causaldm/MIMIC3/mimic3_MRL_data_dict_V2.pickle', 'rb')
mimic3_MRL = pickle.load(file)
mimic3_MRL['reward'] = [1 if r == 0 else r for r in mimic3_MRL['reward']]
mimic3_MRL['reward'] = [0 if r == -1 else r for r in mimic3_MRL['reward']]
MRL_df = pd.read_csv('./causaldm/MIMIC3/mimic3_MRL_df_V2.csv')
MRL_df.iloc[np.where(MRL_df['Died_within_48H']==0)[0],-1]=1
MRL_df.iloc[np.where(MRL_df['Died_within_48H']==-1)[0],-1]=0
MRL_df[MRL_df.icustayid==1006]


# ## CEL: Mediation Analysis with Infinite Horizon

# We processed the MIMIC III data similarly to literature on reinforcement learning by setting the reward of each stage prior to the final stage to 0, and the reward of the final stage to the observed value of Died within 48H. In this section, we analyze the average treatment effect (ATE) of a target policy that provides IV input all of the time compared to a control policy that provides no IV input at all. Using the multiply-robust estimator proposed in [1], we decomposed the ATE into four components, including immediate nature dierct effect (INDE), Immediate nature mediator effect (INME), delayed nature direct effect (DNDE), and delayed nature mediator effect (NDDNME), and estimated each of the effect component. The estimation results are summarized in the table below.
# 
# | INDE           | INME | DNDE           | DNME           | ATE           |
# |---------------|-----|---------------|---------------|---------------|
# | -.0181(.0059) | .0061(.0021)   | -.0049(.0028) | -.0002(.0011) | -.0171(.0058) |
# 
# Specifically, the ATE of the target policy is significantly negative, with an effect size of .0171. Diving deep, we find that the DNME and DNDE are insignificant, whereas the INDE and INME are all statistically significant. Further, taking the effect size into account, we can conclude that the majority of the average treatment effect is directly due to the actions derived from the target treatment policy, while the part of the effect that can be attributed to the mediators is negligible.

# In[2]:


from causaldm.learners.Causal_Effect_Learning.Mediation_Analysis import ME_MDP


# In[11]:


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


# In[12]:


#Fixed hyper-parameter--no need to modify
expectation_MCMC_iter = 50
expectation_MCMC_iter_Q3 = expectation_MCMC_iter_Q_diff = 50
truncate = 50
problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
dim_state=2; dim_mediator = 1
ratio_ndim = 10
d = 2
L = 5
scaler = 'Identity'


# In[21]:


Robust_est = ME_MDP.evaluator(mimic3_MRL, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     ratio_ndim = ratio_ndim, truncate = truncate, l2penalty = 10**(-4),
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = dim_state, dim_mediator = dim_mediator, 
                     Q_settings = {'scaler': scaler,'product_tensor': False, 'beta': 3/7, 
                                   'include_intercept': False, 'expectation_MCMC_iter_Q3': expectation_MCMC_iter_Q3, 
                                   'expectation_MCMC_iter_Q_diff':expectation_MCMC_iter_Q_diff, 
                                   'penalty': 10**(-4),'d': d, 'min_L': L, 't_dependent_Q': False},
                     expectation_MCMC_iter = expectation_MCMC_iter,
                     seed = 0, nature_decomp = True, method = 'Robust')

Robust_est.estimate_DE_ME()
Robust_est.est_IDE, Robust_est.IME, Robust_est.DDE, Robust_est.DME, Robust_est.TE


# In[22]:


Robust_est.IDE_se, Robust_est.IME_se, Robust_est.DDE_se, Robust_est.DME_se, Robust_est.TE_se


# ## Reference
# 
# [1] Ge, L., Wang, J., Shi, C., Wu, Z., & Song, R. (2023). A Reinforcement Learning Framework for Dynamic Mediation Analysis. arXiv preprint arXiv:2301.13348.

# In[ ]:




