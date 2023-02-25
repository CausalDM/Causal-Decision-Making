#!/usr/bin/env python
# coding: utf-8

# ## Mimic3 Demo
# 
# [Mimic3](https://www.kaggle.com/datasets/asjad99/mimiciii) is a large open-access anonymized single-center database which consists of comprehensive clinical data of 61,532 critical care admissions from 2001–2012 collected at a Boston teaching hospital. Dataset consists of 47 features (including demographics, vitals, and lab test results) on a cohort of sepsis patients who meet the sepsis-3 definition criteria.
# 
# Due to the privacy concerns, we utilized a subset of he original Mimic3 data that is publicly available on Kaggle. For illustration purpose, we selected several representative features for the following analysis:
# 
# *   **Glucose**: glucose values of patients
# *   **paO2**: The partial pressure of oxygen
# *   **PaO2_FiO2**: The partial pressure of oxygen (PaO2)/fraction of oxygen delivered (FIO2) ratio.
# *   **SOFA**: Sepsis-related Organ Failure Assessment score to describe organ dysfunction/failure.
# *   **iv-input**: the volumn of fluids that have been administered to the patient.
# *   **died_within_48h_of_out_time**:  the mortality status of the patient after 48 hours of being administered.
# 
# In the next sections, we will start from causal discovery learning to learn significant causal diagram from the data, and then quantify the effect of treatment ('iv_input') on the outcome (mortality status, denoted by 'died_within_48h_of_out_time' variable in the data) through causal effect learning.

# ## Causal Discovery Learning

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import os
import pickle
import random
import math
import time 

from datetime import datetime
import matplotlib.pyplot as plt

from multiprocessing import Pool
 
from tqdm import tqdm
from functools import partial 

os.environ["OMP_NUM_THREADS"] = "1"


# In[2]:


mimic3_data = pd.read_csv("C:/Users/Public/Causal-Decision-Making/5_Case_Study/MIMIC3/subset_mimic3_sepsis_data.csv")
mimic3_data.head(6)


# In[ ]:





# In[3]:


##### Import Packages 
from causaldm.learners.Causal_Discovery_Learning.utils import *
from causaldm.learners.Case_Study.MIMIC3 import *
from numpy.random import randn
from random import seed as rseed
from numpy.random import seed as npseed


# In[ ]:


mimic3_data.columns


# In[ ]:


# ----------- Estimated DAG based on NOTEARS
#from obspy.imaging.beachball import plot_mt
mimic3_data_final = mimic3_data
# selected = ['gender',  
#        're_admission', 'died_within_48h_of_out_time', 
#        'Weight_kg', 'GCS',  'SpO2',
#        'Temp_C', 'FiO2_1',  'Chloride', 'Glucose', 'BUN', 'WBC_count', 'paO2', 'paCO2', 
#             'PaO2_FiO2',
#        'median_dose_vaso', 'max_dose_vaso', 'SOFA', 'SIRS',
#        'vaso_input', 'iv_input', 'reward']
# selected = ['gender',   
#        'Weight_kg', 'GCS', # 'SpO2', 'FiO2_1',  'Chloride', 'paO2', 'paCO2',  'PaO2_FiO2',
#        #'median_dose_vaso', 'max_dose_vaso',  
#             'cumulated_balance', 
#        'vaso_input', 'iv_input', 'SOFA']

selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','died_within_48h_of_out_time']

sample_demo = mimic3_data_final[:5000][selected]
est_mt = notears_linear(np.array(sample_demo), lambda1=0, loss_type='l2',w_threshold=0.1)

# ----------- Plot Associated Matrix for the Estimated DAG based on NOTEARS

# calculate_effect(est_mt)


# In[ ]:


plot_mt(est_mt, labels_name=selected, file_name='demo_res_mt')


# In[ ]:


est_mt[3,4] # SOFA -> iv_input 


# In[ ]:


est_mt[5,3] # iv_input -> died_within_48h_of_out_time


# In[ ]:


plot_net(est_mt, labels_name=selected, file_name='demo_res_net')



# In[ ]:





# In[ ]:


sum(mimic3_data['cumulated_balance'])/len(mimic3_data)


# In[ ]:


sum(mimic3_data['output_total'])/len(mimic3_data)


# In[ ]:


sum(mimic3_data['input_total'])/len(mimic3_data)


# In[ ]:





# ## Causal Effect Learning

# In[ ]:


# in_input is the treatment, SOFA is the mediator, and the died_within_48hour is the outcome. 
# All the rest nodes can be viewed as the confounders.


# In[ ]:


mimic3_data.columns


# In[ ]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','died_within_48h_of_out_time']

sample_demo = mimic3_data[selected]


# In[ ]:


userinfo_index = np.array([0,1,2,4])
# outcome: died_within_48h_of_out_time (binary)
# treatment: iv_input (binary)
# others: covariates


# In[ ]:


sample_demo.iloc[np.where(sample_demo['iv_input']!=0)[0],3]=1 # change the discrete action to binary
data_CEL_selected = sample_demo.copy()
data_CEL_selected.head(6)


# In[ ]:


print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5]==0))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5]==1))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5]==0))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5]==1))
# 58 patients in total


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#mu0 = GradientBoostingClassifier(max_depth=2)
#mu1 = GradientBoostingClassifier(max_depth=2)

mu0 = LogisticRegression()
mu1 = LogisticRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict_proba(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict_proba(data_CEL_selected.iloc[:,userinfo_index]))[:,1]


# In[ ]:


HTE_T_learner


# In[ ]:


mu1.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


mu0.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


np.where(mu1.predict(data_CEL_selected.iloc[:,userinfo_index])-mu0.predict(data_CEL_selected.iloc[:,userinfo_index])==1)[0]


# In[ ]:


sum(HTE_T_learner)/len(data_CEL_selected)


# **Conclusion**: iv-input is expected to improve the death-within-48-hours rate by 13.18%.

# ### exclude SOFA from the covariates list

# In[ ]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input','died_within_48h_of_out_time']

sample_demo = mimic3_data[selected]


# In[ ]:


userinfo_index = np.array([0,1,2])
# outcome: died_within_48h_of_out_time (binary)
# treatment: iv_input (binary)
# others: covariates


# In[ ]:


sample_demo.iloc[np.where(sample_demo['iv_input']!=0)[0],3]=1 # change the discrete action to binary
data_CEL_selected = sample_demo.copy()
data_CEL_selected.head(6)


# In[ ]:


print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],4]==0))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],4]==1))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],4]==0))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],4]==1))
# 58 patients in total


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#mu0 = GradientBoostingClassifier(max_depth=2)
#mu1 = GradientBoostingClassifier(max_depth=2)

mu0 = LogisticRegression()
mu1 = LogisticRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],4] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],4] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict_proba(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict_proba(data_CEL_selected.iloc[:,userinfo_index]))[:,1]


# In[ ]:


HTE_T_learner


# In[ ]:


mu1.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


mu0.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


np.where(mu1.predict(data_CEL_selected.iloc[:,userinfo_index])-mu0.predict(data_CEL_selected.iloc[:,userinfo_index])==1)[0]


# In[ ]:


sum(HTE_T_learner)/len(data_CEL_selected)


# **Conclusion**: iv-input is expected to improve the death-within-48-hours rate by 13.18%.

# In[ ]:





# ### Regard SOFA as outcome variable

# In[ ]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA']

sample_demo = mimic3_data[selected]


# In[ ]:


userinfo_index = np.array([0,1,2])
# outcome: SOFA score (treated as continuous). The smaller, the better
# treatment: iv_input (binary)
# others: covariates


# In[ ]:


sample_demo.iloc[np.where(sample_demo['iv_input']!=0)[0],3]=1 # change the discrete action to binary
data_CEL_selected = sample_demo.copy()
data_CEL_selected.head(6)


# In[ ]:


print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],4]))
print(sum(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],4]))
# 58 patients in total
# the trend looks great: iv_input helps to decrease the overall SOFA score of patients


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#mu0 = LGBMRegressor(max_depth=2)
#mu1 = LGBMRegressor(max_depth=2)

mu0 = LinearRegression()
mu1 = LinearRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],4] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],4] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict(data_CEL_selected.iloc[:,userinfo_index]))


# In[ ]:


mu1.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


mu0.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


HTE_T_learner


# In[ ]:


sum(HTE_T_learner)/len(data_CEL_selected)


# **Conclusion**: iv-input is expected to decrease the SOFA score by 0.958.

# In[ ]:





# ### 2023.02.11 change to another outcome variable

# In[ ]:


# set the reward variable as the original one in the data
mimic3_data


# In[ ]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','reward']

smaple_demo = mimic3_data_final[:5000][selected]


# In[ ]:


userinfo_index = np.array([0,1,2,4])
smaple_demo
# outcome: died_within_48h_of_out_time (binary)
# treatment: iv_input (binary)
# others: covariates


# In[ ]:


len(np.where(smaple_demo['iv_input']==0)[0])


# In[ ]:


data_CEL_selected = smaple_demo
data_CEL_selected.iloc[np.where(smaple_demo['iv_input']!=0)[0],:] = 1
# change the discrete action to binary
data_CEL_selected


# In[ ]:


len(np.where(data_CEL_selected['iv_input']==1)[0])


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#mu0 = GradientBoostingClassifier(max_depth=2)
#mu1 = GradientBoostingClassifier(max_depth=2)

#mu0 = LogisticRegression()
#mu1 = LogisticRegression()

#mu0 = LGBMRegressor(max_depth=3)
#mu1 = LGBMRegressor(max_depth=3)

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict(data_CEL_selected.iloc[:,userinfo_index])


# In[ ]:


HTE_T_learner


# In[ ]:


sum(HTE_T_learner)/5000
# Averaged Treatment Effect


# In[ ]:


#  concern 1: S-learner failed to learn HTE since the scale of iv_input is way smaller than other state variables

