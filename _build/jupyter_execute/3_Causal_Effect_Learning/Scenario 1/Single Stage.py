#!/usr/bin/env python
# coding: utf-8

# ## **Single Stage**
# 
# ### Real Data
# 
# **Mimic3**: https://www.kaggle.com/datasets/asjad99/mimiciii
# 
# 
# Mimic3 is a large open-access anonymized single-center database which consists of comprehensive clinical data of 61,532 critical care admissions from 2001–2012 collected at a Boston teaching hospital. Dataset consists of 47 features (including demographics, vitals, and lab test results) on a cohort of sepsis patients who meet the sepsis-3 definition criteria.
# 
# In causal effect learning, we try to estimate the treatment effect of conducting a specific intervention (e.g use of ventilator) to the patient, either given a particular patient’s characteristics and physiological information, or evaluate all patients treatment effect as a whole.
# 
# The original Mimic3 data was loaded from mimic3_sepsis_data.csv. For illustration purpose, we selected several representative features for the following analysis. 
# 
# 
# 

# #### Data Pre-processing

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression
#from causaldm.data import mimic3_sepsis_data


# In[2]:


# Get data

mimic3_data = pd.read_csv("C:/Users/Public/CausalDM/causaldm/data/mimic3_sepsis_data.csv")
mimic3_data


# In[3]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','reward']
n = 5000
mimic3_data_selected = mimic3_data[:n][selected]
mimic3_data_selected


# In[4]:


userinfo_index = np.array([0,1,2,4]) # record all the indices of patients' information
SandA = mimic3_data_selected.iloc[:, np.array([0,1,2,3,4])]

data_CEL_selected = mimic3_data_selected
data_CEL_selected.iloc[np.where(mimic3_data_selected['iv_input']!=0)[0],:] = 1
# change the discrete action to binary
data_CEL_selected


# #### Final Mimic3 Data Selected for Causal Effect Learning (CEL)
# 
# After pre-processing, we selected 4 features as the state variable in CEL, which represents the baseline information of the patients:
# *   **Glucose**:  glucose values of patients
# *   **paO2**: The partial pressure of oxygen
# *   **PaO2_FiO2**: The partial pressure of oxygen (PaO2)/fraction of oxygen delivered (FIO2) ratio.
# *   **SOFA**: Sepsis-related Organ Failure Assessment score to describe organ dysfunction/failure.
# 
# The action variable is **iv-input**, which denotes the volumn of fluids that have been administered to the patient. Additionally, we set all non-zero iv-input values as $1$ to create a binary action space.
# 
# The last column denotes the reward we evaluated according to the status of patients from several aspects.
# 
# 

# In[ ]:




