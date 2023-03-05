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

##### Import Packages 
from utils import *
from notear import *
  
from numpy.random import randn
from random import seed as rseed
from numpy.random import seed as npseed

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


mimic3 = pd.read_csv("subset_rl_data_final_cont.csv")


# In[3]:


mimic3_base = mimic3[['icustayid', 'Glucose','paO2','PaO2_FiO2',
                           'iv_input', 'SOFA','died_within_48h_of_out_time']]
mimic3_base['died_within_48h_of_out_time'] = - 2 * np.array(mimic3_base['died_within_48h_of_out_time']) + 1
mimic3_base.columns = ['icustayid', 'Glucose','paO2','PaO2_FiO2',
                           'IV Input', 'SOFA','Died within 48H'] 
mimic3_base.head(6)


# In[4]:


plt.hist(mimic3_base['SOFA'])


# In[5]:


mimic_final = mimic3_base[mimic3_base['SOFA']<=12]
plt.hist(mimic_final['SOFA'])


# In[6]:


with open('mimic3_multi_stages.pickle', 'wb') as handle:
    pickle.dump(mimic_final, handle)
    
mimic_final.to_csv (r'mimic3_multi_stages.csv', index = False, header=True)

mimic_final


# In[7]:


# ----------- Set lag data
lag_k = 1
    
#     new_sofa = list(np.array(mimic_final['SOFA'][lag_k:]) - np.array(mimic_final['SOFA'][:-lag_k]))

new_sofa = np.array(mimic_final['SOFA'][:-lag_k])
mimic3_sample = mimic_final.iloc[lag_k:]
mimic3_sample['SOFA'] = new_sofa
mimic3_data = mimic3_sample.groupby('icustayid').mean().reset_index() 
 


# In[8]:


with open('mimic3_single_stage.pickle', 'wb') as handle:
    pickle.dump(mimic3_data, handle)
    
mimic3_data.to_csv (r'mimic3_single_stage.csv', index = False, header=True)
 
mimic3_data.head(6)


# In[9]:


# ----------- Estimated DAG based on NOTEARS

mimic3_data_final = mimic3_data  

selected = ['Glucose','paO2','PaO2_FiO2', 'IV Input', 'SOFA','Died within 48H']

smaple_demo = mimic3_data_final[selected]
est_mt = notears_linear(np.array(smaple_demo), lambda1=0, loss_type='l2',w_threshold=0.1)

# ----------- Plot Associated Matrix for the Estimated DAG based on NOTEARS

plot_mt(est_mt, labels_name=selected, file_name='demo_res_mt')

# calculate_effect(est_mt)


# In[10]:


plot_net(est_mt, labels_name=selected, file_name='demo_res_net')


# In[11]:


calculate_effect(est_mt)


# ## Causal Effect Learning

# In[12]:


mimic3_data.columns


# In[13]:


mimic3_data.head(6)


# According to the amount of fluid administraition throughout the entire treatment period, we plot the average IV input for each patient as below:

# In[14]:


plt.hist(mimic3_data['IV Input'])


# As we can see from the histogram above, there is a small gap when the average IV Input is around $1.5$. This gap naturally split the data into two treatment groups: "High-IV-Input" group and "Low-IV-Input" group. We are interested in whether the highe level fluid intake treatment is able to decrease the SOFA score and the death rate of patients within 48 hours of administration.
# 
# Motivated by this problem, we set the "High-IV-Input" group as the treatment group with $A=1$, and set the "Low-IV-Input" group as the control group with $A=0$. 

# In[15]:


data_CEL_selected = mimic3_data.copy()
data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']<=1.5)[0],4]=0 # change the discrete action to binary
data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']>1.5)[0],4]=1 # change the discrete action to binary

data_CEL_selected.head(6)


# In[16]:


print( "The number of patients in treatment group is ", len(np.where(mimic3_data['IV Input']>1.5)[0]), ";\n", "The number of patients in control group is ", len(np.where(mimic3_data['IV Input']<=1.5)[0]),".")


# ### Regard 'Died_Within_48H' as the outcome variable

# In[17]:


userinfo_index = np.array([1,2,3])
# outcome: Died within 48H (binary)
# treatment: IV Input (binary)
# Glucose, paO2, PaO2_FiO2: covariates


# In[18]:


#from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#mu0 = GradientBoostingClassifier(max_depth=2)
#mu1 = GradientBoostingClassifier(max_depth=2)

mu0 = LogisticRegression()
mu1 = LogisticRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],6] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],6] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict_proba(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict_proba(data_CEL_selected.iloc[:,userinfo_index]))[:,1]


# In[19]:


HTE_T_learner


# As we can see from the estimated treatment effect of each patient, a higher volumn of fluid intake is inclined to cause negative impact on patients' health status. This may seem counterintuitive to us, which may indicates some selection bias within this small dataset. Despite so, this result also remind us to pay attention to the potentially unnecessary fluid intake that may increase the death rate of patients.

# In[20]:


np.where(mu1.predict(data_CEL_selected.iloc[:,userinfo_index])-mu0.predict(data_CEL_selected.iloc[:,userinfo_index])==-2)[0]


# Patient # {0, 32, 48, 52, 55} might be the sufferer of the over intake of fluid.

# In[21]:


sum(HTE_T_learner)/len(data_CEL_selected)


# Overall, IV Input is expected to increase the death-within-48-hours rate of all patients by 20.40%.

# ### Regard 'SOFA' as the outcome variable

# In[22]:


userinfo_index = np.array([1,2,3])
# outcome: SOFA score (treated as continuous). The smaller, the better
# treatment: iv_input (binary)
# Glucose, paO2, PaO2_FiO2: covariates
data_CEL_selected.head(6)


# Similarly, we estimate the causal effect of fluid administration on the average SOFA score of patients to see if higher IV input is able to decrease the SOFA score.

# In[23]:


#from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#mu0 = LGBMRegressor(max_depth=2)
#mu1 = LGBMRegressor(max_depth=2)

mu0 = LinearRegression()
mu1 = LinearRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],5] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],5] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict(data_CEL_selected.iloc[:,userinfo_index]))


# In[24]:


HTE_T_learner


# Although for some patients, higher volumn of fluid intake is able to decrease their overall SOFA score, most of the rest of the patients suffered some bad effects from it.

# In[25]:


sum(HTE_T_learner)/len(data_CEL_selected)


# **Conclusion**: IV Input is expected to increase the SOFA score by 0.086.

# In[ ]:




