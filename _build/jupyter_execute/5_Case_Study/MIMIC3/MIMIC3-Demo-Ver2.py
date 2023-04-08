#!/usr/bin/env python
# coding: utf-8

# ## Mimic3 Demo-Ver2
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


mimic3_data = pd.read_csv('mimic3_single_stage.csv')
mimic3_data.iloc[np.where(mimic3_data['Died within 48H']==-1.0)[0],5]=0 # change the discrete action to binary


# In[3]:


mimic3_data.head(6)


# In[4]:


# ----------- Estimated DAG based on NOTEARS 

mimic3_data_final = mimic3_data  

selected = ['Glucose', 'PaO2_FiO2', 'IV Input', 'SOFA', 'SOFA Post', 'Died within 48H']

sample_demo = mimic3_data_final[selected]
est_mt = notears_linear(np.array(sample_demo), lambda1=0, loss_type='l2',w_threshold=0.1)
 
# ----------- Refit Associated Matrix under LSEM 

est_mt, _ = refit(sample_demo, est_mt, selected) 


# In[5]:


# ----------- Plot Associated Estimated DAG based on NOTEARS 

plot_net(est_mt, labels_name=selected, file_name='demo_res_net')


# In[6]:


topo_list = np.array(selected)[list(nx.topological_sort(nx.DiGraph(est_mt)))].tolist()
topo_list.reverse()
print('Topological order from top to buttom:\n', topo_list)


# ## Causal Effect Learning

# According to the amount of fluid administraition throughout the entire treatment period, we plot the average IV input for each patient as below:

# In[7]:


plt.hist(mimic3_data['IV Input'])


# As we can see from the histogram above, there is a small gap when the average IV Input is around $1.5$. This gap naturally split the data into two treatment groups: "High-IV-Input" group and "Low-IV-Input" group. We are interested in whether the highe level fluid intake treatment is able to decrease the SOFA score and the death rate of patients within 48 hours of administration.
# 
# Motivated by this problem, we set the "High-IV-Input" group as the treatment group with $A=1$, and set the "Low-IV-Input" group as the control group with $A=0$. 

# In[8]:


data_CEL_selected = mimic3_data.copy()
data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']<=1.5)[0],3]=0 # change the discrete action to binary
data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']>1.5)[0],3]=1 # change the discrete action to binary

data_CEL_selected.head(6)


# In[9]:


print( "The number of patients in treatment group is ", len(np.where(mimic3_data['IV Input']>1.5)[0]), ";\n", "The number of patients in control group is ", len(np.where(mimic3_data['IV Input']<=1.5)[0]),".")


# ### Regard 'Died_Within_48H' as the outcome variable

# In[10]:


userinfo_index = np.array([1,2])
# outcome: Died within 48H (binary)
# treatment: IV Input (binary)
# Glucose, PaO2_FiO2: covariates


# In[11]:


print(np.sum(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],5] == 1))
print(np.sum(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],5] == 0))
print(np.sum(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],5] == 1))
print(np.sum(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],5] == 0))


# In[12]:


#from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#mu0 = GradientBoostingClassifier(max_depth=2)
#mu1 = GradientBoostingClassifier(max_depth=2)

mu0 = LogisticRegression()
mu1 = LogisticRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],5] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],5] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict_proba(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict_proba(data_CEL_selected.iloc[:,userinfo_index]))[:,1]


# In[13]:


HTE_T_learner


# As we can see from the estimated treatment effect of each patient, a higher volumn of fluid intake is inclined to cause negative impact on patients' health status. This may seem counterintuitive to us, which may indicates some selection bias within this small dataset. Despite so, this result also remind us to pay attention to the potentially unnecessary fluid intake that may increase the death rate of patients.

# In[14]:


np.where(mu1.predict(data_CEL_selected.iloc[:,userinfo_index])-mu0.predict(data_CEL_selected.iloc[:,userinfo_index])==1)[0]


# Although it generally might be harmful to patients to take fluids, Patient # {0, 24} is expected to be the surviver after the fluid intake.

# In[15]:


sum(HTE_T_learner)/len(data_CEL_selected)


# Overall, IV Input is expected to increase the death-within-48-hours rate of all patients by 21.39%.

# ### Regard 'SOFA' as the outcome variable

# In[16]:


userinfo_index = np.array([1,2])
# outcome: SOFA score (treated as continuous). The smaller, the better
# treatment: iv_input (binary)
# Glucose, PaO2_FiO2: covariates
data_CEL_selected.head(6)


# Similarly, we estimate the causal effect of fluid administration on the average SOFA score of patients to see if higher IV input is able to decrease the SOFA score.

# In[17]:


#from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#mu0 = LGBMRegressor(max_depth=2)
#mu1 = LGBMRegressor(max_depth=2)

mu0 = LinearRegression()
mu1 = LinearRegression()

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==0)[0],4] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['IV Input']==1)[0],4] )


# estimate the HTE by T-learner
HTE_T_learner = (mu1.predict(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict(data_CEL_selected.iloc[:,userinfo_index]))


# In[18]:


HTE_T_learner


# Although for some patients, higher volumn of fluid intake is able to decrease their overall SOFA score, most of the rest of the patients suffered some bad effects from it.

# In[19]:


sum(HTE_T_learner)/len(data_CEL_selected)


# **Conclusion**: IV Input is expected to increase the SOFA score by 0.446.

# In[ ]:




