#!/usr/bin/env python
# coding: utf-8

# ## **Single Stage -- Paradigm 1**
# 
# ### Real Data 1. Movie Lens
# 
# Movie Lens is a movie recommendation website that helps users to find movies and collect their ratings. The goal of the simulation studies in single stage causal effect learning is to infer on the causal effect of treating users 'Drama', versus the control movie genere 'Sci-Fi'. This serves as an offline evaluation of how well people like/dislike a specific movie genere versus the other, and hence provides us a general scope of which movie genere to recommend so as to maximize users' satisfaction.
# 

# #### Data Pre-processing

# In[1]:


# import related packages
import os
os.getcwd()
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
import pickle
import numpy as np
import causaldm.learners.Online.CMAB._env_realCMAB as env

data = env.get_movielens()


# In[5]:


data.keys()


# In[6]:


data_ML = data['Individual']


# In[7]:


userinfo_index = np.array([3,9,11,12,13,14])

users_index = data_ML.keys()
n = len(users_index) # the number of users
movie_generes = ['Comedy', 'Drama', 'Action', 'Thriller', 'Sci-Fi']

data_CEL = {}
 
# initialize the final data we'll use in Causal Effect Learning
for i in movie_generes:
    data_CEL[i] = None   

import pandas as pd
for movie_genere in movie_generes:
      for user in users_index:
            data_CEL[movie_genere] = pd.concat([data_CEL[movie_genere] , data_ML[user][movie_genere]['complete']])


# In[8]:


data_CEL['Comedy']


# In[13]:


data_CEL_all = pd.concat([data_CEL['Drama'], data_CEL['Sci-Fi']]) 
data_CEL_all = data_CEL_all.drop(columns=['Comedy', 'Action', 'Thriller', 'Sci-Fi'])
#data_CEL_all.to_csv("/Users/alinaxu/Documents/CDM/CausalDM/causaldm/data/MovieLens_CEL.csv")
data_CEL_all


# In[ ]:





# #### Final Movie Lens Data Selected for Causal Effect Learning (CEL)
# 
# After pre-processing, the complete data contains 65,642 movie watching history of 175 individuals. We set treatment $A=1$ when the user choose a 'Drama', and $A=0$ if the movie belongs to 'Sci-Fi'. 
# 
# The processed data is saved in 'causaldm/data/MovieLens_CEL.csv' and will be directly used in later subsections.

# ### Real Data 2. Mimic3
# https://www.kaggle.com/datasets/asjad99/mimiciii
# 
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

# In[22]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.linear_model import LinearRegression
#from causaldm.data import mimic3_sepsis_data


# In[25]:


# Get data

mimic3_data = pd.read_csv("/Users/alinaxu/Documents/CDM/CausalDM/causaldm/data/mimic3_sepsis_data.csv")
mimic3_data.head(6)


# In[26]:


selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','reward']
n = 5000
mimic3_data_selected = mimic3_data[:n][selected]
mimic3_data_selected


# In[27]:


userinfo_index = np.array([0,1,2,4]) # record all the indices of patients' information
SandA = mimic3_data_selected.iloc[:, np.array([0,1,2,3,4])]

data_CEL_selected = mimic3_data_selected
data_CEL_selected.iloc[np.where(mimic3_data_selected['iv_input']!=0)[0],:] = 1
# change the discrete action to binary
data_CEL_selected.head(6)


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




