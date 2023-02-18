#!/usr/bin/env python
# coding: utf-8

# ### **1. S-learner**
# 
# 
# The first estimator we would like to introduce is the S-learner, also known as a ``single learner". This is one of the most foundamental learners in HTE esitmation, and is very easy to implement.
# 
# Under three common assumptions in causal inference, i.e. (1) consistency, (2) no unmeasured confounders (NUC), (3) positivity assumption, the heterogeneous treatment effect can be identified by the observed data, where
# \begin{equation*}
# \tau(s)=\mathbb{E}[R|S,A=1]-\mathbb{E}[R|S,A=0].
# \end{equation*}
# 
# The basic idea of S-learner is to fit a model for $\mathbb{E}[R|S,A]$, and then construct a plug-in estimator based on the expression above. Specifically, the algorithm can be summarized as below:
# 
# **Step 1:**  Estimate the combined response function $\mu(s,a):=\mathbb{E}[R|S=s,A=a]$ with any regression algorithm or supervised machine learning methods;
# 
# **Step 2:**  Estimate HTE by 
# \begin{equation*}
# \hat{\tau}_{\text{S-learner}}(s)=\hat\mu(s,1)-\hat\mu(s,0).
# \end{equation*}
# 
# 
# 

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression


# ### Mimic3 Data

# In[2]:


# Get data
n = 5000
selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','reward']
data_CEL_selected = pd.read_csv("C:/Users/Public/CausalDM/causaldm/data/mimic3_CEL_selected.csv")
data_CEL_selected.pop(data_CEL_selected.columns[0])
data_CEL_selected


# In[3]:


userinfo_index = np.array([0,1,2,4])
SandA = data_CEL_selected.iloc[:, np.array([0,1,2,3,4])]


# In[4]:


# S-learner
S_learner = LGBMRegressor(max_depth=5)
#S_learner = LinearRegression()
#SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, data_CEL_selected['reward'])


# In[5]:


SandA_all1 = SandA.copy()
SandA_all0 = SandA.copy()
SandA_all1.iloc[:,3]=np.ones(n)
SandA_all0.iloc[:,3]=np.zeros(n)

HTE_S_learner = S_learner.predict(SandA_all1) - S_learner.predict(SandA_all0)


# In[6]:


S_learner.predict(np.array([100,200,1000,1,5]).reshape(1, -1))


# In[7]:


S_learner.predict(np.array([100,200,1000,0,5]).reshape(1, -1))


# In[8]:


S_learner.predict(np.array([0,0,1000,0,5]).reshape(1, -1))


# Let's focus on the estimated HTEs for the first 8 patients:

# In[9]:


print("S-learner:  ",HTE_S_learner[0:8])


# **Conclusion:** Due to the difference of scales across state variables, S-learner failed to detect the heterogeneous treatment effect in this mimic3 dataset. Although it is the easiest approach to implement, the over-simplicity tends to cover some information that can be better explored with some advanced approaches.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
