#!/usr/bin/env python
# coding: utf-8

# 
# ### **2. T-learner**
# The second learner is called T-learner, which denotes ``two learners". Instead of fitting a single model to estimate the potential outcomes under both treatment and control groups, T-learner aims to learn different models for $\mathbb{E}[R(1)|S]$ and $\mathbb{E}[R(0)|S]$ separately, and finally combines them to obtain a final HTE estimator.
# 
# Define the control response function as $\mu_0(s)=\mathbb{E}[R(0)|S=s]$, and the treatment response function as $\mu_1(s)=\mathbb{E}[R(1)|S=s]$. The algorithm of T-learner is summarized below:
# 
# **Step 1:**  Estimate $\mu_0(s)$ and $\mu_1(s)$ separately with any regression algorithms or supervised machine learning methods;
# 
# **Step 2:**  Estimate HTE by 
# \begin{equation*}
# \hat{\tau}_{\text{T-learner}}(s)=\hat\mu_1(s)-\hat\mu_0(s).
# \end{equation*}
# 
# 

# ### Mimic3 Data

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression


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


mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

mu0.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5] )
mu1.fit(data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index],data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(data_CEL_selected.iloc[:,userinfo_index]) - mu0.predict(data_CEL_selected.iloc[:,userinfo_index])


# Now let's take a glance at the performance of T-learner by comparing it with the true value for the first 8 subjects:

# In[5]:


print("T-learner:  ",HTE_T_learner[0:8])


# This is quite good! T-learner captures the overall trend of the treatment effect w.r.t. the heterogeneity of different subjects.

# **Conclusion:** In Mimic3 data, HTE can be successfully estimated by T-learner. In some cases when the treatment effect is relatively complex, it's likely to yield better performance by fitting two models separately. 
# 
# However, in an extreme case when both $\mu_0(s)$ and $\mu_1(s)$ are nonlinear complicated function of state $s$ while their difference is just a constant, T-learner will overfit each model very easily, yielding a nonlinear treatment effect estimator. In this case, other estimators are often preferred.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
