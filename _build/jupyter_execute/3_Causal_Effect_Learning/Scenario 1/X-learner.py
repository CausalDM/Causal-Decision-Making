#!/usr/bin/env python
# coding: utf-8

# ### **3. X-learner**
# Next, let's introduce the X-learner. As a combination of S-learner and T-learner, the X-learner can use information from the control(treatment) group to derive better estimators for the treatment(control) group, which is provably more efficient than the above two.
# 
# The basic
# 
# 
# **Step 1:**  Estimate $\mu_0(s)$ and $\mu_1(s)$ separately with any regression algorithms or supervised machine learning methods (same as T-learner);
# 
# 
# **Step 2:**  Obtain the imputed treatment effects for individuals
# \begin{equation*}
# \tilde{\Delta}_i^1:=R_i^1-\hat\mu_0(S_i^1), \quad \tilde{\Delta}_i^0:=\hat\mu_1(S_i^0)-R_i^0.
# \end{equation*}
# 
# **Step 3:**  Fit the imputed treatment effects to obtain $\hat\tau_1(s):=\mathbb{E}[\tilde{\Delta}_i^1|S=s]$ and $\hat\tau_0(s):=\mathbb{E}[\tilde{\Delta}_i^0|S=s]$;
# 
# **Step 4:**  The final HTE estimator is given by
# \begin{equation*}
# \hat{\tau}_{\text{X-learner}}(s)=g(s)\hat\tau_0(s)+(1-g(s))\hat\tau_1(s),
# \end{equation*}
# 
# where $g(s)$ is a weight function between $[0,1]$. A possible way is to use the propensity score model as an estimate of $g(s)$.

# ### Mimic3 Data

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression


# In[ ]:


# Get data
n = 5000
selected = ['Glucose','paO2','PaO2_FiO2',  'iv_input', 'SOFA','reward']
data_CEL_selected = pd.read_csv("C:/Users/Public/CausalDM/causaldm/data/mimic3_CEL_selected.csv")
data_CEL_selected.pop(data_CEL_selected.columns[0])
data_CEL_selected


# In[ ]:


userinfo_index = np.array([0,1,2,4])
SandA = data_CEL_selected.iloc[:, np.array([0,1,2,3,4])]


# In[ ]:


# Step 1: Fit two models under treatment and control separately, same as T-learner

import numpy as np
mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

S_T0 = data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],userinfo_index]
S_T1 = data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],userinfo_index]
R_T0 = data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==0)[0],5] 
R_T1 = data_CEL_selected.iloc[np.where(data_CEL_selected['iv_input']==1)[0],5] 

mu0.fit(S_T0, R_T0)
mu1.fit(S_T1, R_T1)


# In[ ]:


# Step 2: impute the potential outcomes that are unobserved in original data

n_T0 = len(R_T0)
n_T1 = len(R_T1)

Delta0 = mu1.predict(S_T0) - R_T0
Delta1 = R_T1 - mu0.predict(S_T1) 


# In[ ]:


# Step 3: Fit tau_1(s) and tau_0(s)

tau0 = LGBMRegressor(max_depth=2)
tau1 = LGBMRegressor(max_depth=2)

tau0.fit(S_T0, Delta0)
tau1.fit(S_T1, Delta1)


# In[ ]:


# Step 4: fit the propensity score model $\hat{g}(s)$ and obtain the final HTE estimator by taking weighted average of tau0 and tau1
from sklearn.linear_model import LogisticRegression 

g = LogisticRegression()
g.fit(data_CEL_selected.iloc[:,userinfo_index],data_CEL_selected['iv_input'])

HTE_X_learner = g.predict_proba(data_CEL_selected.iloc[:,userinfo_index])[:,0]*tau0.predict(data_CEL_selected.iloc[:,userinfo_index]) + g.predict_proba(data_CEL_selected.iloc[:,userinfo_index])[:,1]*tau1.predict(data_CEL_selected.iloc[:,userinfo_index])



# In[ ]:


print("X-learner:  ",HTE_X_learner[0:8])


# **Note**: For more details about the meta learners, please refer to [1].

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
