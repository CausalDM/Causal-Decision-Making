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

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install scikit-uplift')


# In[2]:


# import related packages
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression
from causaldm._util_causaldm import *;


# In[ ]:


n = 10**3  # sample size in observed data
n0 = 10**5 # the number of samples used to estimate the true reward distribution by MC
seed=223


# In[ ]:


# Get data
data_behavior = get_data_simulation(n, seed, policy="behavior")
#data_target = get_data_simulation(n0, seed, policy="target")

# The true expected heterogeneous treatment effect
HTE_true = get_data_simulation(n, seed, policy="1")['R']-get_data_simulation(n, seed, policy="0")['R']



# In[ ]:


# Step 1: Fit two models under treatment and control separately, same as T-learner

import numpy as np
mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

S_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2]
S_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2]
R_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] 
R_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] 

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
g.fit(data_behavior.iloc[:,0:2],data_behavior['A'])

HTE_X_learner = g.predict_proba(data_behavior.iloc[:,0:2])[:,0]*tau0.predict(data_behavior.iloc[:,0:2]) + g.predict_proba(data_behavior.iloc[:,0:2])[:,1]*tau1.predict(data_behavior.iloc[:,0:2])




# In[ ]:


print("X-learner:  ",HTE_X_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# From the result above we can see that X-learner can roughly catch the trend of treatment effect w.r.t. the change of baseline information $S$. In this synthetic example, X-learner also performs slightly better than T-learner.

# In[ ]:


Bias_X_learner = np.sum(HTE_X_learner-HTE_true)/n
Variance_X_learner = np.sum((HTE_X_learner-HTE_true)**2)/n
print("The overall estimation bias of X-learner is :     ", Bias_X_learner, ", \n", "The overall estimation variance of X-learner is :",Variance_X_learner,". \n")


# **Conclusion:** In this toy example, the overall estimation variance of X-learner is the smallest, followed by T-learner, and the worst is given by S-learner.
# 
# 

# **Note**: For more details about the meta learners, please refer to [1].

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
