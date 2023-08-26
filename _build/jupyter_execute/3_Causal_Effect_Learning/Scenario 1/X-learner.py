#!/usr/bin/env python
# coding: utf-8

# ### **3. X-learner**
# Next, let's introduce the X-learner. As a combination of S-learner and T-learner, the X-learner can use information from the control(treatment) group to derive better estimators for the treatment(control) group, which is provably more efficient than the above two.
# 
# The algorithm of X learner can be summarized as the following steps:
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


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ### MovieLens Data

# In[2]:


# Get data
MovieLens_CEL = pd.read_csv("/Users/alinaxu/Documents/CDM/CausalDM/causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL


# In[4]:


n = len(MovieLens_CEL)
userinfo_index = np.array([3,5,6,7,8,9,10])
SandA = MovieLens_CEL.iloc[:, np.array([3,4,5,6,7,8,9,10])]


# In[5]:


# Step 1: Fit two models under treatment and control separately, same as T-learner

import numpy as np
mu0 = GradientBoostingRegressor(max_depth=3)
mu1 = GradientBoostingRegressor(max_depth=3)

S_T0 = MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==0)[0],userinfo_index]
S_T1 = MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==1)[0],userinfo_index]
R_T0 = MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==0)[0],2] 
R_T1 = MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==1)[0],2] 

mu0.fit(S_T0, R_T0)
mu1.fit(S_T1, R_T1)


# In[6]:


# Step 2: impute the potential outcomes that are unobserved in original data

n_T0 = len(R_T0)
n_T1 = len(R_T1)

Delta0 = mu1.predict(S_T0) - R_T0
Delta1 = R_T1 - mu0.predict(S_T1) 


# In[7]:


# Step 3: Fit tau_1(s) and tau_0(s)

tau0 = GradientBoostingRegressor(max_depth=2)
tau1 = GradientBoostingRegressor(max_depth=2)

tau0.fit(S_T0, Delta0)
tau1.fit(S_T1, Delta1)


# In[9]:


# Step 4: fit the propensity score model $\hat{g}(s)$ and obtain the final HTE estimator by taking weighted average of tau0 and tau1
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import GradientBoostingRegressor
g = LogisticRegression()
g.fit(MovieLens_CEL.iloc[:,userinfo_index],MovieLens_CEL['Drama'])

HTE_X_learner = g.predict_proba(MovieLens_CEL.iloc[:,userinfo_index])[:,0]*tau0.predict(MovieLens_CEL.iloc[:,userinfo_index]) + g.predict_proba(MovieLens_CEL.iloc[:,userinfo_index])[:,1]*tau1.predict(MovieLens_CEL.iloc[:,userinfo_index])



# Let's focus on the estimated HTEs for three randomly chosen users:

# In[10]:


print("X-learner:  ",HTE_X_learner[np.array([0,1000,5000])])


# In[11]:


ATE_X_learner = np.sum(HTE_X_learner)/n
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_X_learner,4), "out of 5 points.")


# **Conclusion:** Same as the estimation result provided by S-learner and T-learner, people are more inclined to give higher ratings to drama than science fictions.

# **Note**: For more details about the meta learners, please refer to [1] as a detailed introduction of related methods.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
