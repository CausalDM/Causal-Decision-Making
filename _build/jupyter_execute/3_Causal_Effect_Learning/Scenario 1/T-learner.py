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


mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

mu0.fit(data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] )
mu1.fit(data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(data_behavior.iloc[:,0:2]) - mu0.predict(data_behavior.iloc[:,0:2])


# Now let's take a glance at the performance of T-learner by comparing it with the true value for the first 10 subjects:

# In[ ]:


print("T-learner:  ",HTE_T_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# This is quite good! T-learner captures the overall trend of the treatment effect w.r.t. the heterogeneity of different subjects.

# In[ ]:


Bias_T_learner = np.sum(HTE_T_learner-HTE_true)/n
Variance_T_learner = np.sum((HTE_T_learner-HTE_true)**2)/n
print("The overall estimation bias of T-learner is :     ", Bias_T_learner, ", \n", "The overall estimation variance of T-learner is :",Variance_T_learner,". \n")


# **Conclusion:** In this toy example, the overall estimation variance of T-learner is smaller than that of S-learner. In some cases when the treatment effect is relatively complex, it's likely to yield better performance by fitting two models separately. 
# 
# However, in an extreme case when both $\mu_0(s)$ and $\mu_1(s)$ are nonlinear complicated function of state $s$ while their difference is just a constant, T-learner will overfit each model very easily, yielding a nonlinear treatment effect estimator. In this case, other estimators are often preferred.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
