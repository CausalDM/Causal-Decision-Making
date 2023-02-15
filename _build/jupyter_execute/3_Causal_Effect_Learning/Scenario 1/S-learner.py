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


data_behavior


# In[ ]:


SandA = data_behavior.iloc[:,0:3]


# In[ ]:


# S-learner
S_learner = LGBMRegressor(max_depth=5)
#S_learner = LinearRegression()
#SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, data_behavior['R'])


# In[ ]:


HTE_S_learner = S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.ones(n).reshape(-1,1)))) - S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.zeros(n).reshape(-1,1))))


# To evaluate how well S-learner is in estimating heterogeneous treatment effect, we compare its estimates with the true value for the first 10 subjects:

# In[ ]:


print("S-learner:  ",HTE_S_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_S_learner = np.sum(HTE_S_learner-HTE_true)/n
Variance_S_learner = np.sum((HTE_S_learner-HTE_true)**2)/n
print("The overall estimation bias of S-learner is :     ", Bias_S_learner, ", \n", "The overall estimation variance of S-learner is :",Variance_S_learner,". \n")


# **Conclusion:** The performance of S-learner, at least in this toy example, is not very attractive. Although it is the easiest approach to implement, the over-simplicity tends to cover some information that can be better explored with some advanced approaches.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
