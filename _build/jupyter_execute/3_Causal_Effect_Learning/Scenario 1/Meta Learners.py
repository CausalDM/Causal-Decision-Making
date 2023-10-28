#!/usr/bin/env python
# coding: utf-8

# ## **Meta Learners**
# 
# In this section, we introduce three classical learners in HTE estimation. These meta learners are easy to implement, and can handle general data without additional assumptions. 

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
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from causaldm._util_causaldm import *


# In[2]:


n = 10**3  # sample size in observed data
n0 = 10**5 # the number of samples used to estimate the true reward distribution by MC
seed=223


# In[3]:


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
