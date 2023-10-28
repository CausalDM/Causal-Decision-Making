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


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ### MovieLens Data

# In[2]:


# Get the MovieLens data
import os
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
MovieLens_CEL = pd.read_csv("./causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL = MovieLens_CEL[MovieLens_CEL.columns.drop(['Comedy','Action', 'Thriller'])]
MovieLens_CEL


# In[3]:


n = len(MovieLens_CEL)
userinfo_index = np.array([3,6,7,8,9,10,11])
SandA = MovieLens_CEL.iloc[:, np.array([3,5,6,7,8,9,10,11])]


# In[4]:


mu0 = GradientBoostingRegressor(max_depth=3)
mu1 = GradientBoostingRegressor(max_depth=3)

mu0.fit(MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==0)[0],userinfo_index],MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==0)[0],2] )
mu1.fit(MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==1)[0],userinfo_index],MovieLens_CEL.iloc[np.where(MovieLens_CEL['Drama']==1)[0],2] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(MovieLens_CEL.iloc[:,userinfo_index]) - mu0.predict(MovieLens_CEL.iloc[:,userinfo_index])


# Let's focus on the estimated HTEs for three randomly chosen users:

# In[5]:


print("T-learner:  ",HTE_T_learner[np.array([0,1000,5000])])


# In[6]:


ATE_T_learner = np.sum(HTE_T_learner)/n
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_T_learner,4), "out of 5 points.")


# **Conclusion:** Same as the estimation result provided by S-learner, people are more inclined to give higher ratings to drama than science fictions. The expected causal effect estiamted by T-learner is larger than S-learner. In some cases when the treatment effect is relatively complex, it's likely to yield better performance by fitting two models separately. 
# 
# However, in an extreme case when both $\mu_0(s)$ and $\mu_1(s)$ are nonlinear complicated function of state $s$ while their difference is just a constant, T-learner will overfit each model very easily, yielding a nonlinear treatment effect estimator. In this case, other estimators are often preferred.

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
