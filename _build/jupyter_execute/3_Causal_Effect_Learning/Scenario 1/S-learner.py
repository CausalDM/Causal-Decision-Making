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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ### MovieLens Data

# In[2]:


# Get data
MovieLens_CEL = pd.read_csv("/Users/alinaxu/Documents/CDM/CausalDM/causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL


# In[10]:


n = len(MovieLens_CEL)
userinfo_index = np.array([3,5,6,7,8,9,10])
SandA = MovieLens_CEL.iloc[:, np.array([3,4,5,6,7,8,9,10])]


# In[11]:


# S-learner
S_learner = GradientBoostingRegressor(max_depth=5)
#S_learner = LinearRegression()
#SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, MovieLens_CEL['rating'])


# In[12]:


SandA_all1 = SandA.copy()
SandA_all0 = SandA.copy()
SandA_all1.iloc[:,4]=np.ones(n)
SandA_all0.iloc[:,4]=np.zeros(n)

HTE_S_learner = S_learner.predict(SandA_all1) - S_learner.predict(SandA_all0)


# Let's focus on the estimated HTEs for three randomly chosen users:

# In[22]:


print("S-learner:  ",HTE_S_learner[np.array([0,1000,5000])])


# In[20]:


ATE_S_learner = np.sum(HTE_S_learner)/n
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_S_learner,4), "out of 5 points.")


# **Conclusion:** As we can see from the estimated ATE by S-learner, people are more inclined to give higher ratings to drama than science fictions. 

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
