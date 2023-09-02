#!/usr/bin/env python
# coding: utf-8

# ### **4. R learner**
# The idea of classical R-learner came from Robinson 1988 [3] and was formalized by Nie and Wager in 2020 [2]. The main idea of R learner starts from the partially linear model setup, in which we assume that
# \begin{equation}
#   \begin{aligned}
#     R&=A\tau(S)+g_0(S)+U,\\
#     A&=m_0(S)+V,
#   \end{aligned}
# \end{equation}
# where $U$ and $V$ satisfies $\mathbb{E}[U|D,X]=0$, $\mathbb{E}[V|X]=0$.
# 
# After several manipulations, it’s easy to get
# \begin{equation}
# 	R-\mathbb{E}[R|S]=\tau(S)\cdot(A-\mathbb{E}[A|S])+\epsilon.
# \end{equation}
# Define $m_0(X)=\mathbb{E}[A|S]$ and $l_0(X)=\mathbb{E}[R|S]$. A natural way to estimate $\tau(X)$ is given below, which is also the main idea of R-learner:
# 
# **Step 1**: Regress $R$ on $S$ to obtain model $\hat{\eta}(S)=\hat{\mathbb{E}}[R|S]$; and regress $A$ on $S$ to obtain model $\hat{m}(S)=\hat{\mathbb{E}}[A|S]$.
# 
# **Step 2**: Regress outcome residual $R-\hat{l}(S)$ on propensity score residual $A-\hat{m}(S)$.
# 
# That is,
# \begin{equation}
# 	\hat{\tau}(S)=\arg\min_{\tau}\left\{\mathbb{E}_n\left[\left(\{R_i-\hat{\eta}(S_i)\}-\{A_i-\hat{m}(S_i)\}\cdot\tau(S_i)\right)^2\right]\right\}	
# \end{equation}
# 
# The easiest way to do so is to specify $\hat{\tau}(S)$ to the linear function class. In this case, $\tau(S)=S\beta$, and the problem becomes to estimate $\beta$ by solving the following linear regression:
# \begin{equation}
# 	\hat{\beta}=\arg\min_{\beta}\left\{\mathbb{E}_n\left[\left(\{R_i-\hat{\eta}(S_i)\}-\{A_i-\hat{m}(S_i)\} S_i\cdot \beta\right)^2\right]\right\}.
# \end{equation}
# 
# 

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from causaldm.learners.CEL.Single_Stage.Rlearner import Rlearner


# ### MovieLens Data

# In[4]:


# Get the MovieLens data
import os
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
MovieLens_CEL = pd.read_csv("./causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL


# In[9]:


n = len(MovieLens_CEL)
userinfo_index = np.array([3,5,6,7,8,9,10])
SandA = MovieLens_CEL.iloc[:, np.array([3,4,5,6,7,8,9,10])]


# In[33]:


MovieLens_CEL.columns[userinfo_index]


# In[40]:


# R-learner for HTE estimation
np.random.seed(1)
outcome = 'rating'
treatment = 'Drama'
controls = ['age', 'gender_M', 'occupation_academic/educator',
       'occupation_college/grad student', 'occupation_executive/managerial',
       'occupation_other', 'occupation_technician/engineer']
n_folds = 5
y_model = GradientBoostingRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = GradientBoostingRegressor(max_depth=2)

HTE_R_learner = Rlearner(MovieLens_CEL, outcome, treatment, controls, n_folds, y_model, ps_model, Rlearner_model)
HTE_R_learner = HTE_R_learner.to_numpy()


# Let's focus on the estimated HTEs for three randomly chosen users:

# In[41]:


print("R-learner:  ",HTE_R_learner[np.array([0,1000,5000])])


# In[42]:


ATE_R_learner = np.sum(HTE_R_learner)/n
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_R_learner,4), "out of 5 points.")


# **Conclusion:** Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by 0.0755 out of 5 points.

# ## References
# 
# 2. Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319, 2021.
# 
# 3. Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society, pages 931–954, 1988.
# 

# In[ ]:




