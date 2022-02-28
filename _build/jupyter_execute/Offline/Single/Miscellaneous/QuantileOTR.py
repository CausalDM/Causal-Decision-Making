#!/usr/bin/env python
# coding: utf-8

# # **Quantile Optimal Treatment Regime**
# ---
# 
# *   **Basic Idea:**
# 
#    The aim of this paper is to estimate the quantile-optimal treatment regime given a class of feasible treatment regimes $\mathbb{D}=\{I(X^T\beta>0:\beta\in \mathbb{B})\}$. That is, the objective function is to maximize
#    $$\arg\max_{d\in \mathbb{D}}Q_{\tau}(Y^*(d)),$$
# where $Q_{\tau}$ denotes the quantile value at level $\tau$. \\
# To estimate the quantile value at a given level $\tau$, a doubly robust estimator was implemented:
# $$
# \hat{Q}_{\tau}(\beta)=\frac{1}{n}\sum_{i=1}^n \left[\frac{C_i(\beta)}{\hat{\pi}(X_i,\beta)}\rho_{\tau}(Y_i-a)+\left(1-\frac{C_i(\beta)}{\hat{\pi}}\right)\rho_{\tau}(\hat{Y}_i^{**}-a)\right].
# $$
# 
# *   Next, we will use an simulation data and a real dataset to illustrate the performance of Quantile Optimal Treatment Regime.
# ---

# ## **Demo on Simulation data**

# In[1]:


# define some functions to generate the original data

# definition of behavior policy (or original treatment function)
def b(x1,x2):
    if np.shape(x1) == ():
      x1 = np.array([x1])
    if np.shape(x2) == ():
      x2 = np.array([x2])
    n = len(x1)
    epsilon = 0.25 * np.random.normal(size=n)
    return ((x1+x2 + epsilon > 0) + 0)

# definition of target policy (or the treatment of our interest)
def pi(x1,x2):
    if np.shape(x1) == ():
      x1 = np.array([x1])
    if np.shape(x2) == ():
      x2 = np.array([x2])
    n = len(x1)
    return ((x1+x2 > 0) + 0)

# definition of reward function r(x,a), where x denotes states and a denotes actions(treatments)
def r(x1, x2, a):
    if np.shape(x1) == ():
      x1 = np.array([x1])
    if np.shape(x2) == ():
      x2 = np.array([x2])
    if np.shape(a) == ():
      a = np.array([a])
    n = len(x1)
    epsilon = 0.25 * np.random.normal(size=n)
    return (np.ones(n) - x1 + 2*a*x1 +2*x2 +0.5*a*x2) * (np.ones(n) + epsilon)


# In[2]:


n0 = (10**3)  # the number of samples used to estimate the true reward distribution by MC

np.random.seed(seed=223)
import pandas as pd
import numpy as np
tau=0.5
X_1=np.random.normal(loc=-0.5,size=n0)
X_2=np.random.normal(loc=0.6,size=n0)
A=b(X_1,X_2)
Y=r(X_1, X_2, A) 
data={'X_1':X_1,'X_2':X_2,'A':A,'Y':Y}
data=pd.DataFrame(data)
(data)


# In[ ]:


# initialize the learner
Quantile_OTR=QuantileOTR()

# when a=1, Y ~ 1+x1+2.5*x2
# when a=0, Y ~ 1-x1+2*x2
moCondQuant_0 = ['X_1', 'X_2']
moCondQuant_1 = ['X_1', 'X_2']

coefficient, coef_original_scale, Q_est=Quantile_OTR.DR_Qopt(data=data, tau=0.5, moCondQuant_0=moCondQuant_0, moCondQuant_1=moCondQuant_1,moPropen = "NotBinaryRandom")


# In[ ]:


# the coefficient that maximize the quantile
coef_original_scale


# In[ ]:


# the estimated quantile that corresponds to the optimized coefficient
Q_est


# ##**Reference**
# 
# 
# *   Lan Wang, Yu Zhou, Rui Song and Ben Sherwood. "Quantile-Optimal Treatment Regimes." Journal of the American Statistical Association 2018; 113(523): 1243–1254.
# 
# 
# 
