#!/usr/bin/env python
# coding: utf-8

# ### **5. DR-learner**
# 
# DR-learner is a two-stage doubly robust estimator for HTE estimation. Before Kennedy et al. 2020 [4], there are several related approaches trying to extend the doubly robust procedure to HTE estimation, such as [5, 6, 7]. Compared with the above three estimators, DR-learner is proved to be oracle efficient under some mild assumptions detailed in Theorem 2 of [4].
# 
# The basic steps of DR-learner is given below:
# 
# **Step 1**: Nuisance training: 
# 
# (a)  Using $I_{1}^n$ to construct estimates $\hat{\pi}$ for the propensity scores $\pi$; 
# 
# (b)  Using $I_{1}^n$ to construct estimates $\hat\mu_a(s)$ for $\mu_a(s):=\mathbb{E}[R|S=s,A=a]$;
# 
# **Step 2**: Pseudo-outcome regression: 
# 
# Define $\widehat{\phi}(Z)$ as the pseudo-outcome where 
# \begin{equation}
# \widehat{\phi}(Z)=\frac{A-\hat{\pi}(S)}{\hat{\pi}(S)\{1-\hat{\pi}(S)\}}\Big\{R-\hat{\mu}_A(S)\Big\}+\hat{\mu}_1(S)-\hat{\mu}_0(S),
# \end{equation}
# and regress it on covariates $S$ in the test sample $I_2^n$, yielding 
# \begin{equation}
# \widehat{\tau}_{\text{DR-learner}}(s)=\widehat{\mathbb{E}}_n[\widehat{\phi}(Z)|S=s].
# \end{equation}
# 

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from causaldm.learners.CEL.Single_Stage.DRlearner import DRlearner


# ### MovieLens Data

# In[2]:


# Get the MovieLens data
import os
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
MovieLens_CEL = pd.read_csv("./causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL


# In[3]:


n = len(MovieLens_CEL)
userinfo_index = np.array([3,5,6,7,8,9,10])
SandA = MovieLens_CEL.iloc[:, np.array([3,4,5,6,7,8,9,10])]


# In[9]:


# DR-learner for HTE estimation
np.random.seed(1)

outcome = 'rating'
treatment = 'Drama'
#controls = MovieLens_CEL.columns[userinfo_index]
controls = ['age', 'gender_M', 'occupation_academic/educator',
       'occupation_college/grad student', 'occupation_executive/managerial',
       'occupation_other', 'occupation_technician/engineer']
n_folds = 5
y_model = GradientBoostingRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = GradientBoostingRegressor(max_depth=2)

HTE_DR_learner = DRlearner(MovieLens_CEL, outcome, treatment, controls, y_model, ps_model)
HTE_DR_learner = HTE_DR_learner.to_numpy()


# Let's focus on the estimated HTEs for three randomly chosen users:

# In[10]:


print("DR-learner:  ",HTE_DR_learner[np.array([0,1000,5000])])


# In[11]:


ATE_DR_learner = np.sum(HTE_DR_learner)/n
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_DR_learner,4), "out of 5 points.")


# **Conclusion:** Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by 0.3541 out of 5 points.

# ## References
# 
# 2. Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319, 2021.
# 
# 3. Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society, pages 931–954, 1988.
# 
# 4. Edward H Kennedy. Optimal doubly robust estimation of heterogeneous causal effects. arXiv preprint arXiv:2004.14497, 2020
# 
# 5. M. J. van der Laan. Statistical inference for variable importance. The International Journal of Biostatistics, 2(1), 2006.
# 
# 6. S. Lee, R. Okui, and Y.-J. Whang. Doubly robust uniform confidence band for the conditional average treatment effect function. Journal of Applied Econometrics, 32(7):1207–1225, 2017.
# 
# 7. D. J. Foster and V. Syrgkanis. Orthogonal statistical learning. arXiv preprint arXiv:1901.09036, 2019.
