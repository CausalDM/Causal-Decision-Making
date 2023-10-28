#!/usr/bin/env python
# coding: utf-8

# ### **6. Lp-R-learner**
# 
# As an extension of R-learner, Lp-R-learner combined the idea of residual regression with local polynomial adaptation, and leveraged the idea of cross fitting to further relax the conditions needed to obtain the oracle convergence rate. For brevity of content, we will just introduce their main algorithm. For more details about its theory and real data performance please see the paper written by Kennedy [4]. 
# 	
# Let $(I_{1a}^n, I_{1b}^n,I_{2}^n)$ denote three independent samples of $n$ observations of $Z_i = (S_i, A_i, R_i)$. Let $b:\mathbb{R}^d\rightarrow \mathbb{R}^p$ denote the vector of basis functions consisting of all powers of each covariate, up to order $\gamma$, and all interactions up to degree $\gamma$ polynomials. Let $K_{hs}(S)=\frac{1}{h^d}K\left(\frac{S-s}{h}\right)$ for $k:\mathbb{R}^d\rightarrow \mathbb{R}$ a bounded kernel function with support $[-1,1]^d$, and $h$ is a bandwidth parameter.
# 
# **Step 1**: Nuisance training: 
# 
# (a)  Using $I_{1a}^n$ to construct estimates $\hat{\pi}_a$ of the propensity scores $\pi$; 
# 
# (b)  Using $I_{1b}^n$ to construct estimates $\hat{\eta}$ of the regression function $\eta=\pi\mu_1+(1-\pi)\mu_0$, and estimtes $\hat{\pi}_b$ of the propensity scores $\pi$.
# 
# **Step 2**: Localized double-residual regression: 
# 
# Define $\hat{\tau}_r(s)$ as the fitted value from a kernel-weighted least squares regression (in the test sample $I_2^n$) of outcome residual $(R-\hat{\eta})$ on basis terms $b$ scaled by the treatment residual $A-\hat{\pi}_b$, with weights $\Big(\frac{A-\hat{\pi}_a}{A-\hat{\pi}_b}\Big)\cdot K_{hs}$. Thus $\hat{\tau}_r(s)=b(0)^T\hat{\theta}$ for
# \begin{equation}
# 		\hat{\theta}=\arg\min_{\theta\in\mathbb{R}^p}\mathbb{P}_n\left(K_{hs}(S)\Big\{ \frac{A-\hat{\pi}_a(S)}{A-\hat{\pi}_b(S)}\Big\} \left[  \big\{R-\hat{\eta}(S)\big\}-\theta^Tb(S-s_0)\big\{A-\hat{\pi}_b(S)\big\} \right] \right).
# \end{equation}
# **Step 3**: Cross-fitting(optional): 
# 
# Repeat Step 1–2 twice, first using $(I^n_{1b} , I_2^n)$ for nuisance training and $I_{1a}^n$ as the test samplem and then using $(I^n_{1a} , I_2^n)$ for training and $I_{1b}^n$ as the test sample. Use the average of the resulting three estimators of $\tau$ as the final estimator $\hat{\tau}_r$.
# 
# In the theory section, Kennedy proved that Lp-R-learner, compared with traditional DR learner, can achieve the oracle convergence rate under milder conditions. 

# In[1]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from causaldm.learners.CEL.Single_Stage.LpRlearner import LpRlearner
import warnings
warnings.filterwarnings('ignore')


# ### MovieLens Data

# In[2]:


# Get the MovieLens data
import os
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
MovieLens_CEL = pd.read_csv("./causaldm/data/MovieLens_CEL.csv")
MovieLens_CEL.pop(MovieLens_CEL.columns[0])
MovieLens_CEL = MovieLens_CEL[MovieLens_CEL.columns.drop(['Comedy','Action', 'Thriller'])]
MovieLens_CEL


# In[14]:


n = len(MovieLens_CEL)


# In[17]:


import random
np.random.seed(1)

outcome = 'rating'
treatment = 'Drama'
controls = ['age', 'gender_M', 'occupation_academic/educator',
       'occupation_college/grad student', 'occupation_executive/managerial',
       'occupation_other', 'occupation_technician/engineer']
n_folds = 5
y_model = GradientBoostingRegressor(max_depth=3)
ps_model_a = LogisticRegression()
ps_model_b = LogisticRegression()
s = 1
LpRlearner_model = LinearRegression()

sample_index = random.sample(np.arange(len(MovieLens_CEL)).tolist(),1000)
MovieLens_CEL = MovieLens_CEL.iloc[sample_index,:]

HTE_Lp_R_learner = LpRlearner(MovieLens_CEL, outcome, treatment, controls, y_model, ps_model_a, ps_model_b, s, LpRlearner_model, degree = 1)


# Let's focus on the estimated HTEs for three randomly chosen users:

# In[18]:


print("Lp-R-learner:  ",HTE_Lp_R_learner[np.array([0,300,900])])


# In[19]:


ATE_Lp_R_learner = np.sum(HTE_Lp_R_learner)/1000
print("Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by",round(ATE_Lp_R_learner,4), "out of 5 points.")


# **Conclusion:** Choosing Drama instead of Sci-Fi is expected to improve the rating of all users by 0.2182 out of 5 points.

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

# In[ ]:




