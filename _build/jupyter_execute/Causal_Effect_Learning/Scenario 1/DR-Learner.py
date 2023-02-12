#!/usr/bin/env python
# coding: utf-8

# ### **5. DR-learner**
# 
# DR-learner is a two-stage doubly robust estimator for HTE estimation. Before Kennedy et al. 2020 [4], there are several related approaches trying to extend the doubly robust procedure to HTE estimation, such as [5, 6, 7]. Compared with the above three estimators, DR-learner is proved to be oracle efficient under some mild assumptions detailed in Theorem 2 of [4].
# 
# The basic steps of DR-learner is given below:
# 
# **Step 1**: Nuisance training: \\
# (a)  Using $I_{1}^n$ to construct estimates $\hat{\pi}$ for the propensity scores $\pi$; \\
# (b)  Using $I_{1}^n$ to construct estimates $\hat\mu_a(s)$ for $\mu_a(s):=\mathbb{E}[R|S=s,A=a]$;
# 
# **Step 2**: Pseudo-outcome regression: \\
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


import sys
get_ipython().system('{sys.executable} -m pip install scikit-uplift')


# In[2]:


# import related packages
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from causaldm._util_causaldm import *
from causaldm.learners.Causal_Effect_Learning.Single_Stage.DRlearner import DRlearner


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


# DR-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = LGBMRegressor(max_depth=2)

HTE_DR_learner = DRlearner(data_behavior, outcome, treatment, controls, y_model, ps_model)
HTE_DR_learner = HTE_DR_learner.to_numpy()


# In[ ]:


print("DR-learner:  ",HTE_DR_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_DR_learner = np.sum(HTE_DR_learner-HTE_true)/n
Variance_DR_learner = np.sum((HTE_DR_learner-HTE_true)**2)/n
print("The overall estimation bias of DR-learner is :     ", Bias_DR_learner, ", \n", "The overall estimation variance of DR-learner is :",Variance_DR_learner,". \n")


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
