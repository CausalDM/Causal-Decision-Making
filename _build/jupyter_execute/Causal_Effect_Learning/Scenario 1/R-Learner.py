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


import sys
get_ipython().system('{sys.executable} -m pip install scikit-uplift')


# In[2]:


# import related packages
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from causaldm._util_causaldm import *
from causaldm.learners.Causal_Effect_Learning.Single_Stage.Rlearner import Rlearner


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


# R-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = LGBMRegressor(max_depth=2)

HTE_R_learner = Rlearner(data_behavior, outcome, treatment, controls, n_folds, y_model, ps_model, Rlearner_model)
HTE_R_learner = HTE_R_learner.to_numpy()


# In[ ]:


print("R-learner:  ",HTE_R_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_R_learner = np.sum(HTE_R_learner-HTE_true)/n
Variance_R_learner = np.sum((HTE_R_learner-HTE_true)**2)/n
print("The overall estimation bias of R-learner is :     ", Bias_R_learner, ", \n", "The overall estimation variance of R-learner is :",Variance_R_learner,". \n")


# **Conclusion:** It's amazing to see that the bias of R-learner is significantly smaller than all other approaches.

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
