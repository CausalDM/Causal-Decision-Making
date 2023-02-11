#!/usr/bin/env python
# coding: utf-8

# ### **6. Lp-R-learner**
# 
# As an extension of R-learner, Lp-R-learner combined the idea of residual regression with local polynomial adaptation, and leveraged the idea of cross fitting to further relax the conditions needed to obtain the oracle convergence rate. For brevity of content, we will just introduce their main algorithm. For more details about its theory and real data performance please see the paper written by Kennedy [4]. 
# 	
# Let $(I_{1a}^n, I_{1b}^n,I_{2}^n)$ denote three independent samples of $n$ observations of $Z_i = (S_i, A_i, R_i)$. Let $b:\mathbb{R}^d\rightarrow \mathbb{R}^p$ denote the vector of basis functions consisting of all powers of each covariate, up to order $\gamma$, and all interactions up to degree $\gamma$ polynomials. Let $K_{hs}(S)=\frac{1}{h^d}K\left(\frac{S-s}{h}\right)$ for $k:\mathbb{R}^d\rightarrow \mathbb{R}$ a bounded kernel function with support $[-1,1]^d$, and $h$ is a bandwidth parameter.
# 
# **Step 1**: Nuisance training: \\
# (a)  Using $I_{1a}^n$ to construct estimates $\hat{\pi}_a$ of the propensity scores $\pi$; \\
# (b)  Using $I_{1b}^n$ to construct estimates $\hat{\eta}$ of the regression function $\eta=\pi\mu_1+(1-\pi)\mu_0$, and estimtes $\hat{\pi}_b$ of the propensity scores $\pi$.
# 
# **Step 2**: Localized double-residual regression: \\
# Define $\hat{\tau}_r(s)$ as the fitted value from a kernel-weighted least squares regression (in the test sample $I_2^n$) of outcome residual $(R-\hat{\eta})$ on basis terms $b$ scaled by the treatment residual $A-\hat{\pi}_b$, with weights $\Big(\frac{A-\hat{\pi}_a}{A-\hat{\pi}_b}\Big)\cdot K_{hs}$. Thus $\hat{\tau}_r(s)=b(0)^T\hat{\theta}$ for
# \begin{equation}
# 		\hat{\theta}=\arg\min_{\theta\in\mathbb{R}^p}\mathbb{P}_n\left(K_{hs}(S)\Big\{ \frac{A-\hat{\pi}_a(S)}{A-\hat{\pi}_b(S)}\Big\} \left[  \big\{R-\hat{\eta}(S)\big\}-\theta^Tb(S-s_0)\big\{A-\hat{\pi}_b(S)\big\} \right] \right).
# \end{equation}
# **Step 3**: Cross-fitting(optional): \\
# Repeat Step 1–2 twice, first using $(I^n_{1b} , I_2^n)$ for nuisance training and $I_{1a}^n$ as the test samplem and then using $(I^n_{1a} , I_2^n)$ for training and $I_{1b}^n$ as the test sample. Use the average of the resulting three estimators of $\tau$ as the final estimator $\hat{\tau}_r$.
# 
# In the theory section, Kennedy proved that Lp-R-learner, compared with traditional DR learner, can achieve the oracle convergence rate under milder conditions. 

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
from causaldm.learners.Causal_Effect_Learning.Single_Stage.LpRlearner import LpRlearner


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


# Lp-R-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model_a = LogisticRegression()
ps_model_b = LogisticRegression()
s = 1
LpRlearner_model = LinearRegression()

HTE_Lp_R_learner = LpRlearner(data_behavior, outcome, treatment, controls, y_model, ps_model_a, ps_model_b, s, LpRlearner_model, degree = 1)


# In[ ]:


print("Lp_R-learner:  ",HTE_Lp_R_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_Lp_R_learner = np.sum(HTE_Lp_R_learner-HTE_true)/n
Variance_Lp_R_learner = np.sum((HTE_Lp_R_learner-HTE_true)**2)/n
print("The overall estimation bias of Lp_R-learner is :     ", Bias_Lp_R_learner, ", \n", "The overall estimation variance of Lp_R-learner is :",Variance_Lp_R_learner,". \n")


# **Conclusion**: It will cost more time to use Lp-R-learner than other approaches. However, the overall estimation variance of Lp-R-learner is incredibly smaller than other approaches.

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
