#!/usr/bin/env python
# coding: utf-8

# ## **R-Learner, DR-Learner, and Lp-R-Learner**
# 

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

# In[ ]:


from causaldm.learners.Causal_Effect_Learning.Single_Stage.DRlearner import DRlearner


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

# In[ ]:


from causaldm.learners.Causal_Effect_Learning.Single_Stage.LpRlearner import LpRlearner


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
