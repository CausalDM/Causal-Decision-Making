#!/usr/bin/env python
# coding: utf-8

# # Heterogeneous Treatment Effect Estimation (Single Stage)
# In the previous section, we've introduced the estimation of average treatment effect, where we aims to estimate the difference of potential outcomes by executing action $A=1$ v.s. $A=0$. That is, 
# \begin{equation*}
# \text{ATE}=\mathbb{E}[R(1)-R(0)].
# \end{equation*}
# 
# In this section, we will focus on the estimation of heterogeneous treatment effect (HTE), which is also one of the main focuses in causal inference.
# 
# 
# 
# ## Main Idea
# Let's first consider the single stage setup, where the observed data can be written as a state-action-reward triplet $\{S_i,A_i,R_i\}_{i=1}^n$ with a total of $n$ trajectories. Heterogeneous treatment effect, as we can imagine from its terminology, aims to measure the heterogeneity of the treatment effect for different subjects. Specifically, we define HTE as $\tau(s)$, where
# \begin{equation*}
# \tau(s)=\mathbb{E}[R(1)-R(0)|S=s],
# \end{equation*}
# 
# where $S=s$ denotes the state information of a subject. 
# 
# The estimation of HTE is widely used in a lot of real cases such as precision medicine, advertising, recommendation systems, etc. For example, in adversiting system, the company would like to know the impact (such as annual income) of exposing an ad to a group of customers. In this case, $S$ contains all of the information of a specific customer, $A$ denotes the status of ads exposure ($A=1$ means exposed and $A=0$ means not), and $R$ denotes the reward one can observe when assigned to policy $A$. 
# 
# Suppose the ad is a picture of a dress that can lead the customers to a detail page on a shopping website. In this case, females are more likely to be interested to click the picture and look at the detail page of a dress, resulting in a higher conversion rate than males. The difference of customers preference in clothes can be regarded as the heterogeneity of the treatment effect. By looking at the HTE for each customer, we can clearly estimate the reward of ads exposure from a granular level. 
# 
# Another related concept is conditional averge treatment effect, which is defined as
# \begin{equation*}
# \text{CATE}=\mathbb{E}[R(1)-R(0)|Z],
# \end{equation*}
# 
# where $Z$ is a collection of states with some specific characsteristics. For example, if the company is interested in the treatment effect of exposing the dress to female customers, $Z$ can be defined as ``female", and the problem can be addressed under the structure CATE estimation.
# 
# 
# 
# ## Different approaches in single-stage HTE estimation
# Next, let's briefly summarize some state-of-the-art approaches in estimating the heterogeneous treatment effect. There are several review papers which summarize some commonly-used approaches in literature, some of which are also detailed in the following subsections here. For more details please refer to [1], etc.
# 

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
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression
from causaldm._util_causaldm import *;


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


data_behavior


# In[ ]:


SandA = data_behavior.iloc[:,0:3]


# In[ ]:


# S-learner
S_learner = LGBMRegressor(max_depth=5)
#S_learner = LinearRegression()
#SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, data_behavior['R'])


# In[ ]:


HTE_S_learner = S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.ones(n).reshape(-1,1)))) - S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.zeros(n).reshape(-1,1))))


# To evaluate how well S-learner is in estimating heterogeneous treatment effect, we compare its estimates with the true value for the first 10 subjects:

# In[ ]:


print("S-learner:  ",HTE_S_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_S_learner = np.sum(HTE_S_learner-HTE_true)/n
Variance_S_learner = np.sum((HTE_S_learner-HTE_true)**2)/n
print("The overall estimation bias of S-learner is :", Bias_S_learner, ", \n", "The overall estimation variance of S-learner is :",Variance_S_learner)


# **Conclusion:** The performance of S-learner, at least in this toy example, is not very attractive. Although it is the easiest approach to implement, the over-simplicity tends to cover some information that can be better explored with some advanced approaches.

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

# In[ ]:


mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

mu0.fit(data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] )
mu1.fit(data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(data_behavior.iloc[:,0:2]) - mu0.predict(data_behavior.iloc[:,0:2])


# Now let's take a glance at the performance of T-learner by comparing it with the true value for the first 10 subjects:

# In[ ]:


print("T-learner:  ",HTE_T_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# This is quite good! T-learner captures the overall trend of the treatment effect w.r.t. the heterogeneity of different subjects.

# In[ ]:


Bias_T_learner = np.sum(HTE_T_learner-HTE_true)/n
Variance_T_learner = np.sum((HTE_T_learner-HTE_true)**2)/n
print("The overall estimation bias of T-learner is :", Bias_T_learner, ", \n", "The overall estimation variance of T-learner is :",Variance_T_learner)


# **Conclusion:** In this toy example, the overall estimation variance of T-learner is smaller than that of S-learner. In some cases when the treatment effect is relatively complex, it's likely to yield better performance by fitting two models separately. 
# 
# However, in an extreme case when both $\mu_0(s)$ and $\mu_1(s)$ are nonlinear complicated function of state $s$ while their difference is just a constant, T-learner will overfit each model very easily, yielding a nonlinear treatment effect estimator. In this case, other estimators are often preferred.

# ### **3. X-learner**
# Next, let's introduce the X-learner. As a combination of S-learner and T-learner, the X-learner can use information from the control(treatment) group to derive better estimators for the treatment(control) group, which is provably more efficient than the above two.
# 
# The basic
# 
# 
# **Step 1:**  Estimate $\mu_0(s)$ and $\mu_1(s)$ separately with any regression algorithms or supervised machine learning methods (same as T-learner);
# 
# 
# **Step 2:**  Obtain the imputed treatment effects for individuals
# \begin{equation*}
# \tilde{\Delta}_i^1:=R_i^1-\hat\mu_0(S_i^1), \quad \tilde{\Delta}_i^0:=\hat\mu_1(S_i^0)-R_i^0.
# \end{equation*}
# 
# **Step 3:**  Fit the imputed treatment effects to obtain $\hat\tau_1(s):=\mathbb{E}[\tilde{\Delta}_i^1|S=s]$ and $\hat\tau_0(s):=\mathbb{E}[\tilde{\Delta}_i^0|S=s]$;
# 
# **Step 4:**  The final HTE estimator is given by
# \begin{equation*}
# \hat{\tau}_{\text{X-learner}}(s)=g(s)\hat\tau_0(s)+(1-g(s))\hat\tau_1(s),
# \end{equation*}
# 
# where $g(s)$ is a weight function between $[0,1]$. A possible way is to use the propensity score model as an estimate of $g(s)$.

# In[ ]:


# Step 1: Fit two models under treatment and control separately, same as T-learner

mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

S_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2]
S_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2]
R_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] 
R_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] 

mu0.fit(S_T0, R_T0)
mu1.fit(S_T1, R_T1)


# In[ ]:


# Step 2: impute the potential outcomes that are unobserved in original data

n_T0 = len(R_T0)
n_T1 = len(R_T1)

Delta0 = mu1.predict(S_T0) - R_T0
Delta1 = R_T1 - mu0.predict(S_T1) 


# In[ ]:


# Step 3: Fit tau_1(s) and tau_0(s)

tau0 = LGBMRegressor(max_depth=2)
tau1 = LGBMRegressor(max_depth=2)

tau0.fit(S_T0, Delta0)
tau1.fit(S_T1, Delta1)


# In[ ]:


# Step 4: fit the propensity score model $\hat{g}(s)$ and obtain the final HTE estimator by taking weighted average of tau0 and tau1
from sklearn.linear_model import LogisticRegression 

g = LogisticRegression()
g.fit(data_behavior.iloc[:,0:2],data_behavior['A'])

HTE_X_learner = g.predict_proba(data_behavior.iloc[:,0:2])[:,0]*tau0.predict(data_behavior.iloc[:,0:2]) + g.predict_proba(data_behavior.iloc[:,0:2])[:,1]*tau1.predict(data_behavior.iloc[:,0:2])



# In[ ]:


print("X-learner:  ",HTE_X_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# X-learner also performs OK.

# In[ ]:


Bias_X_learner = np.sum(HTE_X_learner-HTE_true)/n
Variance_X_learner = np.sum((HTE_X_learner-HTE_true)**2)/n
print("The overall estimation bias of X-learner is :", Bias_X_learner, ", \n", "The overall estimation variance of X-learner is :",Variance_X_learner)


# **Conclusion:** In this toy example, the overall estimation variance of X-learner is the smallest, followed by T-learner, and the worst is given by S-learner.

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

# In[ ]:


# example


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


# example


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


# example


# ### **7. Generalized Random Forest**
# 
# Developed by Susan Athey, Julie Tibshirani and Stefan Wager, Generalized Random Forest aims to give the solution to a set of local moment equations:
# \begin{equation}
#   \mathbb{E}\big[\psi_{\tau(s),\nu(s)}(O_i)\big| S_i=s\big]=0,
# \end{equation}
# where $\tau(s)$ is the parameter we care about and $\nu(s)$ is an optional nuisance parameter. In the problem of Heterogeneous Treatment Effect Evaluation, our parameter of interest $\tau(s)=\xi\cdot \beta(s)$ is identified by 
# \begin{equation}
#   \psi_{\beta(s),\nu(s)}(R_i,A_i)=(R_i-\beta(s)\cdot A_i-c(s))(1 \quad A_i^T)^T.
# \end{equation}
# The induced estimator $\hat{\tau}(s)$ for $\tau(s)$ can thus be solved by
# \begin{equation}
#   \hat{\tau}(s)=\xi^T\left(\sum_{i=1}^n \alpha_i(s)\big(A_i-\bar{A}_\alpha\big)^{\otimes 2}\right)^{-1}\sum_{i=1}^n \alpha_i(s)\big(A_i-\bar{A}_\alpha\big)\big(R_i-\bar{R}_\alpha\big),
# \end{equation}
# where $\bar{A}_\alpha=\sum \alpha_i(s)A_i$ and $\bar{R}_\alpha=\sum \alpha_i(s)R_i$, and we write $v^{\otimes 2}=vv^T$.
# 
# Notice that this formula is just a weighted version of R-learner introduced above. However, instead of using ordinary kernel weighting functions that are prone to a strong curse of dimensionality, GRF uses an adaptive weighting function $\alpha_i(s)$ derived from a forest designed to express heterogeneity in the specified quantity of interest. 
#     
# To be more specific, in order to obtain $\alpha_i(s)$, GRF first grows a set of $B$ trees indexed by $1,\dots,B$. Then for each such tree, define $L_b(s)$ as the set of training samples falling in the same ``leaf" as x. The weights $\alpha_i(s)$ then capture the frequency with which the $i$-th training example falls into the same leaf as $s$:
# \begin{equation}
#   \alpha_{bi}(s)=\frac{\boldsymbol{1}\big(\{S_i\in L_b(s)\}\big)}{\big|L_b(s)\big|},\quad \alpha_i(s)=\frac{1}{B}\sum_{b=1}^B \alpha_{bi}(s).
# \end{equation}
# 
# To sum up, GRF aims to leverage the splitting result of a series of trees to decide the ``localized” weight for HTE estimation at each point $x_0$. Compared with kernel functions, we may expect tree-based weights to be more flexible and better performed in real settings.
# 
# 

# In[ ]:


# example


# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
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




