#!/usr/bin/env python
# coding: utf-8

# # A-Learning (Single Stage)
# 
# ## Main Idea
# A-Learning, also known as Advantage Learning, is one of the main approaches to learning the optimal regime and works similarly to Q-learning. However, while Q-learning requires positing regression models to fit the expected outcome, A-learning models the contrasts between treatments and control, directly informing the optimal decision. For example, in the case of **Personalized Incentives**, A-learning aims to find the optimal incentive ($A$) for each user by modeling the difference in expected return-on-investment ($R$) between treatments. A detailed comparison between Q-learning and A-learning can be found in [1]. While [1] mainly focus on the case with binary treatment options, a complete review of A-learning with multiple treatment options can be found in [2]. Here, following the algorithm in [1], we consider contrast-based A-learning. However, there is an alternative regret-based A-learning introduced in [3]. Some recent extensions to conventional A-learning, such as deep A-learning [4] and high-dimensional A-Learning [5], will be added soon. Overall, A-learning is doubly-robust. In other words, it is less sensitive and more robust to model misspecification. 
# 
# Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., 0,1,2,3,4, where 0 stands for the control group by convention), and the outcome of interest R is **continuous** and **non-negative**, where the larger the $R$ the better. 
# 
# ## Algorithm Details
# Suppose there are $m$ number of options, and the action space $\mathcal{A}=\{0,1,\dots,m-1\}$. Contrast-based A-learning, as the name suggested, aims to learn and estimate the constrast function, $C_{j}(\boldsymbol{S})$ for each treatment $j=1,2,\cdots, m-1$. Furthermore, we also need to posit a model for the conditional expected potential outcome for the control option (treatment $0$), $Q(\boldsymbol{S},0)$, and the propensity function $\omega(\boldsymbol{S},A)$, if the true values are not specified. Detailed definitions are provided in the following.
# *   Q-function:
#     \begin{align}
#     Q(\boldsymbol{s},a)=E[R(a)|\boldsymbol{S}=\boldsymbol{s}],
#     \end{align}
#     Alternatively, with the contrast function $C_j(\boldsymbol{S})$ which will be defined later,
#     \begin{align}
#     Q(\boldsymbol{s},j) = Q(\boldsymbol{s},0) + C_{j}(\boldsymbol{s}),\quad j=0,\dots,m-1.
#     \end{align}
# *   Contrast functions (optimal blip to zero functions)
#     \begin{align}
#     C_{j}(\boldsymbol{s})=Q(\boldsymbol{s},j)-Q(\boldsymbol{s},0),\quad j=0,\dots,m-1,
#     \end{align}
#     where $C_{0}(\boldsymbol{s}) = 0$.
# *   Propensity score
#     \begin{align}
#     \omega(\boldsymbol{s},a)=P(A=a|\boldsymbol{S}=\boldsymbol{s})
#     \end{align}
# *   Optimal regime
#     \begin{align}
#     d^{opt}(\boldsymbol{s})=\arg\max_{j\in\mathcal{A}}C_{j}(\boldsymbol{s})
#     \end{align}
# Positting models, $C_{j}(\boldsymbol{s},\boldsymbol{\psi}_{j})$,$Q(\boldsymbol{s},0,\boldsymbol{\phi})$,and $\omega(\boldsymbol{s},a,\boldsymbol{\gamma})$, A-learning aims to estimate $\boldsymbol{\psi}_{j}$, $\boldsymbol{\phi}$, and $\boldsymbol{\gamma}$ by g-estimation. With the $\hat{\boldsymbol{\psi}}_{j}$ in hand, the optimal decision $d^{opt}(\boldsymbol{s})$ can be directly derived.
# 
# 
# ## Key Steps
# **Policy Learning:**
# 1. Fitted a model $\omega(\boldsymbol{s},a,\boldsymbol{\gamma})$, which can be solved directly by existing approaches (i.e., logistic regression, .etc),
# 2. Substituting the $\hat{\boldsymbol{\gamma}}$, we estimate the $\hat{\boldsymbol{\psi}}_{j}$ and $\hat{\boldsymbol{\gamma}}$ by solving the euqations in Appendix A.1 jointly.      
# 3. For each individual find the optimal action $d^{opt}(\boldsymbol{s}_{i})$ such that $d^{opt}(\boldsymbol{s}_{i}) = \arg\max_{j\in\mathcal{A}}C_{j}(h,\hat{\boldsymbol{\psi}_{j}})$.
#     
# **Policy Evaluation:**    
# 1. Fitted the functions $\omega(\boldsymbol{s},a,\boldsymbol{\gamma})$， $\hat{Q}(\boldsymbol{s},0,\hat{\boldsymbol{\beta}})$, and $\hat{C}_{j}(\boldsymbol{s},\hat{\boldsymbol{\psi}}_{j})$, based on the sampled dataset
# 2. Estimated the value of a given regime $d$ using the estimated functions, such that, $\hat{R}_{i} = \hat{Q}(\boldsymbol{s}_{i},0,\hat{\boldsymbol{\beta}})+I\{d(\boldsymbol{s}_{i})=j\}\hat{C}_{j}(\boldsymbol{s}_i,\hat{\boldsymbol{\psi}}_{j})$, and the estimated value is the average of $\hat{R}_{i}$.
# 
# **Note** we also provide an option for bootstrapping. Particularly, for a given policy, we utilze the boostrap resampling to get the estimated value of the regime and the corresponding estimated standard error. Basically, for each round of bootstrap, we resample a dataset of the same size as the original dataset with replacement, fitted the $Q(\boldsymbol{s},0)$ function and contrast functions based on the sampled dataset, and estimated the value of a given regime using the estimated $Q(\boldsymbol{s},0)$ function and contrast functions function. 
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# import learner
from causaldm._util_causaldm import *
from causaldm.learners.CPL13.disc import ALearning


# In[2]:


# get the data
S,A,R = get_data(target_col = 'spend', binary_trt = False)


# In[3]:


# transform the data into 2d numpy array
R = np.array(R)
S = np.hstack([np.ones((len(S),1)),np.array(S)])# add an intercept column
A = np.array(A)[:, np.newaxis]


# In[4]:


#1. specify the model you would like to use
model_info = [{'X_prop': [0,1,2], #[0,1,2] here stands for the intercept, recency and history
              'X_q0': [0,1,2],
               'X_C':{1:[0,1,2],2:[0,1,2]},
              'action_space': {'A':[0,1,2]}}] #A in [0,1,2]


# By specifing the model_info, we assume  the following models:
# \begin{align}
# Q(\boldsymbol{s},0,\boldsymbol{\phi}) &= \phi_{00}+\phi_{01}*recency+\phi_{02}*history,\\
# C_{1}(\boldsymbol{s},\boldsymbol{\psi}_{0}) &= 0\\
# C_{1}(\boldsymbol{s},\boldsymbol{\psi}_{1}) &= \psi_{10}+\psi_{11}*recency+\psi_{12}*history,\\
# C_{2}(\boldsymbol{s},\boldsymbol{\psi}_{2}) &= \psi_{20}+\psi_{21}*recency+\psi_{22}*history,\\
# \omega(\boldsymbol{s},a=j,\boldsymbol{\gamma}) &= \frac{exp(\gamma_{j0}+\gamma_{j1}*recency+\gamma_{j2}*history)}{\sum_{j=0}^{2}exp(\gamma_{j0}+\gamma_{j1}*recency+\gamma_{j2}*history)},\\
# \end{align}
# where $\gamma_{00}=\gamma_{01}=\gamma_{02}=0$.

# In[5]:


#2. initialize the learner
ALearn = ALearning.ALearning()
#3. train the policy
ALearn.train(S, A, R, model_info, T=1)


# In[6]:


# recommend action
opt_d = ALearn.recommend_action(S).value_counts()
# get the estimated value of the optimal regime
V_hat = ALearn.predict_value(S)
print("fitted contrast model:",ALearn.fitted_model['contrast'])
print("opt regime:",opt_d)
print("opt value:",V_hat)


# **Interpretation:** the fitted contrast models are 
# \begin{align}
# C_{0}(\boldsymbol{s},\boldsymbol{\psi}_{0}) &= 0\\
# C_{1}(\boldsymbol{s},\boldsymbol{\psi}_{1}) &= 23.39-4.03*recency+.005*history,\\
# C_{2}(\boldsymbol{s},\boldsymbol{\psi}_{2}) &= 21.03-4.71*recency-.003*history,\\
# \end{align}
# Therefore, the estimated optimal regime is:
# 1. We would recommend $A=0$ (No E-mail) if $23.39-4.03*recency+.005*history<0$ and $21.03-4.71*recency-.003*history<0$
# 2. Else, we would recommend $A=1$ (Womens E-mail) if $23.39-4.03*recency+.005*history>21.03-4.71*recency-.003*history$
# 3. Else, we would recommend $A=2$ (Mens E-Mail).
# 
# The estimated value for the estimated optimal regime is 126.19.

# In[7]:


# Optional: 
#we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
ALearn = ALearning.ALearning()
model_info = [{'X_prop': [0,1,2], #[0,1,2] here stands for the intercept, recency and history
              'X_q0': [0,1,2],
               'X_C':{1:[0,1,2],2:[0,1,2]},
              'action_space': {'A':[0,1,2]}}] #A in [0,1,2]
ALearn.train(S, A, R, model_info, T=1, bootstrap = True, n_bs = 100)
fitted_params,fitted_value,value_avg,value_std,params=ALearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


# **Interpretation:** Based on the boostrap with 200 replicates, the estimated optimal value is 132.38 with a standard error of 6.99.

# ### 2. Policy Evaluation

# In[8]:


#1. specify the fixed regime to be tested
# For example, regime d = 0 for all subjects
N=len(S)
regime = pd.DataFrame({'A':[0]*N})
#2. evaluate the regime
ALearn = ALearning.ALearning()
model_info = [{'X_prop': [0,1,2], #[0,1,2] here stands for the intercept, recency and history
              'X_q0': [0,1,2],
               'X_C':{1:[0,1,2],2:[0,1,2]},
              'action_space': {'A':[0,1,2]}}] #A in [0,1,2]
ALearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True)
ALearn.predict_value(S)


# **Interpretation:** the estimated value of the regime that always sends no emails ($A=0$) is 116.37, under the specified model.

# In[9]:


# bootstrap average and the std of estimate value
ALearn.train(S, A, R, model_info, T=1, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=ALearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


# **Interpretation:** the bootstrapped estimated value of the regime that always sends no emails is 116.37 with a bootstrapped standard error 10.20, under the specified model.

# 💥 Placeholder for C.I.

# ## References
# 1. Schulte, P. J., Tsiatis, A. A., Laber, E. B., & Davidian, M. (2014). Q-and A-learning methods for estimating optimal dynamic treatment regimes. Statistical science: a review journal of the Institute of Mathematical Statistics, 29(4), 640.
# 2. Robins, J. M. (2004). Optimal structural nested models for optimal sequential decisions. In Proceedings of the second seattle Symposium in Biostatistics (pp. 189-326). Springer, New York, NY.
# 3. Murphy, S. A. (2003). Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 65(2), 331-355.
# 4. Liang, S., Lu, W., & Song, R. (2018). Deep advantage learning for optimal dynamic treatment regime. Statistical theory and related fields, 2(1), 80-88.
# 5. Shi, C., Fan, A., Song, R., & Lu, W. (2018). High-dimensional A-learning for optimal dynamic treatment regimes. Annals of statistics, 46(3), 925.

# ## A.1
# $$
# \begin{aligned}
# &\sum_{i=1}^n \frac{\partial C_{j}(\boldsymbol{S}_{i};\boldsymbol{\psi}_{j})}{\partial \boldsymbol{\psi}_{j}}\{\mathbb{I}\{A_{i}=j\}-\omega(\boldsymbol{S}_{i},j;\hat{\boldsymbol{\gamma}})\}\times \Big\{R_i-\sum_{j'=1}^{m-1} \mathbb{I}\{A_{i}=j'\}C_{j'}(\boldsymbol{S}_{i;\boldsymbol{\psi}_{j'}})-Q(\boldsymbol{S}_{i},0;\boldsymbol{\phi})\Big\}=0\\
# &\sum_{i=1}^n \frac{\partial Q(\boldsymbol{S}_{i},0;\boldsymbol{\phi})}{\partial \boldsymbol{\phi}}\Big\{R_i-\sum_{j'=1}^{m-1} \mathbb{I}\{A_{i}=j'\}C_{j'}(\boldsymbol{S}_{i};\boldsymbol{\psi}_{j'}) Q(\boldsymbol{S}_{i},0;\boldsymbol{\phi})\Big\}=0
# \end{aligned}
# $$

# In[ ]:




