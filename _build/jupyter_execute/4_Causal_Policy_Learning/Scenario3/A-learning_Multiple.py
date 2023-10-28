#!/usr/bin/env python
# coding: utf-8

# # A-Learning (Multiple Stages)
# 
# ## Main Idea
# A-Learning, also known as Advantage Learning, is one of the main approaches to learning the optimal regime and works similarly to Q-learning. However, while Q-learning requires positing regression models to fit the expected outcome, A-learning models the contrasts between treatments and control, directly informing the optimal decision. For example, in the case of **Personalized Incentives**, A-learning aims to find the optimal incentive ($A$) for each user by modeling the difference in expected return-on-investment ($R$) between treatments. A detailed comparison between Q-learning and A-learning can be found in [1]. While [1] mainly focus on the case with binary treatment options, a complete review of A-learning with multiple treatment options can be found in [2]. Here, following the algorithm in [1], we consider contrast-based A-learning. However, there is an alternative regret-based A-learning introduced in [3]. Some recent extensions to conventional A-learning, such as deep A-learning [4] and high-dimensional A-Learning [5], will be added soon. Overall, A-learning is doubly-robust. In other words, it is less sensitive and more robust to model misspecification. 
# 
# Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., 0,1,2,3,4, where 0 stands for the control group by convention), and the outcome of interest R is **continuous** and **non-negative**, where the larger the $R$ the better. 
# 
# ## Algorithm Details
# Suppose there are $m_t$ number of options at decision point $t$, and the corresponding action space $\mathcal{A}_t=\{0,1,\dots,m_t-1\}$. At each decision point $t$, contrast-based A-learning aims to learn and estimate the constrast function, $C_{tj}(h_{t}), j=1,2,\dots,m_t-1$. Here, $h_{t}=\{\boldsymbol{S}_{1i},A_{1i},\cdots,\boldsymbol{S}_{ti}\})$ includes all the information observed till step t. Furthermore, we also need to posit a model for the conditional expected outcome for the control option (treatment $0$), $Q_t(h_t,0)$, and the propensity function $\omega(h_{t},a_{t})$. Detailed definitions are provided in the following:
# *   Q-function:
#     For the final step $T$, 
#     \begin{align}
#     Q_T(h_T,a_{T})=E[R|H_{T}=h_{T}, A_{T}=a_{T}],
#     \end{align}
#     
#     If there is a multi-stage case with total step $T>1$, for the step $t=1,\cdots,T-1$,
#     \begin{align}
#     Q_t(h_t,a_{t})=E[V_{t+1}|H_{t}=h_{t}, A_{t}=a_{t}],
#     \end{align}
#     where 
#     \begin{align}
#     V_{t}(h_{t}) = \max_{j\in\mathcal{A}_t}Q_{t}(h_t,j)
#     \end{align}
#     Alternatively, with the contrast function $C_{tj}(h_t)$, which will be defined later,
#     \begin{align}
#     Q_t(h_t,j) = Q_t(h_t,0) + C_{tj}(h_t),\quad j=0,\dots,m_t-1,\quad t=1,\dots,T.
#     \end{align}
# *   Contrast functions (optimal blip to zero functions)
#     \begin{align}
#     C_{tj}(h_t)=Q_t(h_t,j)-Q_t(h_t,0),\quad j=0,\dots,m_t-1,\quad t=1,\dots,T,
#     \end{align}
#     where $C_{t0}(h_t) = 0$.
# *   Propensity score
#     \begin{align}
#     \omega_{t}(h_t,a_t)=P(A_t=a_t|H_t=h_t)
#     \end{align}
# *   Optimal regime
#     \begin{align}
#     d_t^{opt}(h_t)=\arg\max_{j\in\mathcal{A}_t}C_{tj}(h_t)
#     \end{align}
# 
# A backward approach was proposed to find the optimized treatment regime at each decision point. 
# 
# At Decision $T$, similar as what we did previously with single decision point, we estimate the $\boldsymbol{\psi}_{Tj}$, $\boldsymbol{\phi}_T$ and $\boldsymbol{\gamma}_T$ by solving the eqautions in A.1 jointly, and the optimal decision at time $T$ is calculated accordingly. 
# 
# Then, at Decision $t=T-1,\dots,1$, we use similar trick as decision $T$, except for changing $R$ in the estimating eqautions to some pseudo outcome $\tilde{R}_{t+1,i}$, such that:
# \begin{align}
# \tilde{R}_{ti}=\tilde{R}_{t+1,i}+\max_{j=0,\dots,m_t-1}C_{tj}(h_{ti},\hat{\boldsymbol{\psi}}_{tj})-\sum_{j=1}^{m_k-1}\mathbb{I}\{A_{ti}=j\}C_{tj}(h_{ti},\hat{\boldsymbol{\psi}}_{tj}),
# \end{align}
# where $\tilde{R}_{T+1,i} = R_{i}$.
#     
# Estimating $\boldsymbol{\psi}_{tj}$, $\boldsymbol{\phi}_t$ and $\boldsymbol{\gamma}_t$ iteratively for $t=T-1,\cdots,1$, we calculated the optimal decision at time $t$, $d^{opt}_{t}(h_{ti})$ as
# \begin{align}
# d^{opt}_{t}(h_{ti})=\arg\max_{j=0,\dots,m_t-1} C_{tj}(h_{ti};\hat{\boldsymbol{\psi}}_{tj}).
# \end{align}
# 
# 
# ## Key Steps
# **Policy Learning:**
# 1. At the final decision point $t=T$, fitted a model $\omega_{T}(h_{T},a,\hat{\boldsymbol{\gamma}}_{T})$, and estimating $\boldsymbol{\psi}_{Tj}$, $\boldsymbol{\phi}_{T}$ by solving the equations in A.2 jointly;
# 2. For each individual $i$, calculated the pseudo-outcome $\tilde{R}_{Ti}$, and the optimal action $a_{Ti}$;
# 3. For decision point $t = T-1,\cdots, 1$,
#     1. fitted a model $\omega_{t}(h_{t},a,\hat{\boldsymbol{\gamma}}_{t})$, and estimating $\boldsymbol{\psi}_{tj}$, $\boldsymbol{\phi}_{t}$ by solving the equations in A.2 jointly with the pseudo-outcome $\tilde{R}_{t+1}$
#     2. For each individual $i$, calculated the pseudo-outcome $\tilde{R}_{ti}$, and the optimal action $d_t^{opt}(h_ti)$;
# 
# **Policy Evaluation:**    
# We use the backward iteration as what we did in policy learning. However, here for each round, the pseudo outcome is not the maximum of Q values. Instead, the pseudo outcome at decision point t is defined as below:
# \begin{align}
# \tilde{R}_{ti}=\tilde{R}_{t+1,i}+C_{tj*}(h_{ti},\hat{\boldsymbol{\psi}}_{tj*})-\sum_{j=1}^{m_k-1}\mathbb{I}\{A_{ti}=j\}C_{tj}(h_{ti},\hat{\boldsymbol{\psi}}_{tj}),
# \end{align}
# where $j*=d(H_{ti})$, and $d$ is the fixed regime that we want to evaluate.
# The estimated value of the policy is then the average of $\tilde{R}_{1}$.
# 
# **Note** we also provide an option for bootstrapping. Particularly, for a given policy, we utilze the boostrap resampling to get the estimated value of the regime and the corresponding estimated standard error. 
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# A demo with code on how to use the package
from causaldm.learners.CPL13.disc import ALearning
from causaldm.test import shared_simulation
import numpy as np


# In[2]:


import numpy as np
import pandas as pd
dataMDP = pd.read_csv("dataMDP_feasible.txt", sep=',')
R = np.array(dataMDP['Y'])
S = np.hstack([np.ones((len(R),1)),np.array(dataMDP[['CD4_0','CD4_6','CD4_12']])])
A = np.array(dataMDP[['A1','A2','A3']])


# In[3]:


ALearn = ALearning.ALearning()
model_info = [{'X_prop': list(range(2)),
              'X_q0': list(range(2)),
               'X_C':{1:list(range(2))},
              'action_space': {'A1':[0,1]}},
             {'X_prop': list(range(3)),
              'X_q0': list(range(3)),
               'X_C':{1:list(range(3))},
              'action_space': {'A2':[0,1]}},
             {'X_prop': list(range(4)),
              'X_q0': list(range(4)),
               'X_C':{1:list(range(4))},
              'action_space': {'A3':[0,1]}}]
# train the policy
ALearn.train(S, A, R, model_info, T=3)


# In[4]:


# recommend action
opt_d = ALearn.recommend_action(S).head()
# get the estimated value of the optimal regime
V_hat = ALearn.predict_value(S)
print("fitted contrast model:",ALearn.fitted_model['contrast'])
print("opt regime:",opt_d)
print("opt value:",V_hat)


# In[5]:


# Optional: we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
ALearn = ALearning.ALearning()
model_info = [{'X_prop': list(range(2)),
              'X_q0': list(range(2)),
               'X_C':{1:list(range(2))},
              'action_space': {'A1':[0,1]}},
             {'X_prop': list(range(3)),
              'X_q0': list(range(3)),
               'X_C':{1:list(range(3))},
              'action_space': {'A2':[0,1]}},
             {'X_prop': list(range(4)),
              'X_q0': list(range(4)),
               'X_C':{1:list(range(4))},
              'action_space': {'A3':[0,1]}}]
ALearn.train(S, A, R, model_info, T=3, bootstrap = True, n_bs = 100)
fitted_params,fitted_value,value_avg,value_std,params=ALearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)
##estimated contrast model at t = 0
print('estimated_contrast:',params[0]['contrast'])


# ### 2. Policy Evaluation

# In[6]:


#specify the fixed regime to be tested
# For example, regime d = 0 for all subjects
N, p = S.shape
ALearn = ALearning.ALearning()
# regime should be in the same format as A, which is a dict
regime = pd.DataFrame({'A1':np.array([0]*N),'A2':np.array([0]*N),'A3':np.array([0]*N)})
model_info = [{'X_prop': list(range(2)),
              'X_q0': list(range(2)),
               'X_C':{1:list(range(2))},
              'action_space': {'A1':[0,1]}},
             {'X_prop': list(range(3)),
              'X_q0': list(range(3)),
               'X_C':{1:list(range(3))},
              'action_space': {'A2':[0,1]}},
             {'X_prop': list(range(4)),
              'X_q0': list(range(4)),
               'X_C':{1:list(range(4))},
              'action_space': {'A3':[0,1]}}]
ALearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True)
ALearn.predict_value(S)


# In[7]:


# bootstrap average and the std of estimate value
ALearn.train(S, A, R, model_info, T=3, regime = regime, evaluate = True, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=ALearn.predict_value_boots(S)
print('Value_hat:',value_avg,'Value_std:',value_std)


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
# &\sum_{i=1}^n \left[\frac{\partial C_{Tj}(H_{Ti};\boldsymbol{\psi}_{Tj})}{\partial \psi_{Tj}}\{\mathbb{I}\{A_{Ti}=j\}-\omega_T(H_{Ti},j;\boldsymbol{\gamma}_T)\}\times \Big\{R_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\boldsymbol{\psi}_{Tj'})-Q_T(H_{Ti},0;\boldsymbol{\phi}_{T})\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial Q_T(H_{Ti},0;\boldsymbol{\phi}_T)}{\partial \boldsymbol{\phi}_T}\Big\{R_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\boldsymbol{\psi}_{Tj'})-Q_T(H_{Ti},0;\boldsymbol{\phi}_T)\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial \omega_T(H_{Ti},j;\boldsymbol{\gamma}_T)}{\partial \boldsymbol{\gamma}_T}\Big\{R_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\boldsymbol{\psi}_{Tj'})-Q_T(H_{Ti},0;\boldsymbol{\phi}_T)\Big\}\right]=0
# \end{aligned}
# $$
# 
# ## A.2
# $$
# \begin{aligned}
# &\sum_{i=1}^n \left[\frac{\partial C_{tj}(H_{ti};\boldsymbol{\psi}_{tj})}{\partial \boldsymbol{\psi}_{tj}}\{\mathbb{I}\{A_{ti}=j\}-\omega_T(H_{ti},j;\boldsymbol{\gamma}_t)\}\times \Big\{\tilde{R}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\boldsymbol{\psi}_{tj'})-Q_t(H_{ti},0;\boldsymbol{\phi}_{t})\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial Q_t(H_{ti},0;\boldsymbol{\phi}_t)}{\partial \boldsymbol{\phi}_t}\Big\{\tilde{R}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\boldsymbol{\psi}_{tj'})-Q_t(H_{ti},0;\boldsymbol{\phi}_t)\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial \omega_t(H_{ti},j;\boldsymbol{\gamma}_t)}{\partial \boldsymbol{\gamma}_t}\Big\{\tilde{R}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\boldsymbol{\psi}_{tj'})-Q_t(H_{ti},0;\boldsymbol{\phi}_t)\Big\}\right]=0
# \end{aligned}
# $$
# 

# In[ ]:




