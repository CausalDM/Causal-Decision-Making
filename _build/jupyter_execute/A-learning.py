#!/usr/bin/env python
# coding: utf-8

# # A-Learning

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## Main Idea
# A-Learning, also known as Advantage Learning, is one of the main approaches to learning the optimal regime and works similarly to Q-learning. However, while Q-learning requires positing regression models to fit the expected outcome, A-learning models the contrasts between treatments and control, which can directly inform the optimal decision. A detailed comparison between Q-learning and A-learning can be found in [1]. While [1] mainly focus on the case with binary treatment options, a complete review of A-learning with multiple treatment options can be found in [2]. Here, following the algorithm in [1], we consider contrast-based A-learning. However, there is an alternative regret-based A-learning introduced in [3]. Some recent extensions to conventional A-learning, such as deep A-learning [4] and high-dimensional A-Learning [5], will be added soon.
# 
# Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., 0,1,2,3,4, where 0 stands for the control group by convention), and the outcome of interest Y is **continuous** and **non-negative**, where the larger the $Y$ the better. 
# 
# contrast-based A-learning, as the name suggested, aims to learn and estimate the constrast function, $C_{tj}(h_{t})$. Here, $h_{t}=\{X_{1i},A_{1i},\cdots,X_{ti}\})$ includes all the information observed till step t. Furthermore, we also need to posit a model for the conditional expected outcome for the control option (treatment $0$), $Q_t(h_t,0)$, and the propensity function $\omega(h_{t},a_{t})$. Detailed definitions are provided in the following. Suppose there are $m_t$ number of options, and the action space $\mathcal{A}_t=\{0,1,\dots,m_t-1\}$ for each step t. With decision point $t$, we define thoes key functions as follows:
# *   Q-function:
#     For the final step $T$, 
#     $$
#     Q_T(h_T,a_{T})=E[Y|H_{T}=h_{T}, A_{T}=a_{T}],
#     $$
#     If there is a multi-stage case with total step $T>1$, for the step $t=1,\cdots,T-1$,
#     $$
#     Q_t(h_t,a_{t})=E[V_{t+1}|H_{t}=h_{t}, A_{t}=a_{t}],
#     $$
#     where 
#     $$
#     V_{t}(h_{t}) = \max_{j\in\mathcal{A}_t}Q_{t}(h_t,j)
#     $$
#     Alternatively, with the contrast function defined in the follwing,
#     $$
#     Q_t(h_t,j) = Q_t(h_t,0) + C_{tj}(h_t),\quad j=0,\dots,m_k-1,\quad t=1,\dots,T.
#     $$
# *   Contrast functions (optimal blip to zero functions)
#     $$
#     C_{tj}(h_t)=Q_t(h_t,j)-Q_t(h_t,0),\quad j=0,\dots,m_k-1,\quad t=1,\dots,T.
#     $$
# *   Propensity score
#     $$\omega_{t}(h_t,a_t)=P(A_t=a_t|H_t=h_t)$$
# *   Optimal regime
#     $$
#     d_t^{opt}(h_t)=\arg\max_{j\in\mathcal{A}_t}C_{tj}(h_t)
#     $$
# 
# 
# In the following, we would start from a simple case having only one decision point and then introduce the multistage case with multiple decision points. 
# 
# ### Single Decision Point
# 
# - **Basic Logic**: Positting models, $C_{j}(h,\psi_{j})$,$Q(h,0,\phi)$,and $\omega(h,a,\gamma)$, A-learning aims to estimate $\psi_{j}$, $\phi$, and $\gamma$ by g-estimation. With the $\hat{\psi}_{j}$ in hand, the optimal decision is directly derived.
# 
# - **Key Steps**:
#     1. Fitted a model $\omega_{1}(h_1,a_1,\gamma)$, which can be solved directly by existing approaches (i.e., logistic regression, .etc),
#     2. Substituting the $\hat{\gamma}$, we estimate the $\hat{\psi}_{j}$ and $\gamma$ by solving the euqations in Appendix A.1 jointly.      
#     2. For each individual find the optimal action $a_{i}$ such that $a_{i} = \arg\max_{j\in\mathcal{A}}C_{j}(h,\hat{\psi_{j}})$.
# 
# ### Multiple Decision Points
# 
# - **Basic Logic**: Similar to Q learning, a backward approach was proposed to find the optimized treatment regime at each decision point. 
# 
#     At Decision $T$, similar as what we did previously with single decision point, we estimate the $\psi_{Kj}$, $\phi_K$ and $\gamma_K$ by solving the eqautions in A.2 jointly, and the optimal decision at time $T$ is calculated accordingly. 
#     
#     At Decision $t=T-1,\dots,1$, we use similar trick as decision $T$, except for changing $Y$ in the estimating eqautions to some pseudo outcome $\tilde{Y}_{t+1,i}$, such that:
#     $$
#     \tilde{Y}_{ti}=\tilde{Y}_{t+1,i}+\max_{j=0,\dots,m_t-1}C_{tj}(H_{ti},\hat{\psi}_{tj})-\sum_{j=1}^{m_k-1}\mathbb{I}\{A_{ti}=j\}C_{tj}(H_{ti},\hat{\psi}_{tj}),
#     $$ 
#     where $\tilde{Y}_{T+1,i} = Y_{i}$.
#     
#     Estimating $\psi_{tj}$, $\phi_t$ and $\gamma_t$ iteratively for $t=T-1,\cdots,1$, we calculated the optimal decision at time $t$, $a_{ti}$ as
#     $$
#     a_{ti}=\arg\max_{j=0,\dots,m_t-1} C_{tj}(h_{ti};\hat{\psi}_{tj}).
#     $$
# 
# 
# - **Key Steps**: 
#     1. At the final decision point $t=T$, fitted a model $\omega_{T}(h_{T},a,\hat{\gamma}_{T})$, and estimating $\psi_{Tj}$, $\phi_{T}$ by solving the equations in A.2 jointly;
#     2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{Ti}$, and the optimal action $a_{Ti}$;
#     3. For decision point $t = T-1,\cdots, 1$,
#         1. fitted a model $\omega_{t}(h_{t},a,\hat{\gamma}_{t})$, and estimating $\psi_{tj}$, $\phi_{t}$ by solving the equations in A.3 jointly with the pseudo-outcome $\tilde{Y}_{t+1}$
#         2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{ti}$, and the optimal action $a_{ti}$;

# ## Demo

# ## Single Decision Point

# In[45]:


# A demo with code on how to use the package
from causaldm.learners import QLearning
from causaldm.test import shared_simulation
import numpy as np


# In[49]:


#prepare the dataset (dataset from the DTR book)
import pandas as pd
file = pd.read_csv("hyper.txt", sep=',')
file['Y'] = file['SBP0']-file['SBP6']
hyper = file
Y = hyper['Y']
X = hyper[['W','K','Cr','Ch']]
A = hyper['A']


# ## Multiple Decision Point

# In[54]:


# TODO: test the function & feasible set

# A demo with code on how to use the package
from causaldm.learners import QLearning
from causaldm.test import shared_simulation
import numpy as np


# In[55]:


#prepare the dataset (dataset from the DTR book)
import pandas as pd
dataMDP = pd.read_csv("dataMDP_feasible.txt", sep=',')
Y = dataMDP['Y']
X = dataMDP[['CD4_0','CD4_6','CD4_12']]
A = dataMDP[['A1','A2','A3']]


# ## References
# 1. Schulte, P. J., Tsiatis, A. A., Laber, E. B., & Davidian, M. (2014). Q-and A-learning methods for estimating optimal dynamic treatment regimes. Statistical science: a review journal of the Institute of Mathematical Statistics, 29(4), 640.
# 2. Robins, J. M. (2004). Optimal structural nested models for optimal sequential decisions. In Proceedings of the second seattle Symposium in Biostatistics (pp. 189-326). Springer, New York, NY.
# 3. Murphy, S. A. (2003). Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 65(2), 331-355.
# 4. Liang, S., Lu, W., & Song, R. (2018). Deep advantage learning for optimal dynamic treatment regime. Statistical theory and related fields, 2(1), 80-88.
# 5. Shi, C., Fan, A., Song, R., & Lu, W. (2018). High-dimensional A-learning for optimal dynamic treatment regimes. Annals of statistics, 46(3), 925.

# ## A.1
# \begin{align}
# \sum_{i=1}^n \frac{\partial C_{j}(H_{i};\psi_{j})}{\partial \psi_{j}}\{\mathbb{I}\{A_{i}=j\}-\omega(H_{i},j;\hat{\gamma})\}\times \Big\{Y_i-\sum_{j'=1}^{m-1} \mathbb{I}\{A_{i}=j'\}C_{j'}(H_{i;\psi_{j'}})-Q(H_{i},0;\phi)\Big\}=0
# \end{align}
# \begin{align}
# \sum_{i=1}^n \frac{\partial Q(H_{i},0;\phi)}{\partial \phi}\Big\{Y_i-\sum_{j'=1}^{m-1} \mathbb{I}\{A_{i}=j'\}C_{j'}(H_{i};\psi_{j'}) Q(H_{i}0;\phi)\Big\}=0
# \end{align}
# 
# ## A.2
# $$
# \begin{aligned}
# &\sum_{i=1}^n \left[\frac{\partial C_{Tj}(H_{Ti};\psi_{Tj})}{\partial \psi_{Tj}}\{\mathbb{I}\{A_{Ti}=j\}-\omega_T(H_{Ti},j;\gamma_T)\}\times \Big\{Y_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\psi_{Tj'})-Q_T(H_{Ti},0;\phi_{T})\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial Q_T(H_{Ti},0;\phi_T)}{\partial \phi_T}\Big\{Y_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\psi_{Tj'})-Q_T(H_{Ti},0;\phi_T)\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial \omega_T(H_{Ti},j;\gamma_T)}{\partial \gamma_T}\Big\{Y_i-\sum_{j'=1}^{m_T-1} \mathbb{I}\{A_{Ti}=j'\}C_{Tj'}(H_{Ti};\psi_{Tj'})-Q_T(H_{Ti},0;\phi_T)\Big\}\right]=0
# \end{aligned}
# $$
# 
# ## A.3
# $$
# \begin{aligned}
# &\sum_{i=1}^n \left[\frac{\partial C_{tj}(H_{ti};\psi_{tj})}{\partial \psi_{tj}}\{\mathbb{I}\{A_{ti}=j\}-\omega_T(H_{ti},j;\gamma_t)\}\times \Big\{\tilde{Y}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\psi_{tj'})-Q_t(H_{ti},0;\phi_{t})\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial Q_t(H_{ti},0;\phi_t)}{\partial \phi_t}\Big\{\tilde{Y}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\psi_{tj'})-Q_t(H_{ti},0;\phi_t)\Big\}\right]=0\\
# &\sum_{i=1}^n \left[\frac{\partial \omega_t(H_{ti},j;\gamma_t)}{\partial \gamma_t}\Big\{\tilde{Y}_{t+1,i}-\sum_{j'=1}^{m_t-1} \mathbb{I}\{A_{ti}=j'\}C_{tj'}(H_{ti};\psi_{tj'})-Q_t(H_{ti},0;\phi_t)\Big\}\right]=0
# \end{aligned}
# $$
# 

# In[ ]:




