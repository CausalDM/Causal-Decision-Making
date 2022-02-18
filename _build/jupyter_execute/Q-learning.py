#!/usr/bin/env python
# coding: utf-8

# # Q-Learning

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## Main Idea
# 
# Q-learning is a classic method of Reinforcement Learning. Early in 2000, it was adapted to decision-making problems[1] and kept evolving with various extensions, such as penalized Q-learning [2]. In the following, we would start from a simple case having only one decision point and then introduce the multistage case with multiple decision points. Note that, we assume the action space is either **binary** (i.e., 0,1) or **multinomial** (i.e., A,B,C,D), and the outcome of interest Y is **continuous** and **non-negative**, where the larger the $Y$ the better. 
# 
# ### Single Decision Point
# 
# - **Application Situation**: Suppose we have a dataset containing observations from $N$ individuals. For each individual $i$, we have $\{\mathbf{X}_{i},A_{i},Y_{i}\}$, $i=1,\cdots,N$. $\mathbf{X}_{i}$ includes the feature information, $A_{i}$ is the action taken, and $Y_{i}$ is the observed reward received. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received. In other words, by training a model with the observed dataset, we want to find an optimal policy that can help us determine the optimal action for each individual to optimize the reward.
# 
# - **Basic Logic**: Q-learning with a single decision point is mainly a regression modeling problem, as the major component is to find the relationship between $Y$ and $\{X,A\}$. Let's first define a Q-function, such that
# \begin{align}
#     Q(x,a) = E(Y|X=x, A=a).
# \end{align} Then, to find the optimal policy is equivalent to solve
# \begin{align}
#     \text{arg max}_{\pi}Q(x_{i},\pi(x_{i})).
# \end{align} 
# 
# - **Key Steps**:
#     1. Fitted a model $\hat{Q}(x,a,\hat{\beta})$, which can be solved directly by existing approaches (i.e., OLS, .etc),
#     2. For each individual find the optimal action $a_{i}$ such that $a_{i} = \text{arg max}_{a}\hat{Q}(x_{i},a,\hat{\beta})$.
# 
# ### Multiple Decision Points
# 
# - **Application Situation**: Suppose we have a dataset containning observations from $N$ individuals. For each individual $i$, the observed data is structured as follows
# \begin{align}
# (X_{1i},A_{1i},\cdots,X_{Ti},A_{Ti},Y), i=1,\cdots, N.
# \end{align} Let $h_{ti}=\{X_{1i},A_{1i},\cdots,X_{ti}\})$ includes all the information observed till step t. The target of Q-learning is to find an optimal policy $\pi$ that can maximize the expected reward received at the end of the final decision point $T$. In other words, by training a model with the observed dataset, we want to find an optimal policy that can help us determine the optimal sequence of actions for each individual to optimize the reward.
# 
# - **Basic Logic**: For multistage cases, we apply a backward iterative approach, which means that we start from the final decision point T and work our way backward to the initial decision point. At the final step $T$, it is again a standard regression modeling problem that is the same as what we did for the single decision point case. Particularly, we posit a model $Q_{T}(h_{T},a_{T})$ for the observed outcome $Y$, and then the optimal policy at step $T$ is derived as $\text{arg max}_{\pi_{T}}Q_{T}(h_{T},\pi_{T}(h_{T}))$. For the decision point $T-1$ till the decision point $1$, a new term is introduced, which is the pseudo-outcome $\tilde{Y}_{t}$.
# \begin{align}
# \tilde{Y}_{t} = \text{max}_{\pi_{t}}\hat{Q}_{t}(h_{t},\pi_{t}(h_{t}),\hat{\beta}_{t})
# \end{align} By doing so, the pseudo-outcome taking the **delayed effect** into account to help explore the optimal policy. Then, for each decision point $t<T$, with the $\tilde{Y}_{t+1}$ calculated, we repeat the regression modeling step for $\tilde{Y}_{t+1}$. After obtaining the fitted model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$, the optimal policy is obtained as $\text{arg max}_{\pi_{t}}Q_{t}(h_{t},\pi_{t}(h_{t}))$.
# 
# - **Key Steps**: 
#     1. At the final decision point $t=T$, fitted a model $\hat{Q}_{T}(h_{T},a_{T},\hat{\beta}_{T})$ for the observed outcome $Y$;
#     2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{Ti}=\text{max}_{\pi}\hat{Q}_{T}(h_{Ti},\pi(h_{Ti}),\hat{\beta}_{T})$, and the optimal action $a_{Ti}=\text{arg max}_{a}\hat{Q}_{T}(h_{Ti},a,\hat{\beta}_{T})$;
#     3. For decision point $t = T-1,\cdots, 1$,
#         1. fitted a model $\hat{Q}_{t}(h_{t},a_{t},\hat{\beta}_{t})$ for the pseudo-outcome $\tilde{Y}_{t+1}$
#         2. For each individual $i$, calculated the pseudo-outcome $\tilde{Y}_{ti}=\text{max}_{\pi}\hat{Q}_{t}(h_{ti},\pi(h_{ti}),\hat{\beta}_{t})$, and the optimal action $a_{ti}=\text{arg max}_{a}\hat{Q}_{t}(h_{ti},a,\hat{\beta}_{t})$;

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


# In[50]:


# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
# If want to include all the variable in X and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~Ch+K+A+Ch*A+K*A",
              'action_space':{'A':[0,1]}}]

# train the policy
QLearn.train(X, A, Y, model_info, T=1)


# In[51]:


# recommend action
QLearn.recommend(X, A, Y, T=1).head()


# In[52]:


# get the estimated value of the optimal regime
QLearn.estimate_value(X,A)


# In[53]:


# Fitted Model
QLearn.fitted_model[0].params


# In[8]:


# generate sample data
from causaldm.test import OWL_simu
import pandas as pd
import statsmodels.api as sm
instance = OWL_simu.generate_test_case(setup = 'case1', N = 1000, seed = 0, p = 5, sigma = 1)
X, A, Y = instance['XAY']
X = pd.DataFrame(X);A = pd.DataFrame(A);Y = pd.DataFrame(Y)

# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
model_info = [{"model": "Y~.", 
              'action_space':{'A':[0,1]}}]

# train the policy
QLearn.train(X, A, Y, model_info, T=1)


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


# In[56]:


# initialize the learner
QLearn = QLearning.QLearning()
# specify the model you would like to use
# If want to include all the variable in X and A with no specific model structure, then use "Y~."
# Otherwise, specify the model structure by hand
# Note: if the action space is not binary, use C(A) in the model instead of A
model_info = [{"model": "Y~CD4_0+A1+CD4_0*A1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+A2+CD4_6*A2",
              'action_space':{'A2':[0,1]}},
             {"model": "Y~CD4_0+CD4_6+CD4_12+A3+CD4_12*A3",
              'action_space':{'A3':[0,1]}}]

# train the policy
QLearn.train(X, A, Y, model_info, T=3)


# In[58]:


# recommend action
QLearn.recommend(X, A, Y, T=1).head()


# In[59]:


# get the estimated value of the optimal regime
QLearn.estimate_value(X,A)


# ## References
# 1. Murphy, S. A. (2005). A generalization error for Q-learning.
# 2. Song, R., Wang, W., Zeng, D., & Kosorok, M. R. (2015). Penalized q-learning for dynamic treatment regimens. Statistica Sinica, 25(3), 901.
