#!/usr/bin/env python
# coding: utf-8

# # Outcome Weighted Learning

# In[1]:


# After we publish the package, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## Main Idea
# 
# A natural idea for policy learning is to stay close to the behaviour policy in those areas where it performs well. 
# Outcome Weighted Learning (OWL) shares similar ideas. 
# OWL was first proposed in [1] under the binary treatment case, and extended in [3] to allow multiple treatments. 
# The foundation of OWL is built on the relationship that, maximizing $V(\pi)$ is equivalent to solve
# 
# \begin{align}
#     \text{arg min}_{\pi} \mathbb{E}\Big[ \frac{Y_i}{b(A_i|X_i)}\mathbb{I}(A_i \neq \pi(X_i))\Big]. 
# \end{align}
# 
# When $Y_i$ is non-negative, this goal corresponds to the objective function of a cost-sensitive classification problem with ${Y_i}/{b(A_i|X_i)}$ as the weight, 
# $A_i$ as the true label, 
# and $\pi$ as the classifier to be learned. 
# Intuitively, a large value of $Y_i$ implies a large weight that encourages the policy to take the same action as observed; 
# while a small reward has the opposite effect. 
# This is why the estimator is called *outcome weighted*. 
# $b(A_i|X_i)$ is used to remove the sampling bias. 
# 
# Based on the relationship, OWL has the following key steps:
# 1. Estimate the weight of data point $i$ as $w_i = (Y_i + c) / b(A_i|X_i)$
#     1. Here $c$ is a constant such that $Y_i + c$ are all non-negative, which is required to use cost-sensitive classification algorithms. Note that such a shift will not affect the solution of (1), though with finite sample it may cause instability. 
#     2. With binary treatment, we implement the approach in [2] to estimate a shift constant and hence the algorithm is adaptive. 
# 2. Solve the policy with a user-specified cost-sensitive classifier. The theory is developed based on SVM.

# ## When should I use OWL?
# * TBD

# ## Demo

# In[1]:


# A demo with code on how to use the package
from causaldm.learners.CPL13.disc import OWL
from causaldm.test import shared_simulation
from causaldm.test import OWL_simu
from causaldm.metric import metric
import numpy as np


# In[15]:


# generate sample data
instance = OWL_simu.generate_test_case(setup = 'case1', N = 1000, seed = 0, p = 5, sigma = 1)
X, A, Y = instance['XAY']


# In[16]:


# initialize the learner
owl = OWL.OutcomeWeightedLearning()
# specify the classifier you would like to use
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
clf = SVC(kernel='linear') # fit_intercept = True, 
# Cs = np.logspace(-6, -1, 10)
# clf = GridSearchCV(estimator=clf, param_grid=dict(C=Cs),
#                    n_jobs=-1)

# specify the assignment_prob probability, if your data is from an experiment 
assignment_prob = np.ones(len(A)) / 0.5

# train the policy
owl.train(X, A, Y, classifier = clf, assignment_prob = assignment_prob)


# In[19]:


# recommend action
owl.recommend_action(X)[:10]


# ## Sparse OWL
# 
# In many applications, we have a large number of features. [4] extend OWL to these use cases by assuming a sparsity structure, i.e., most features do not have effect in the policy. Under this assumption, [4] develops a penalized policy learner and proved its consistency as well as asymptotic distribution. Notably, one can achieve variable selection in the meantime. 
# 
# TBD

# In[ ]:





# ## References
# 1. Zhao, Yingqi, et al. "Estimating individualized treatment rules using outcome weighted learning." Journal of the American Statistical Association 107.499 (2012): 1106-1118.
# 2. Liu, Ying, et al. "Augmented outcome‐weighted learning for estimating optimal dynamic treatment regimens." Statistics in medicine 37.26 (2018): 3776-3788.
# 3. Lou, Zhilan, Jun Shao, and Menggang Yu. "Optimal treatment assignment to maximize expected outcome with multiple treatments." Biometrics 74.2 (2018): 506-516.
# 4. Song, Rui, et al. "On sparse representation for optimal individualized treatment selection with penalized outcome weighted learning." Stat 4.1 (2015): 59-68.

# ## A1: Derivations
# 
# \begin{align*}
# V(\pi)
# &= \mathbb{E}_{A_i \sim b(X_i)}\Big[ \frac{\mathbb{I}(A_i = \pi(X_i))}{b(A_i|X_i)}Y_i\Big]\\
# &= \mathbb{E}_{A_i \sim b(X_i)}\Big[ \frac{1 - \mathbb{I}(A_i \neq \pi(X_i))}{b(A_i|X_i)}Y_i\Big]\\
# &= \text{const} - \mathbb{E}_{A_i \sim b(X_i)}\Big[ \frac{\mathbb{I}(A_i \neq \pi(X_i))}{b(A_i|X_i)}Y_i\Big]\\
# &= \text{const} - \mathbb{E}_{A_i \sim b(X_i)}\Big[ \frac{Y_i}{b(A_i|X_i)}\mathbb{I}(A_i \neq \pi(X_i))\Big]. 
# \end{align*}
# 
