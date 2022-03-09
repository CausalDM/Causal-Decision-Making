#!/usr/bin/env python
# coding: utf-8

# # Learner Name (Single/Multiple Stages/Infinite Horizon)

# In[1]:


# change direction to the main folder
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')


# ## Main Idea
# Include an overview of the learner with key concepts. 

# - **Basic Logic**: an overview of the specific algorithm used
# 
# - **Key Steps**: an abstract pseudo algorithm

# ### 1. Optimal Decision

# In[1]:


# code used to import the learner
from causaldm.learners import ALearning
from causaldm.test import shared_simulation
import numpy as np


# Demo code to find an optimal regime

# ### 2. Policy Evaluation

# Demo code to evaluate a fixed policy

# ## References
