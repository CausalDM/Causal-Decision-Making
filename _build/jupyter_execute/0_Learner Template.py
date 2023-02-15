#!/usr/bin/env python
# coding: utf-8

# # Learner Name (Single/Multiple Stages/Infinite Horizon)
# 
# ## Main Idea
# An overview of the learner include:
# 1. a brief introduction of the learner 
# 2. evolution of the learner (i.e. when it is first developed, any alternative extensions?)
# 3. Application situations: Describe the data structure that can be analyzed, and make a connection between the real application situations (mentioned in the Motivating Examples) and the learner (i.e., when can we use the learner) 
# 4. the advantage of the learner
# 
# ## Algorithm Details
# a detailed description of the learner with clear definitions of key concepts.
# 
# ## Key Steps
# an abstract pseudo-code for policy learning and policy evaluation
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# code used to import the learner
from causaldm.learners import ALearning
from causaldm.test import shared_simulation
import numpy as np


# In[1]:


#Demo code to find an optimal regime, 
#using the appropriate real data described in the motivating examples


# **Interpretation:** A sentence to include the analysis result: the estimated optimal regime is...

# ### 2. Policy Evaluation

# In[2]:


#Demo code to evaluate a fixed policy,
#using the appropriate real data described in the motivating examples


# **Interpretation:** A sentence to include the analysis result: the estimated value of the policy ... is ...

# ## References
