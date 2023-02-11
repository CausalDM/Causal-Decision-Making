#!/usr/bin/env python
# coding: utf-8

# ### **8. Dragon Net**
# 
# 
# 
# 

# In[1]:


# The code is available at https://github.com/claudiashi57/dragonnet


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install scikit-uplift')


# In[3]:


# import related packages
from causaldm._util_causaldm import *


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





# ## References
# 
# 8. Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.
