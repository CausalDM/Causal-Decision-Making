#!/usr/bin/env python
# coding: utf-8

# # Ooline Policy Learning in Non-Markovian Environments 
# 
# An important extension of the [Markov assumption-based RL](section:online_RL) is to the non-Markovian environment. 
# 
# 
# The extension is valuable when either (i) the system dynamic depends on multiple or infinite lagged time points and hence it is infeasible to summarize historical information in a fixed-dimensional vector (the DTR problem that we studied in Paradigm 2). 
# 
# ## Model
# 
# No that assumption. 
# 
# for DTR
# 
# 
# ```{image} POMDP_comparison.png
# :alt: d2ope
# :width: 800px
# :align: center
# ```
# 
# No unmeasured confounder issue. 
# 
# shi2022off
# 
# for POMDP
# 
# ```{image} POMDP.png
# :alt: d2ope
# :width: 500px
# :align: center
# ```
# 
# ## Policy learning
# 
# It is challenging to learn the optimal policy in a POMDP due to the model complexity. Below, we mainly introduce three classes of approaches. 
# 
# 1. When the horizon is short (e.g., 2 or 3), it is still feasible to learn a time-dependent policy that directly utilizes the vector of all historical information for decision making. This is the online version of the DTR problem and was recently studied in {cite:t}`hu2020dtr`. However, when the horiozn is longer, such an approach quickly becomes computationally infeasible and certain dimension reduction method is then needed. 
# 2. In POMDP, a classic approach is to infer the underlying state via the history information, and then use the inferred state distribution as a *belief state* for decision making {cite:p}`spaan2012partially`. 
# 3. In recent years, there is also a growing literature on applying memory-based NN architecture for directly learning the policy from a sequence of history transition tuples {cite:p}`zhu2017improving, meng2021memory`. 

# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
