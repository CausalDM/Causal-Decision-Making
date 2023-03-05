#!/usr/bin/env python
# coding: utf-8

# # Ooline Policy Learning in Non-Markovian Environments 
# 
# An important extension of the [Markov assumption-based RL](section:online_RL) is to the non-Markovian environment. 
# We provide a brief introduction in this chapter. 
# 
# ## Model
# 
# Recall the model we introduce in [online RL with MDP](section:online_RL). 
# In comparison, we essentially cannot assume the MA and CMIA conditions any longer, and hence need to use all historical information in decision making. 
# We make the comparison in {numref}`POMDP_comparison`. 
# 
# ```{image} POMDP_comparison.png
# :name: POMDP_comparison
# :alt: d2ope
# :width: 800px
# :align: center
# ```
# 
# The extension is valuable when either (i) the system dynamic depends on multiple or infinite lagged time points and hence it is infeasible to summarize historical information in a fixed-dimensional vector (the DTR problem that we studied in Paradigm 2), or when (ii) the underlying model is an MDP yet the state is not observable, which corresponds to the well-known Partially observable Markov decision process (POMDP). 
# Note that the model of POMDP is actually slightly different, which we summarize in {numref}`POMDP`. 
# Fortunately, unlike in the offline setting {cite:p}`shi2022off`, in the online setup, even with unobservable variables, there would be no causal bias, since the action selection does not depend on unmeasured variables. 
# 
# ```{image} POMDP.png
# :name: POMDP
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
