#!/usr/bin/env python
# coding: utf-8

# # Confidence Interval in OPE
# 

# 
# ***Advantages***:
# 
# ## Main Idea
# 
# 
# ### Asymptotic distribution based CI for DRL
# 
# In addition to the approaches reviewed in Section \ref{sec:CI},  another commonly adopted CI construction  approach in statistics is to utilize the asymptotic distribution of a point estimator. 
# Although such a CI is typically only asymptotically valid, it is generally more computationally efficient than Bootstrap-based methods and tighter than concentration inequality-based CIs. 
# However, this kind of CIs for OPE is rare in the literature, due to the challenge of deriving the   asymptotic distribution for OPE point estimators. 
# 
# We begin our proposal by constructing a CI based on DRL introduced in Section \ref{sec:curse_horizon}. 
# Although the CI has not been explicitly proposed and evaluated in  \citet{kallus2019efficiently}, 
# given the derived asymptotic normal distribution for $\widehat{\eta}_{\textrm{DRL}}$, 
# a Wald-type CI for $\eta^{\pi}$ can be constructed following the standard procedure. 
# Specifically, recall that $\widehat{\eta}_{\textrm{DRL}}$ is defined as the average of $\{\psi_{i,t}\}$, an estimator of the asymptotic variance \eqref{lower_bound} can be derived as the sampling variance $\widehat{\sigma}^2=(nT-1)^{-1} \sum_{i,t} (\psi_{i,t}-\widehat{\eta}_{\textrm{DRL}})^2$ and we can prove it is consistent.  
# Then, an asymptotic $(1 - \alpha)$-CI is given by
# \begin{equation}\label{eqn:CI_DRL}
#     [\widehat{\eta}_{\textrm{DRL}} - z_{\alpha/2} (nT)^{-1/2}	\widehat{\sigma} \; , \; \widehat{\eta}_{\textrm{DRL}}+z_{\alpha/2} (nT)^{-1/2}	\widehat{\sigma}], 
# \end{equation}
# where $z_{\alpha}$ corresponds to the upper $\alpha$th quantile of a standard normal random variable.
# 
# 

# ## Demo [TODO]

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')



# ## References
# 1. Shi C, Wan R, Chernozhukov V, et al. Deeply-debiased off-policy interval estimation[C]//International Conference on Machine Learning. PMLR, 2021: 9580-9591.

# ## Note
# 

# In[ ]:




