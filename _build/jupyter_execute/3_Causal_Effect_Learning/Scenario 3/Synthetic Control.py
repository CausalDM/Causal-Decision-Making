#!/usr/bin/env python
# coding: utf-8

# ### **Synthetic Control**
# 
# Unlike the Difference-in-Differences (DiD) method, synthetic control is a frequently employed technique when dealing with datasets where there is a significant imbalance between the number of control units and treated units. DiD methods typically demand a high degree of comparability between the treated and control groups to establish the critical "parallel trend" assumption. However, this assumption becomes challenging to fulfill when the dataset contains only a limited number, or even just a single, treated unit, often due to issues related to data collection or funding constraints. In this situation, synthetic control aims to reweight the substantial information in control group to provide another perspective to learn the conterfactuals for treated unit(s).
# 
# To illustrate the basic idea of synthetic control, we suppose that there are $N$ units and $T$ time periods in total, and denote $Y_{it}$ as the outcome for unit $i$ in period $t$. Without the loss of generality, e suppose the first $N_{\text{co}}$ units are in the control group, who have no possibility to be exposed to treatment at any time. The rest $N_{\text{tr}} := N-N_{\text{co}}$ units belong to the treated group, and will receive treatment starting from period $T_0 +1$.
# 
# 
# There are two main steps in synthetic control methods: 
# 
# **Step 1:** Calculate the weights $\hat{\omega}_i^{\text{sdid}}$ that align pre-exposure trends in the outcome of control units for treated units；\
# 
# \begin{equation}
#     \hat{Y}_{it} = \hat{\omega}_{i0} + \sum_{j=1}^{N_{\text{co}}}\hat{\omega}_{ij} Y_{jt}, \qquad \forall i\in\{N_{\text{co}+1},\dots, N\}, \forall t\in \{1,\dots,T\},
# \end{equation}
# where 
# \begin{equation}
# \hat{\omega}_i = \arg\min_{\omega} \sum_{1\leq t\leq T_0} \bigg(Y_{it} - \omega_{i0} -\sum_{j=1}^{N_{\text{co}}} \omega_{ij} Y_{jt}\bigg)^2
# \end{equation}
# 
# 
# **Step 2:** Use the weights to estimate the post-exposure conterfactuals in causal effect estimation.
# 
# 
# 

# In[1]:


# 


# In[ ]:





# ## References
# 
# 1. Abadie, A., Diamond, A., and Hainmueller, J. (2010), “Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California’s Tobacco Control Program,” Journal of the American Statistical Association, 105, 493–505. [2068,2069,2070,2071]
# 
# 2. Li, K. T. (2020), “Statistical Inference for Average Treatment Effects Esti-mated by Synthetic Control Methods,”Journal of the American StatisticalAssociation, 115, 2068–2083. [1716]

# In[ ]:




