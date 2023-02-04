#!/usr/bin/env python
# coding: utf-8

# ## **Other Approaches**
# 

# ### **7. Generalized Random Forest**
# 
# Developed by Susan Athey, Julie Tibshirani and Stefan Wager, Generalized Random Forest [8] aims to give the solution to a set of local moment equations:
# \begin{equation}
#   \mathbb{E}\big[\psi_{\tau(s),\nu(s)}(O_i)\big| S_i=s\big]=0,
# \end{equation}
# where $\tau(s)$ is the parameter we care about and $\nu(s)$ is an optional nuisance parameter. In the problem of Heterogeneous Treatment Effect Evaluation, our parameter of interest $\tau(s)=\xi\cdot \beta(s)$ is identified by 
# \begin{equation}
#   \psi_{\beta(s),\nu(s)}(R_i,A_i)=(R_i-\beta(s)\cdot A_i-c(s))(1 \quad A_i^T)^T.
# \end{equation}
# The induced estimator $\hat{\tau}(s)$ for $\tau(s)$ can thus be solved by
# \begin{equation}
#   \hat{\tau}(s)=\xi^T\left(\sum_{i=1}^n \alpha_i(s)\big(A_i-\bar{A}_\alpha\big)^{\otimes 2}\right)^{-1}\sum_{i=1}^n \alpha_i(s)\big(A_i-\bar{A}_\alpha\big)\big(R_i-\bar{R}_\alpha\big),
# \end{equation}
# where $\bar{A}_\alpha=\sum \alpha_i(s)A_i$ and $\bar{R}_\alpha=\sum \alpha_i(s)R_i$, and we write $v^{\otimes 2}=vv^T$.
# 
# Notice that this formula is just a weighted version of R-learner introduced above. However, instead of using ordinary kernel weighting functions that are prone to a strong curse of dimensionality, GRF uses an adaptive weighting function $\alpha_i(s)$ derived from a forest designed to express heterogeneity in the specified quantity of interest. 
#     
# To be more specific, in order to obtain $\alpha_i(s)$, GRF first grows a set of $B$ trees indexed by $1,\dots,B$. Then for each such tree, define $L_b(s)$ as the set of training samples falling in the same ``leaf" as x. The weights $\alpha_i(s)$ then capture the frequency with which the $i$-th training example falls into the same leaf as $s$:
# \begin{equation}
#   \alpha_{bi}(s)=\frac{\boldsymbol{1}\big(\{S_i\in L_b(s)\}\big)}{\big|L_b(s)\big|},\quad \alpha_i(s)=\frac{1}{B}\sum_{b=1}^B \alpha_{bi}(s).
# \end{equation}
# 
# To sum up, GRF aims to leverage the splitting result of a series of trees to decide the ``localized” weight for HTE estimation at each point $x_0$. Compared with kernel functions, we may expect tree-based weights to be more flexible and better performed in real settings.
# 
# 

# In[1]:


# import related packages
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
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



# The generalized random forest (GRF) approach has been implemented in package *grf* for R and C++, and *econml* in python. Here we implement the package of *econml* for a simple illustration.

# In[ ]:


# import the package for Causal Random Forest
get_ipython().system(' pip install econml')


# In[ ]:


# A demo code of Causal Random Forest
from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
est = CausalForest(criterion='het', n_estimators=400, min_samples_leaf=5, max_depth=None,
                    min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                    min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=.45,
                    warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                    honest=True, verbose=0, n_jobs=-1, random_state=1235)


est.fit(data_behavior.iloc[:,0:2], data_behavior['A'], data_behavior['R'])

HTE_GRF = est.predict(data_behavior.iloc[:,0:2], interval=False, alpha=0.05)
HTE_GRF = HTE_GRF.flatten()


# In[ ]:


print("Generalized Random Forest:  ",HTE_GRF[0:8])
print("true value:                 ",HTE_true[0:8].to_numpy())


# Causal Forest performs just okay in this example.

# In[ ]:


Bias_GRF = np.sum(HTE_GRF-HTE_true)/n
Variance_GRF = np.sum((HTE_GRF-HTE_true)**2)/n
print("The overall estimation bias of Generalized Random Forest is :     ", Bias_GRF, ", \n", "The overall estimation variance of Generalized Random Forest is :",Variance_GRF ,". \n")


# ### **8. Dragon Net**
# 
# 
# 
# 

# In[ ]:





# ## References
# 
# 8. Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.
