#!/usr/bin/env python
# coding: utf-8

# # Deep Jump Learner for Continuous Actions
# 
# 
# Despite the popularity of developing methods for policy optimization and evaluation with discrete treatments, less attention has been paid to continuous action domains such as personalized dose-finding and dynamic pricing. We review the classical solutions for continous domains and detail the **deep jump learner** for continuous actions. 
# 
# ***Application situations***: 
#     
# 1. Continuous treatment settings
# 2. Discontinuous or continuous value function 
# 
# ***Advantage of the learner***:
# 
# 1. Policy optimization: 
#     - Return an individualized interval-valued decision rule that recommends an interval of treatment options for each individual;
#     - More flexible to implement in practice.
#     
# 2. Policy evaluation:
#     - A faster convergence rate than the kernel-based methods for discontinuous value function;
#     - No need to select kernel bandwidth.
# 
# ## Main Idea
# 
# ### Overview
# 
# To handle **continuous treatment settings**, we detail the jump interval-learning to develop an individualized interval-valued decision rule (I2DR) that maximizes the expected outcome. Unlike individualized decision rules (IDRs) that recommend a single treatment, the newly proposed I2DR yields an interval of treatment options for each individual, making it more flexible to implement in practice. On the other hand, for off-policy evaluation, we review the deep jump learning, of which the key ingredient lies in adaptively discretizing the treatment space using deep discretization, by leveraging deep learning and multi-scale change point detection. This allows us to apply existing off-policy evaluation methods in discrete treatments to handle continuous treatments, and overcome the limitations of kernel-based methods in the literature and handle both **continuous/discontinuous value function**. 
#  
# ### Difficulties in Continuous Actions
# 
# Let the observed offline datasets as $\{(X_i,A_i,Y_i)\}_{1\le i\le n}$ and $n$ as the total sample size. A decision rule or policy $d: \mathcal{X} \to \mathcal{A}$. We use $b$ to denote the propensity score and $p(\bullet|x)$ denotes the probability density function of $A$ given $X=x$. Define the $Q$-function as $Q(x, a) = E(Y|X=x,A=a).$ The doubly robust (DR) estimator of value $V(d) = E [Q\{X, d(X)\}]$ for discrete treatments is
# \begin{eqnarray}\label{eqn:DR}
# \frac{1}{n}\sum_{i=1}^n \psi(O_i,d,\widehat{Q},\widehat{p})=\frac{1}{n}\sum_{i=1}^n \left[\widehat{Q}\{X_i,d(X_i)\}+\frac{\mathbb{I}\{A_i=d(X_i)\}}{\widehat{p}(A_i|X_i)}\{Y_i-\widehat{Q}(X_i,A_i)\}\right],
# \end{eqnarray} 
# where $\mathbb{I}(\bullet)$ denotes the indicator function, $\widehat{Q}$ and $\widehat{p}(a|x)$ denote some estimators for the conditional mean function $Q$ and the propensity score $b$, respectively. In continuous treatment domains, the indicator function $\mathbb{I}\{A_i=d(X_i)\}$ equals zero almost surely. Consequently, naively applying the DR method yields a plug-in estimator $\sum_{i=1}^n \widehat{Q}\{X_i,d(X_i)\}/n$. To address this concern, the kernel-based methods proposed to replace the indicator function with a kernel function $K[\{A_i-d(X_i)\}/h]$, i.e.,
# \begin{eqnarray}\label{eqn:kernel}
# 	\frac{1}{n}\sum_{i=1}^n \psi_h(O_i,d,\widehat{Q},\widehat{p})=\frac{1}{n}\sum_{i=1}^n \left[\widehat{Q}\{X_i,d(X_i)\}+\frac{K\{{A_i-d(X_i)\over h}\}}{\widehat{p}(A_i|X_i)}\{Y_i-\widehat{Q}(X_i,A_i)\}\right].
# \end{eqnarray}
# Here, the bandwidth $h$ represents a trade-off. The variance of the resulting value estimator decays with $h$. Yet, its bias increases with $h$. This method requires the expected second derivative of the function $Q(x,a)$ exists, and thus $Q(x,a)$ needs to be a **smooth** function of $a$.  
# 
# 
# ### Details of Deep Jump Learner 
# To overcome these difficulties, the deep jump learner proposes to adaptively discretizing the treatment space. We define $\mathcal{B}(m)$ as the set of discretizations $\mathcal{D}$ such that each interval $\mathcal{I}\in \mathcal{D}$ corresponds to a union of some of the $m$ initial intervals.  Each discretization $\mathcal{D}\in \mathcal{B}(m)$ is associated with a set of functions $\{q_{\mathcal{I}}\}_{\mathcal{I}\in \mathcal{D}}$. We model these $q_{\mathcal{I}}$ using deep neural networks (DNNs), to capture the complex dependence between the outcome and features.  Thus, $\widehat{\mathcal{D}}$ can be estimated by solving 
# \begin{eqnarray}\label{eqn:optimize}
#  \left(\widehat{\mathcal{D}},\{\widehat{q}_{\mathcal{I}}:\mathcal{I}\in \widehat{\mathcal{D}} \}\right)=argmin_{\left(\substack{\mathcal{D}\in \mathcal{B}(m),\\ \{q_{\mathcal{I}}\in \mathcal{Q}_{\mathcal{I}}: \mathcal{I}\in \mathcal{D} \} }\right)}\left(\sum_{\mathcal{I}\in \mathcal{D}} \left[ {1\over n} \sum_{i=1}^{n} \mathbb{I}(A_i\in \mathcal{I})  \big\{Y_i - q_{\mathcal{I}}(X_i) \big\}^2\right]+\gamma_n |\mathcal{D}| \right),
# \end{eqnarray}
# for some regularization parameter $\gamma_n$ and some function class of DNNs $\mathcal{Q}_{\mathcal{I}}$.
# 
# ***Policy Optimization***:
# 
# Given $\widehat{\mathcal{D}}$ and $\{\widehat{q}_{\mathcal{I}}:\mathcal{I}\in \widehat{\mathcal{D}} \}$, to maximize the expected outcome of interest, our proposed I2DR is then given by
# \begin{eqnarray}\label{I2DR}
# 	\widehat{d}(x)=argmax_{\mathcal{I}\in \widehat{\mathcal{D}}}  \widehat{q}_{\mathcal{I}}(x),\,\,\,\,\,\,\,\,\forall x\in \mathbb{X}.
# \end{eqnarray}
# When the argmax in the above equation is not unique, $\widehat{d}(\cdot)$ outputs the interval that contains the smallest treatment.  
# 
# ***Policy Evaluation***:
# 
# Given $\widehat{\mathcal{D}}$ and $\{\widehat{q}_{\mathcal{I}}:\mathcal{I}\in \widehat{\mathcal{D}} \}$, we apply the DR estimator to derive the value estimate for any target policy of interest $d$, i.e., 
# \begin{eqnarray}\label{value_djqe}
# 	\widehat{V}^{DR}(d)={1\over n} \sum_{\mathcal{I}\in \widehat{\mathcal{D}}}  \sum_{i=1}^{n} \left( \mathbb{I}\{d(X_i)\in \mathcal{I}\}  \left[  {\mathbb{I}(A_i\in \mathcal{I})\over  \widehat{p}_\mathcal{I}(X_i)}\big\{Y_i - \widehat{q}_\mathcal{I}(X_i) \big\} + \widehat{q}_\mathcal{I}(X_i) \right]\right),
# \end{eqnarray} 
# where $\widehat{p}_{\mathcal{I}}(x)$ is some estimator of the generalized propensity score function $p(A\in \mathcal{I}|X=x)$. 
# 
# 
# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Policy Learning

# In[1]:


# To find the optimal I2DR by the deep jump learner on the Real Data 
# of Warfarin Dosing (20 features)

# code used to import the learner  
from DJL_opt import *

# real data generator
data_gen = data_generator.RealDataGenerator(file_name='real_envir.pickle') 
  
n = 300 # total number of visit 

# deep jump learner

DJL_partition, DJL_agent = DJLearn_opt(data_gen, n, seed=2333, mlp_max_iter=50)


# **Interpretation:** The estimated best partition of the continuous action space is [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.333, 1], with spent time 0.5 minute. 

# In[9]:


#Demo code to find an optimal regime
DJL_agent.opt_policy(DJL_partition, np.array(DJL_agent.train_data['xt'][0]).reshape(1,-1))


# **Interpretation:** The estimated optimal regime (I2DR) for 0th patient is [0.333, 1.0].

# ### 2. Policy Evaluation

# In[3]:


# To evaluate the deep jump learner on the Real Data of Warfarin Dosing (20 features)

# code used to import the learner  
from DJL_eval import *

# real data generator
data_gen = data_generator.RealDataGenerator(file_name='real_envir.pickle') 

# policy to be evaluate
def pi_evaluate(context):
    act_list = np.linspace(0, 1, 100)
    x_max = np.max(np.array([data_gen.org_data.iloc[i]['xt'] for i in range(len(data_gen.org_data))]), 0)
    x_min = np.min(np.array([data_gen.org_data.iloc[i]['xt'] for i in range(len(data_gen.org_data))]), 0)
    val = []
    for act in act_list:
        val.append(data_gen.regr_mean.predict(np.append(np.array(context),np.array(act)).reshape(1, len(context)+1))[0])
    return act_list[val.index(max(val))] 

n = 100 # total number of visit 

# policy evaluation

DJLearn_eval(data_gen, n, pi_evaluate, seed=2333, mlp_max_iter=10)


# **Interpretation:** The analysis result: the estimated value of the calibrated optimal policy under sample size $n=100$ and $p=20$ features is -0.1.

# ## References
# 
# [1] Cai, H., Shi, C., Song, R., & Lu, W. (2021). Deep jump learning for off-policy evaluation in continuous treatment settings. Advances in Neural Information Processing Systems, 34.
# 
# [2] Cai, H., Shi, C., Song, R., & Lu, W. (2021). Jump Interval-Learning for Individualized Decision Making. arXiv preprint arXiv:2111.08885.
# 
# [3] Zhu, L., Lu, W., Kosorok, M. R., & Song, R. (2020, August). Kernel assisted learning for personalized dose finding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 56-65).
# 
# [4] Kallus, N., & Zhou, A. (2018, March). Policy evaluation and optimization with continuous treatments. In International conference on artificial intelligence and statistics (pp. 1243-1251). PMLR.
