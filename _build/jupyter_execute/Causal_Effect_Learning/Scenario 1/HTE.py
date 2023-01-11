#!/usr/bin/env python
# coding: utf-8

# # Heterogeneous Treatment Effect Estimation (Single Stage)
# In the previous section, we've introduced the estimation of average treatment effect, where we aims to estimate the difference of potential outcomes by executing action $A=1$ v.s. $A=0$. That is, 
# \begin{equation*}
# \text{ATE}=\mathbb{E}[R(1)-R(0)].
# \end{equation*}
# 
# In this section, we will focus on the estimation of heterogeneous treatment effect (HTE), which is also one of the main focuses in causal inference.
# 
# 
# 
# ## Main Idea
# Let's first consider the single stage setup, where the observed data can be written as a state-action-reward triplet $\{S_i,A_i,R_i\}_{i=1}^n$ with a total of $n$ trajectories. Heterogeneous treatment effect, as we can imagine from its terminology, aims to measure the heterogeneity of the treatment effect for different subjects. Specifically, we define HTE as $\tau(s)$, where
# \begin{equation*}
# \tau(s)=\mathbb{E}[R(1)-R(0)|S=s],
# \end{equation*}
# 
# where $S=s$ denotes the state information of a subject. 
# 
# The estimation of HTE is widely used in a lot of real cases such as precision medicine, advertising, recommendation systems, etc. For example, in adversiting system, the company would like to know the impact (such as annual income) of exposing an ad to a group of customers. In this case, $S$ contains all of the information of a specific customer, $A$ denotes the status of ads exposure ($A=1$ means exposed and $A=0$ means not), and $R$ denotes the reward one can observe when assigned to policy $A$. 
# 
# Suppose the ad is a picture of a dress that can lead the customers to a detail page on a shopping website. In this case, females are more likely to be interested to click the picture and look at the detail page of a dress, resulting in a higher conversion rate than males. The difference of customers preference in clothes can be regarded as the heterogeneity of the treatment effect. By looking at the HTE for each customer, we can clearly estimate the reward of ads exposure from a granular level. 
# 
# Another related concept is conditional averge treatment effect, which is defined as
# \begin{equation*}
# \text{CATE}=\mathbb{E}[R(1)-R(0)|Z],
# \end{equation*}
# 
# where $Z$ is a collection of states with some specific characsteristics. For example, if the company is interested in the treatment effect of exposing the dress to female customers, $Z$ can be defined as ``female", and the problem can be addressed under the structure CATE estimation.
# 
# 
# 
# ## Different approaches in single-stage HTE estimation
# Next, let's briefly summarize some state-of-the-art approaches in estimating the heterogeneous treatment effect. 
# 

# ### **1. S-learner**
# 
# 
# The first estimator we would like to introduce is the S-learner, also known as a ``single learner". This is one of the most foundamental learners in HTE esitmation, and is very easy to implement.
# 
# Under three common assumptions in causal inference, i.e. (1) consistency, (2) no unmeasured confounders (NUC), (3) positivity assumption, the heterogeneous treatment effect can be identified by the observed data, where
# \begin{equation*}
# \tau(s)=\mathbb{E}[R|S,A=1]-\mathbb{E}[R|S,A=0].
# \end{equation*}
# 
# The basic idea of S-learner is to fit a model for $\mathbb{E}[R|S,A]$, and then construct a plug-in estimator based on the expression above. Specifically, the algorithm can be summarized as below:
# 
# **Step 1:**  Estimate the combined response function $\mu(s,a):=\mathbb{E}[R|S=s,A=a]$ with any regression algorithm or supervised machine learning methods;
# 
# **Step 2:**  Estimate HTE by 
# \begin{equation*}
# \hat{\tau}_{\text{S-learner}}(s)=\hat\mu(s,1)-\hat\mu(s,0).
# \end{equation*}
# 
# 
# 

# In[1]:


{
    "tags": [
        "hide-cell"
    ];
    # import related packages
    from matplotlib import pyplot as plt;
    from lightgbm import LGBMRegressor;
    from causaldm._util_causaldm import *;
}


# In[ ]:


# Get data
S,A,R = get_data(target_col = 'spend', binary_trt = True)
# S-learner
np.random.seed(1)
S_learner = LGBMRegressor(max_depth=3)
SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, R)


# In[ ]:


S_learner_HTE = S_learner.predict(np.hstack((S.to_numpy(),np.ones(len(A)).reshape(-1,1)))) - S_learner.predict(np.hstack((S.to_numpy(),np.zeros(len(A)).reshape(-1,1))))

print(S_learner_HTE[0:8])


# In[ ]:


np.sum(S_learner_HTE)/578


# 
# ### **2. T-learner**
# The second learner is called T-learner, which denotes ``two learners". Instead of fitting a single model to estimate the potential outcomes under both treatment and control groups, T-learner aims to learn different models for $\mathbb{E}[R(1)|S]$ and $\mathbb{E}[R(0)|S]$ separately, and finally combines them to obtain a final HTE estimtor.
# 
# Define the control response function as $\mu_0(s)=\mathbb{E}[R(0)|S=s]$, and the treatment response function as $\mu_1(s)=\mathbb{E}[R(1)|S=s]$. The algorithm of T-learner is summarized below:
# 
# **Step 1:**  Estimate $\mu_0(s)$ and $\mu_1(s)$ separately with any regression algorithms or supervised machine learning methods;
# 
# **Step 2:**  Estimate HTE by 
# \begin{equation*}
# \hat{\tau}_{\text{T-learner}}(s)=\hat\mu_1(s)-\hat\mu_0(s).
# \end{equation*}
# 
# 

# In[ ]:





# ### **3. X-learner**
# Next, let's introduce the X-learner. As a combination of S-learner and T-learner, the X-learner can use information from the control(treatment) group to derive better estimators for the treatment(control) group, which is provably more efficient than the above two.
# 
# The basic
# 
# 
# **Step 1:**  Estimate $\mu_0(s)$ and $\mu_1(s)$ separately with any regression algorithms or supervised machine learning methods (same as T-learner);
# 
# 
# **Step 2:**  Obtain the imputed treatment effects for individuals
# \begin{equation*}
# \tilde{\Delta}_i^1:=R_i^1-\hat\mu_0(S_i^1), \quad \tilde{\Delta}_i^0:=R_i^0-\hat\mu_0(S_i^0).
# \end{equation*}
# 
# **Step 3:**  Fit the imputed treatment effects to obtain $\hat\tau_1(s):=\mathbb{E}[\tilde{\Delta}_i^1|S=s]$ and $\hat\tau_0(s):=\mathbb{E}[\tilde{\Delta}_i^0|S=s]$;
# 
# **Step 4:**  The final HTE estimator is given by
# \begin{equation*}
# \hat{\tau}_{\text{X-learner}}(s)=g(s)\hat\tau_0(s)+(1-g(s))\hat\tau_1(s),
# \end{equation*}
# 
# where $g(s)$ is a weight function.

# ### **4. R learner**
# 
# 
# 

# ### **5. DR-learner**
# 
# 
# 
# 

# ### **6. Lp-R-learner**
# 
# 

# ### **7. Causal Forest**
# 
# 
# 
# 

# ## Demo Code
# In the following, we exhibit how to apply the learner on real data to do policy learning and policy evaluation, respectively.

# ### 1. Meta-Leaners

# By specifing the model_info, we assume a regression model that:
# \begin{align}
# Q(s,a,\beta) &= \beta_{00}+\beta_{01}*recency+\beta_{02}*history\\
# &+I(a=1)*\{\beta_{10}+\beta_{11}*recency+\beta_{12}*history\} \\
# &+I(a=2)*\{\beta_{20}+\beta_{21}*recency+\beta_{22}*history\} 
# \end{align}

# ####**Result Interpretation:** 

# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.

# In[ ]:




