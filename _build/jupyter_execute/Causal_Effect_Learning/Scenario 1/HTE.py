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
# Next, let's briefly summarize some state-of-the-art approaches in estimating the heterogeneous treatment effect. There are several review papers which summarize some commonly-used approaches in literature, some of which are also detailed in the following subsections here. For more details please refer to [1], etc.
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


# import related packages
from matplotlib import pyplot as plt;
from lightgbm import LGBMRegressor;
from sklearn.linear_model import LinearRegression
from causaldm._util_causaldm import *;


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


data_behavior


# In[ ]:


SandA = data_behavior.iloc[:,0:3]


# In[ ]:


# S-learner
S_learner = LGBMRegressor(max_depth=5)
#S_learner = LinearRegression()
#SandA = np.hstack((S.to_numpy(),A.to_numpy().reshape(-1,1)))
S_learner.fit(SandA, data_behavior['R'])


# In[ ]:


HTE_S_learner = S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.ones(n).reshape(-1,1)))) - S_learner.predict(np.hstack(( data_behavior.iloc[:,0:2].to_numpy(),np.zeros(n).reshape(-1,1))))


# To evaluate how well S-learner is in estimating heterogeneous treatment effect, we compare its estimates with the true value for the first 10 subjects:

# In[ ]:


print("S-learner:  ",HTE_S_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_S_learner = np.sum(HTE_S_learner-HTE_true)/n
Variance_S_learner = np.sum((HTE_S_learner-HTE_true)**2)/n
print("The overall estimation bias of S-learner is :     ", Bias_S_learner, ", \n", "The overall estimation variance of S-learner is :",Variance_S_learner,". \n")


# **Conclusion:** The performance of S-learner, at least in this toy example, is not very attractive. Although it is the easiest approach to implement, the over-simplicity tends to cover some information that can be better explored with some advanced approaches.

# 
# ### **2. T-learner**
# The second learner is called T-learner, which denotes ``two learners". Instead of fitting a single model to estimate the potential outcomes under both treatment and control groups, T-learner aims to learn different models for $\mathbb{E}[R(1)|S]$ and $\mathbb{E}[R(0)|S]$ separately, and finally combines them to obtain a final HTE estimator.
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


mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

mu0.fit(data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] )
mu1.fit(data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2],data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] )


# estimate the HTE by T-learner
HTE_T_learner = mu1.predict(data_behavior.iloc[:,0:2]) - mu0.predict(data_behavior.iloc[:,0:2])


# Now let's take a glance at the performance of T-learner by comparing it with the true value for the first 10 subjects:

# In[ ]:


print("T-learner:  ",HTE_T_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# This is quite good! T-learner captures the overall trend of the treatment effect w.r.t. the heterogeneity of different subjects.

# In[ ]:


Bias_T_learner = np.sum(HTE_T_learner-HTE_true)/n
Variance_T_learner = np.sum((HTE_T_learner-HTE_true)**2)/n
print("The overall estimation bias of T-learner is :     ", Bias_T_learner, ", \n", "The overall estimation variance of T-learner is :",Variance_T_learner,". \n")


# **Conclusion:** In this toy example, the overall estimation variance of T-learner is smaller than that of S-learner. In some cases when the treatment effect is relatively complex, it's likely to yield better performance by fitting two models separately. 
# 
# However, in an extreme case when both $\mu_0(s)$ and $\mu_1(s)$ are nonlinear complicated function of state $s$ while their difference is just a constant, T-learner will overfit each model very easily, yielding a nonlinear treatment effect estimator. In this case, other estimators are often preferred.

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
# \tilde{\Delta}_i^1:=R_i^1-\hat\mu_0(S_i^1), \quad \tilde{\Delta}_i^0:=\hat\mu_1(S_i^0)-R_i^0.
# \end{equation*}
# 
# **Step 3:**  Fit the imputed treatment effects to obtain $\hat\tau_1(s):=\mathbb{E}[\tilde{\Delta}_i^1|S=s]$ and $\hat\tau_0(s):=\mathbb{E}[\tilde{\Delta}_i^0|S=s]$;
# 
# **Step 4:**  The final HTE estimator is given by
# \begin{equation*}
# \hat{\tau}_{\text{X-learner}}(s)=g(s)\hat\tau_0(s)+(1-g(s))\hat\tau_1(s),
# \end{equation*}
# 
# where $g(s)$ is a weight function between $[0,1]$. A possible way is to use the propensity score model as an estimate of $g(s)$.

# In[ ]:


# Step 1: Fit two models under treatment and control separately, same as T-learner

import numpy as np
mu0 = LGBMRegressor(max_depth=3)
mu1 = LGBMRegressor(max_depth=3)

S_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],0:2]
S_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],0:2]
R_T0 = data_behavior.iloc[np.where(data_behavior['A']==0)[0],3] 
R_T1 = data_behavior.iloc[np.where(data_behavior['A']==1)[0],3] 

mu0.fit(S_T0, R_T0)
mu1.fit(S_T1, R_T1)


# In[ ]:


# Step 2: impute the potential outcomes that are unobserved in original data

n_T0 = len(R_T0)
n_T1 = len(R_T1)

Delta0 = mu1.predict(S_T0) - R_T0
Delta1 = R_T1 - mu0.predict(S_T1) 


# In[ ]:


# Step 3: Fit tau_1(s) and tau_0(s)

tau0 = LGBMRegressor(max_depth=2)
tau1 = LGBMRegressor(max_depth=2)

tau0.fit(S_T0, Delta0)
tau1.fit(S_T1, Delta1)


# In[ ]:


# Step 4: fit the propensity score model $\hat{g}(s)$ and obtain the final HTE estimator by taking weighted average of tau0 and tau1
from sklearn.linear_model import LogisticRegression 

g = LogisticRegression()
g.fit(data_behavior.iloc[:,0:2],data_behavior['A'])

HTE_X_learner = g.predict_proba(data_behavior.iloc[:,0:2])[:,0]*tau0.predict(data_behavior.iloc[:,0:2]) + g.predict_proba(data_behavior.iloc[:,0:2])[:,1]*tau1.predict(data_behavior.iloc[:,0:2])




# In[ ]:


print("X-learner:  ",HTE_X_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# From the result above we can see that X-learner can roughly catch the trend of treatment effect w.r.t. the change of baseline information $S$. In this synthetic example, X-learner also performs slightly better than T-learner.

# In[ ]:


Bias_X_learner = np.sum(HTE_X_learner-HTE_true)/n
Variance_X_learner = np.sum((HTE_X_learner-HTE_true)**2)/n
print("The overall estimation bias of X-learner is :     ", Bias_X_learner, ", \n", "The overall estimation variance of X-learner is :",Variance_X_learner,". \n")


# **Conclusion:** In this toy example, the overall estimation variance of X-learner is the smallest, followed by T-learner, and the worst is given by S-learner.

# ### **4. R learner**
# The idea of classical R-learner came from Robinson 1988 [3] and was formalized by Nie and Wager in 2020 [2]. The main idea of R learner starts from the partially linear model setup, in which we assume that
# \begin{equation}
#   \begin{aligned}
#     R&=A\tau(S)+g_0(S)+U,\\
#     A&=m_0(S)+V,
#   \end{aligned}
# \end{equation}
# where $U$ and $V$ satisfies $\mathbb{E}[U|D,X]=0$, $\mathbb{E}[V|X]=0$.
# 
# After several manipulations, it’s easy to get
# \begin{equation}
# 	R-\mathbb{E}[R|S]=\tau(S)\cdot(A-\mathbb{E}[A|S])+\epsilon.
# \end{equation}
# Define $m_0(X)=\mathbb{E}[A|S]$ and $l_0(X)=\mathbb{E}[R|S]$. A natural way to estimate $\tau(X)$ is given below, which is also the main idea of R-learner:
# 
# **Step 1**: Regress $R$ on $S$ to obtain model $\hat{\eta}(S)=\hat{\mathbb{E}}[R|S]$; and regress $A$ on $S$ to obtain model $\hat{m}(S)=\hat{\mathbb{E}}[A|S]$.
# 
# **Step 2**: Regress outcome residual $R-\hat{l}(S)$ on propensity score residual $A-\hat{m}(S)$.
# 
# That is,
# \begin{equation}
# 	\hat{\tau}(S)=\arg\min_{\tau}\left\{\mathbb{E}_n\left[\left(\{R_i-\hat{\eta}(S_i)\}-\{A_i-\hat{m}(S_i)\}\cdot\tau(S_i)\right)^2\right]\right\}	
# \end{equation}
# 
# The easiest way to do so is to specify $\hat{\tau}(S)$ to the linear function class. In this case, $\tau(S)=S\beta$, and the problem becomes to estimate $\beta$ by solving the following linear regression:
# \begin{equation}
# 	\hat{\beta}=\arg\min_{\beta}\left\{\mathbb{E}_n\left[\left(\{R_i-\hat{\eta}(S_i)\}-\{A_i-\hat{m}(S_i)\} S_i\cdot \beta\right)^2\right]\right\}.
# \end{equation}
# 
# 

# In[ ]:


# a demo code of R-learner

def Rlearner(df, outcome, treatment, controls, n_folds, y_model, ps_model, Rlearner_model):
    """
    Parameters
    ----------
    df : pd.dataframe
        data
    outcome : str
        outcome label.
    treatment : str
        treatment label.
    controls : list
        list of all controls.
    n_folds : int
        number of folds for cross-fitting.
    y_model : sklearn class
        the model for outcome regression learner.
    ps_model : sklearn class
        the model for general propensity score learner.

    Returns
    -------
    Rlearner_pred : Length: n, dtype: float64
        Estimated Heterogeneous Treatemnt Effect by Simple R-learner with linear regression
    """

    # =============================================================================
    # # estimate with R-learner
    # =============================================================================

    print('estimate with R-learner')

    import numpy as np
    import pandas as pd

    # estimate p(x) by GBDT(Gradient Boosting Decision Tree)
    # estimate m(x) by Random Forest
    n_controls=len(controls)
    folds=np.random.randint(1,n_folds+1,size=df.shape[0])
    
    y_learner=[y_model]*n_folds
    ps_learner=[ps_model]*n_folds


    y_pred=pd.Series(index=df.index,dtype=np.float64)
    ps_pred=pd.Series(index=df.index,dtype=np.float64)
    
    
    for i in range(n_folds):
        fold=i+1
        #y_learner for outcome prediction
        y_learner[i].fit(df[folds!=fold][controls],df[folds!=fold][outcome])
        y_pred.loc[folds==fold]=y_learner[i].predict(df[folds==fold][controls])

        #ps_learner for propensity score prediction
        ps_learner[i].fit(df[folds!=fold][controls],df[folds!=fold][treatment])
        ps_pred.loc[folds==fold]=ps_learner[i].predict_proba(df[folds==fold][controls])[:,1]

        #model performance output
        print('fold {},testing r2 y_learner: {:.3f}, ps_learner: {:.3f}'.format(fold, 
                        y_learner[i].score(df[folds==fold][controls],df[folds==fold][outcome]),
                        ps_learner[i].score(df[folds==fold][controls],df[folds==fold][treatment])
                                            ))
      
    x_residual=df[controls]
    x_residual['Intercept']=1
    
    y_residual=df[outcome]-y_pred
    ps_residual=df[treatment]-ps_pred
    x_tilde=ps_residual.to_numpy().reshape(-1,1)*(x_residual.to_numpy())
    
    data=pd.DataFrame(x_tilde)
    data['y_residual']=y_residual
    
    # R learner: conducting regressison on residuals: (Y-y_pred)~(A-ps_pred)*X'*beta
    # any parametric/nonparametric regression method is fine
    Rlearner_pred=pd.Series(index=df.index,dtype=np.float64)

    #Rlearner_model=GradientBoostingRegressor(n_estimators=50, max_depth=5)
    #Rlearner_model=LinearRegression(fit_intercept=False) # almost failed: testing r2 R-learner: 0.041
    #Rlearner_model=ElasticNet() # almost failed
    #Rlearner_model=Lasso() # almost failed
    R_learner=[Rlearner_model]*n_folds   
    
    for i in range(n_folds):
        fold=i+1
        #R_learner for residual regression
        R_learner[i].fit(data[folds!=fold][range(n_controls+1)],data[folds!=fold]['y_residual'])
        Rlearner_pred.loc[folds==fold]=R_learner[i].predict(x_residual[folds==fold])

        #model performance output
        print('fold {}, training r2 R-learner: {:.3f}, testing r2 R-learner: {:.3f}'.format(fold, R_learner[i].score(data[folds!=fold][range(n_controls+1)],data[folds!=fold]['y_residual']), R_learner[i].score(data[folds==fold][range(n_controls+1)],data[folds==fold]['y_residual'])  ))
    
    return Rlearner_pred



# In[ ]:


# R-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = LGBMRegressor(max_depth=2)

HTE_R_learner = Rlearner(data_behavior, outcome, treatment, controls, n_folds, y_model, ps_model, Rlearner_model)
HTE_R_learner = HTE_R_learner.to_numpy()


# In[ ]:


print("R-learner:  ",HTE_R_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_R_learner = np.sum(HTE_R_learner-HTE_true)/n
Variance_R_learner = np.sum((HTE_R_learner-HTE_true)**2)/n
print("The overall estimation bias of R-learner is :     ", Bias_R_learner, ", \n", "The overall estimation variance of R-learner is :",Variance_R_learner,". \n")


# **Conclusion:** It's amazing to see that the bias of R-learner is significantly smaller than all other approaches.

# ### **5. DR-learner**
# 
# DR-learner is a two-stage doubly robust estimator for HTE estimation. Before Kennedy et al. 2020 [4], there are several related approaches trying to extend the doubly robust procedure to HTE estimation, such as [5, 6, 7]. Compared with the above three estimators, DR-learner is proved to be oracle efficient under some mild assumptions detailed in Theorem 2 of [4].
# 
# The basic steps of DR-learner is given below:
# 
# **Step 1**: Nuisance training: \\
# (a)  Using $I_{1}^n$ to construct estimates $\hat{\pi}$ for the propensity scores $\pi$; \\
# (b)  Using $I_{1}^n$ to construct estimates $\hat\mu_a(s)$ for $\mu_a(s):=\mathbb{E}[R|S=s,A=a]$;
# 
# **Step 2**: Pseudo-outcome regression: \\
# Define $\widehat{\phi}(Z)$ as the pseudo-outcome where 
# \begin{equation}
# \widehat{\phi}(Z)=\frac{A-\hat{\pi}(S)}{\hat{\pi}(S)\{1-\hat{\pi}(S)\}}\Big\{R-\hat{\mu}_A(S)\Big\}+\hat{\mu}_1(S)-\hat{\mu}_0(S),
# \end{equation}
# and regress it on covariates $S$ in the test sample $I_2^n$, yielding 
# \begin{equation}
# \widehat{\tau}_{\text{DR-learner}}(s)=\widehat{\mathbb{E}}_n[\widehat{\phi}(Z)|S=s].
# \end{equation}
# 

# In[ ]:


# A demo code of DR-learner

def DRlearner(df, outcome, treatment, controls, y_model, ps_model, n_folds=5):
    """
    Parameters
    ----------
    df : pd.dataframe
        data
    outcome : str
        outcome label.
    treatment : str
        treatment label.
    controls : list
        list of all controls.
    y_model : sklearn class
        the model for outcome regression learner.
    ps_model : sklearn class
        the model for general propensity score learner.
    n_folds : int
        number of folds for cross-fitting.
    Returns
    -------
    TE_DR : Length: n, dtype: float64
        Estimated Heterogeneous Treatemnt Effect by DR-learner
    """
    # =============================================================================
    # # estimate with DR-learner
    # =============================================================================
    print('estimate with DR-learner')

    import pandas as pd
    import subprocess,os,pdb
    from sklearn.metrics import r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Lasso,ElasticNet
    from scipy import stats

    from sklearn.metrics import r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import Lasso,ElasticNet

    import pdb
    import numpy as np
    from scipy.sparse import diags
    
    dt_len=np.shape(df)[0]
    
    np.random.seed(525)
    folds=np.random.randint(1,n_folds+1,size=df.shape[0])
    
    y_learner=[y_model]*n_folds
    ps_learner=[ps_model]*n_folds
    
    y_pred=pd.Series(index=df.index,dtype=np.float64)
    ps_pred=pd.Series(index=df.index,dtype=np.float64)

    Y1_pred=pd.Series(index=df.index,dtype=np.float64)
    Y0_pred=pd.Series(index=df.index,dtype=np.float64)
    ps_pred=pd.Series(index=df.index,dtype=np.float64)

    df['T_1']=1
    df['T_0']=0
    
    
    # estimate classical DR 
    for i in range(n_folds):
        fold=i+1
        #baselearner for outcome prediction
        y_learner[i].fit(df[folds!=fold][controls+[treatment]],df[folds!=fold][outcome])


        Y1_pred.loc[folds==fold]=y_learner[i].predict(df[folds==fold][controls+['T_1']])
        Y0_pred.loc[folds==fold]=y_learner[i].predict(df[folds==fold][controls+['T_0']])

        ps_learner[i].fit(df[folds!=fold][controls],df[folds!=fold][treatment])

        ps_pred.loc[folds==fold]=ps_learner[i].predict_proba(df[folds==fold][controls])[:,1]

        print('fold {}, testing r2 baselearner: {:.3f}, pslearner: {:.3f}'.format(fold, 
                        y_learner[i].score(df[folds!=fold][controls+[treatment]],df[folds!=fold][outcome]),
                        ps_learner[i].score(df[folds!=fold][controls],df[folds!=fold][treatment])
                                            ))



    #gps_pred[np.where(gps_pred<1e-2)[0]]=1e-2
    #gps_pred[np.where(gps_pred>1-1e-2)[0]]=1-1e-2

    
    # DR estimator
    TE_DR=Y1_pred-Y0_pred+df[treatment]*(df[outcome]-Y1_pred)/ps_pred-(1-df[treatment])*(df[outcome]-Y0_pred)/(1-ps_pred)
    
    
    return TE_DR
    


# In[ ]:


# DR-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model = LogisticRegression()
Rlearner_model = LGBMRegressor(max_depth=2)

HTE_DR_learner = DRlearner(data_behavior, outcome, treatment, controls, y_model, ps_model)
HTE_DR_learner = HTE_DR_learner.to_numpy()


# In[ ]:


print("DR-learner:  ",HTE_DR_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_DR_learner = np.sum(HTE_DR_learner-HTE_true)/n
Variance_DR_learner = np.sum((HTE_DR_learner-HTE_true)**2)/n
print("The overall estimation bias of DR-learner is :     ", Bias_DR_learner, ", \n", "The overall estimation variance of DR-learner is :",Variance_DR_learner,". \n")


# ### **6. Lp-R-learner**
# 
# As an extension of R-learner, Lp-R-learner combined the idea of residual regression with local polynomial adaptation, and leveraged the idea of cross fitting to further relax the conditions needed to obtain the oracle convergence rate. For brevity of content, we will just introduce their main algorithm. For more details about its theory and real data performance please see the paper written by Kennedy [4]. 
# 	
# Let $(I_{1a}^n, I_{1b}^n,I_{2}^n)$ denote three independent samples of $n$ observations of $Z_i = (S_i, A_i, R_i)$. Let $b:\mathbb{R}^d\rightarrow \mathbb{R}^p$ denote the vector of basis functions consisting of all powers of each covariate, up to order $\gamma$, and all interactions up to degree $\gamma$ polynomials. Let $K_{hs}(S)=\frac{1}{h^d}K\left(\frac{S-s}{h}\right)$ for $k:\mathbb{R}^d\rightarrow \mathbb{R}$ a bounded kernel function with support $[-1,1]^d$, and $h$ is a bandwidth parameter.
# 
# **Step 1**: Nuisance training: \\
# (a)  Using $I_{1a}^n$ to construct estimates $\hat{\pi}_a$ of the propensity scores $\pi$; \\
# (b)  Using $I_{1b}^n$ to construct estimates $\hat{\eta}$ of the regression function $\eta=\pi\mu_1+(1-\pi)\mu_0$, and estimtes $\hat{\pi}_b$ of the propensity scores $\pi$.
# 
# **Step 2**: Localized double-residual regression: \\
# Define $\hat{\tau}_r(s)$ as the fitted value from a kernel-weighted least squares regression (in the test sample $I_2^n$) of outcome residual $(R-\hat{\eta})$ on basis terms $b$ scaled by the treatment residual $A-\hat{\pi}_b$, with weights $\Big(\frac{A-\hat{\pi}_a}{A-\hat{\pi}_b}\Big)\cdot K_{hs}$. Thus $\hat{\tau}_r(s)=b(0)^T\hat{\theta}$ for
# \begin{equation}
# 		\hat{\theta}=\arg\min_{\theta\in\mathbb{R}^p}\mathbb{P}_n\left(K_{hs}(S)\Big\{ \frac{A-\hat{\pi}_a(S)}{A-\hat{\pi}_b(S)}\Big\} \left[  \big\{R-\hat{\eta}(S)\big\}-\theta^Tb(S-s_0)\big\{A-\hat{\pi}_b(S)\big\} \right] \right).
# \end{equation}
# **Step 3**: Cross-fitting(optional): \\
# Repeat Step 1–2 twice, first using $(I^n_{1b} , I_2^n)$ for nuisance training and $I_{1a}^n$ as the test samplem and then using $(I^n_{1a} , I_2^n)$ for training and $I_{1b}^n$ as the test sample. Use the average of the resulting three estimators of $\tau$ as the final estimator $\hat{\tau}_r$.
# 
# In the theory section, Kennedy proved that Lp-R-learner, compared with traditional DR learner, can achieve the oracle convergence rate under milder conditions. 

# In[ ]:


# A demo code of Lp-R-learner

def LpRlearner(df, outcome, treatment, controls, y_model, ps_model_a, ps_model_b, s, LpRlearner_model, degree = 1):
    """
    
    Parameters
    ----------
    df : pd.dataframe
        data
    outcome : str
        outcome label.
    treatment : str
        treatment label.
    controls : list
        list of all controls.
    y_model : sklearn class
        the model for outcome regression learner.
    ps_model_a : sklearn class
        the model for general propensity score learner in fold 1a.
    ps_model_b : sklearn class
        the model for general propensity score learner in fold 1b.
        s:  float64
            bandwidth of gauss kernel function in deciding the weight of regression
   LpRlearner_model:  sklearn class
        the model for residual regression learner in fold 2.
    n_folds : int
        number of folds for cross-fitting. Set as a fixed number, 3, as indicated in the paper    
    Returns
    -------
    LpRlearner_pred : Length: n, dtype: float64
        Estimated Heterogeneous Treatemnt Effect by Lp-R-learner with kernel-weighted polynomial regression
    """
    # =============================================================================
    # # estimate with Lp-R-learner
    # =============================================================================


    print('estimate with Lp-R-learner')


    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso,LogisticRegression
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd

    from sklearn.metrics import r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import Lasso,ElasticNet
    from sklearn.preprocessing import PolynomialFeatures

    n_all = len(df)
    
    n_folds = 3
    folds=np.random.randint(1,n_folds+1,size=df.shape[0])
    poly = PolynomialFeatures(degree = degree)
    

    
    tau=np.zeros((n_all,3))
    LpRlearner_pred = pd.Series(np.zeros(n_all))

    
    for j in range(n_all):  
        
        y_learner=[y_model]*n_folds
        ps_learner_a=[ps_model_a]*n_folds
        ps_learner_b=[ps_model_b]*n_folds
        Lp_R_learner=[LpRlearner_model]*n_folds

        y_pred=pd.Series(index=df.index,dtype=np.float64)
        ps_pred=pd.Series(index=df.index,dtype=np.float64)
        LpRlearner_pred=pd.Series(index=df.index,dtype=np.float64)

        for i in range(n_folds):
            fold=i+1

            # define the three-folds cross fitting index according to Kennedy's paper
            fold1a=fold
            fold1b=(fold+1)%n_folds
            fold2=(fold+2)%n_folds  
            if (fold1a == 0):
                fold1a = fold1a + n_folds
            if (fold1b == 0):
                fold1b = fold1b + n_folds
            if (fold2 == 0):
                fold2 = fold2 + n_folds

                
            # step 1: nuisance training
            ps_learner_a[fold1a-1].fit(df[folds==fold1a][controls],df[folds==fold1a][treatment])

            y_learner[fold1b-1].fit(df[folds==fold1b][controls],df[folds==fold1b][outcome])
            ps_learner_b[fold1b-1].fit(df[folds==fold1b][controls],df[folds==fold1b][treatment])

            
            #1st stage model performance output
            #print('fold {},training r2 y_learner: {:.3f}, ps_learner: {:.3f}'.format(fold1a,y_learner[fold1b-1].score(df[folds==fold1b][controls],df[folds==fold1b][outcome]), ps_learner_a[fold1a-1].score(df[folds==fold1a][controls],df[folds==fold1a][treatment])  ))
            #print('fold {},testing r2 y_learner: {:.3f}, ps_learner: {:.3f}'.format(fold1a,y_learner[fold1b-1].score(df[folds!=fold1b][controls],df[folds!=fold1b][outcome]),ps_learner_a[fold1a-1].score(df[folds!=fold1a][controls],df[folds!=fold1a][treatment])))
            
     
            x0=df[controls].iloc[j].to_numpy()#.reshape(-1,1)  ############## define another vector in argument line##
            X=df[controls][folds==fold2].to_numpy()
            
            #print(np.shape(X))
            n=len(df[folds==fold2])


            # choose h to ensure the support to be in between [-1,1]^d
            h=0
            for k in range(n):
                temp=np.max(abs(X[k,:]-x0))
                if (temp>h):
                    h=temp
            h=np.ceil(h)
            #print('the value of h is {:.3f}'.format(h) )
            
            # step 2: kernel-weighted least squares regression
            # kernel calculation
            # use gauss kernel to determine the weight of regression
            Kernel_X=np.exp(-sum( ((x - y)/h)**2 for (x, y) in zip(X.transpose(), x0) ) / s**2)

            ps_a=ps_model_a.predict_proba(df[folds==fold2][controls])[:,1]
            ps_b=ps_model_b.predict_proba(df[folds==fold2][controls])[:,1]


            ps_a[np.where(ps_a<1e-5)]=1e-5
            ps_b[np.where(ps_b<1e-5)]=1e-5
            ps_a[np.where(1-ps_a<1e-5)]=1-1e-5
            ps_b[np.where(1-ps_b<1e-5)]=1-1e-5

            weight=Kernel_X * (df[folds==fold2][treatment]-ps_a) / (df[folds==fold2][treatment]-ps_b)

            
            # polynomial regression at point x0
            X_poly = poly.fit_transform(df[folds==fold2][controls]-x0)
            y_residual = df[outcome][folds==fold2]-y_model.predict(df[folds==fold2][controls])
            x_tilde = X_poly * (df[folds==fold2][treatment]-ps_model_b.predict_proba(df[folds==fold2][controls])[:,1]).to_numpy().reshape(-1,1)
            
            p = np.shape(X_poly)[1]
            
            #poly.fit(X_poly_train,y_train)
            Lp_R_learner[fold2-1].fit(x_tilde, y_residual, sample_weight=weight)
            
            Theta = Lp_R_learner[fold2-1].coef_
            #LpRlearner_pred[0].loc[folds==(fold+2)]=LpRlearner_model.predict(X_poly)
            

            #model performance output
            X_poly_test = poly.fit_transform(df[folds!=fold2][controls]-x0)
            y_residual_test = df[outcome][folds!=fold2]-y_model.predict(df[folds!=fold2][controls])
            x_tilde_test = X_poly_test * (df[folds!=fold2][treatment]-ps_model_b.predict_proba(df[folds!=fold2][controls])[:,1]).to_numpy().reshape(-1,1)
            #print('fold {},training r2 of Lp-R-learner_model: {:.3f},testing r2 of Lp-R-learner_model: {:.3f}'.format(fold1a, Lp_R_learner[fold2-1].score(x_tilde,y_residual),Lp_R_learner[fold2-1].score(x_tilde_test,y_residual_test)))
            
            tau[j,i]=Theta[0] #the intercept of the linear regression
            
    LpRlearner_pred = np.sum(tau,axis=1)/n_folds        
            
    return LpRlearner_pred
    


# In[ ]:





# In[ ]:


# Lp-R-learner for HTE estimation
outcome = 'R'
treatment = 'A'
controls = ['S1','S2']
n_folds = 5
y_model = LGBMRegressor(max_depth=2)
ps_model_a = LogisticRegression()
ps_model_b = LogisticRegression()
s = 1
LpRlearner_model = LinearRegression()

HTE_Lp_R_learner = LpRlearner(data_behavior, outcome, treatment, controls, y_model, ps_model_a, ps_model_b, s, LpRlearner_model, degree = 1)


# In[ ]:


print("Lp_R-learner:  ",HTE_Lp_R_learner[0:8])
print("true value: ",HTE_true[0:8].to_numpy())


# In[ ]:


Bias_Lp_R_learner = np.sum(HTE_Lp_R_learner-HTE_true)/n
Variance_Lp_R_learner = np.sum((HTE_Lp_R_learner-HTE_true)**2)/n
print("The overall estimation bias of Lp_R-learner is :     ", Bias_Lp_R_learner, ", \n", "The overall estimation variance of Lp_R-learner is :",Variance_Lp_R_learner,". \n")


# **Conclusion**: It will cost more time to use Lp-R-learner than other approaches. However, the overall estimation variance of Lp-R-learner is incredibly smaller than other approaches.

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


# ## References
# 1. Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences 116, 4156–4165.
# 
# 2. Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319, 2021.
# 
# 3. Peter M Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society, pages 931–954, 1988.
# 
# 4. Edward H Kennedy. Optimal doubly robust estimation of heterogeneous causal effects. arXiv preprint arXiv:2004.14497, 2020
# 
# 5. M. J. van der Laan. Statistical inference for variable importance. The International Journal of Biostatistics, 2(1), 2006.
# 
# 6. S. Lee, R. Okui, and Y.-J. Whang. Doubly robust uniform confidence band for the conditional average treatment effect function. Journal of Applied Econometrics, 32(7):1207–1225, 2017.
# 
# 7. D. J. Foster and V. Syrgkanis. Orthogonal statistical learning. arXiv preprint arXiv:1901.09036, 2019.
# 
# 8. Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.
