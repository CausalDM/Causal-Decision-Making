#!/usr/bin/env python
# coding: utf-8

# # Mediation Analysis
# 
# In the context of causal effect estimation, we aim to evaluate the effect of a specific treatment $A$ on the outcome $Y$ of interest. However, there may exist other variables that can be influnced by treamtent, and affect the outcome at the same time. We denote these variables as the `Mediators`, denoted as $M$.
# 
# Let's borrow a classical example from [4] to illustrate the necessity of mediation analysis. Researchers would like to evaluate the direct effect of a birth-control pill on the incidence of thrombosis. However, it is also known that the pill has a negative indirect effect on thrombosis by reducing the probability of pregnancy. In this example, we would want to estimate the effect of birth-control pill on thrombosis in the sense that, independent of marital status and other potential mediators that may not be accounted for in the study, in order to obtain reliable and consistent results.
# 
# 

# In[1]:


from IPython import display
import os
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
display.Image("./images/CEL-Mediation-IID.png")


# 
# ## Definitions
# 
# 
# In general mediation analysis, there are two potential paths that can cause the treatment effect on the outcome: 
# 
# 1. The direct path from treament to outcome, denoted by $A\rightarrow R$;
# 
# 2. The indirect path from treatment to outcome through the mediator $M$, denoted by $A\rightarrow M\rightarrow R$.
# 
# More specifically, when adjusting the action from $A=a_0$ to $A=a_1$,  we define the total effect (TE), natural direct effect (DE), and the natural indirect effect (DE) as below:
# 
# \begin{equation}
# \begin{aligned}
# \text{TE}&= \mathbb{E}[R|do(A=a_1)]-\mathbb{E}[R|do(A=a_0)]\\
# \text{DE}&= \mathbb{E}[R|do(A=a_1,M=m^{(a_0)})]-\mathbb{E}[R|do(A=a_0)]\\
# \text{IE}&= \mathbb{E}[R|do(A=a_0,M=m^{(a_1)})]-\mathbb{E}[R|do(A=a_0)]\\
# \end{aligned}
# \end{equation}
# 
# Under the potential outcome's structure, we define $M_a$ as the potential mediator when treatment $A=a$, and define $R_{a,m}$ as the potential outcome/reward one would observe under $(A=a, M=m)$. In some literature, the above effects can be samely written as
# 
# \begin{equation}
# \begin{aligned}
# \text{TE}&= \mathbb{E}[R_{a_1,m_{a_1}}]-\mathbb{E}[R_{a_0,m_{a_0}}]\\
# \text{DE}&= \mathbb{E}[R_{a_1,m_{a_0}}]-\mathbb{E}[R_{a,m_{a_0}}]\\
# \text{IE}&= \mathbb{E}[R_{a_1,m_{a_1}}]-\mathbb{E}[R_{a_1,m_{a_0}}]\\
# \end{aligned}
# \end{equation}
# 
# 
# ## Identification
# 
# Assumptions:
# 1. `Consistency`: $M_a = M$ when $A=a$, and $R_{a,m}=R$ when $A=a, M=m$.
# 
# 2. `No unmeasured confounders` (i.e. `NUC`): $\{R_{a',m},M_a\}\perp A|X$, and $R_{a',m}\perp M|A=a,X$.
# 
# 3. `Positivity`: $p(m|A,X)>0$ for each $m\in \mathcal{M}$, and $p(a|X)>0$ for each $a\in \mathcal{A}$.
# 
# Under the above three assumptions, Imai et al. [3] proved the identifiability of $\mathbb{E}[R_{1,M_0}]$ and $\mathbb{E}[R_{a,M_a}]$ in binary action space, which is given by
# 
# \begin{equation}
# \begin{aligned}
# \mathbb{E}[R_{1,M_0}] &= \int\int \mathbb{E}[R|A=1,M=m,S=s]p(m|A=0,S=s)p(s) d\mu(m,s)\\
# \mathbb{E}[R_{a,M_a}] &= \int\int \mathbb{E}[R|A=a,M=m,S=s]p(m|A=a,S=s)p(s) d\mu(m,s)
# \end{aligned}
# \end{equation}

# ## Estimation
# 
# In this section, we introduce three estimators that are commonly used in mediation analysis when no unmeasured confounders (NUC) assumption holds.
# 

# ### 1. Direct Estimator
# The first estimator is the direct method, which is a plug-in estimator based on the identification result above.  Since TE, DE and IE can be written as a function of $\mathbb{E}[R_{a,m_{a'}}]$, it suffice to estimate them separately for any $a, a'\in \mathcal{A}$ and construct a DM estimator as below:
# 
# \begin{equation}
# \begin{aligned}
# \widehat{\text{DE}}_{\text{DM}}&= \frac{1}{N}\sum_{i,m} \bigg\{R(S_i,a_1,m)p(m|S_i,a_0)-R(S_i,a_0,m)p(m|S_i,a_0)\bigg\}\\
# \widehat{\text{IE}}_{\text{DM}}&= \frac{1}{N}\sum_{i,m} \bigg\{R(S_i,a_1,m)p(m|S_i,a_1)- R(S_i,a_1,m)p(m|S_i,a_0)\bigg\}
# \end{aligned}
# \end{equation}
# 
# 

# In[ ]:





# ### 2. IPW Estimator
# The second estimator in literature is named as the inverse probability weighting estimator, which is similar to the IPW estimator in ATE. Under the existence of mediators, the IPW estimators [2] of DE and IE are given by
# 
# \begin{equation}
# \begin{aligned}
# \widehat{\text{DE}}_{\text{IPW}}&= \frac{1}{N}\sum_{i=1}^N \bigg\{\frac{\mathbb{1}\{A_i=a_1\}\rho(S_i,A_i,M_i)}{p_a(A_i|S_i)}-\frac{\mathbb{1}\{A_i=a_0\}}{p_a(A_i|S_i)}\bigg\}\cdot R_i\\
# \widehat{\text{IE}}_{\text{IPW}}&= \frac{1}{N}\sum_{i=1}^N \bigg\{\frac{\mathbb{1}\{A_i=a_1\}}{p_a(A_i|S_i)}-\frac{\mathbb{1}\{A_i=a_1\}\rho(S_i,A_i,M_i)}{p_a(A_i|S_i)}\bigg\}\cdot R_i
# \end{aligned}
# \end{equation}
# where $\rho(S,A,M)=\frac{p(M|S,A=a_0)}{p(M|S,A)}$ is the probability ratio that can adjust for the bias caused by distribution shift.

# In[ ]:





# ### 3. Multiple Robust (MR) Estimator
# The last estimator is called the multiple robust estimator, which was proposed by Tchetgen and Shpitser [5] based on the efficient influence function in semiparametric theory. The final MR estimator for DE and IE are derived as
# 
# \begin{equation}
# \begin{aligned}
# \widehat{\text{DE}}_{\text{MR}}&= \frac{1}{N}\sum_{i=1}^N \bigg[\frac{\mathbb{1}\{A_i=a_1\}}{p_a(A_i|S_i)}\rho(S_i,A_i,M_i)\big\{R_i-\mathbb{E}[R|S_i,A_i=1,M_i]\big\}\\
# &+\frac{\mathbb{1}\{A_i=a_0\}}{p_a(A_i|S_i)}\big\{\mathbb{E}[R|S_i,A_i=1,M_i]-R_i-\eta(a_1,a_0,S)+\eta(a_0,a_0,S)\big\}+\eta(a_1,a_0,S)-\eta(a_0,a_0,S)\big\}\bigg]\\
# \widehat{\text{IE}}_{\text{MR}}&= \frac{1}{N}\sum_{i=1}^N \bigg[\frac{\mathbb{1}\{A_i=a_1\}}{p_a(A_i|S_i)}\Big\{R_i-\eta(a_1,a_1,S)-\rho(S_i,A_i,M_i)\big\{R_i-\mathbb{E}[R|S_i,A=1,M_i]\big\}\Big\}\\
# &-\frac{\mathbb{1}\{A_i=a_0\}}{p_a(A_i|S_i)}\big\{\mathbb{E}[R|S_i,A=1,M_i]-\eta(a_1,a_0,S_i)\big\}+\eta(a_1,a_1,S_i)-\eta(a_1,a_0,S_i)\bigg]
# \end{aligned}
# \end{equation}

# In[1]:





# ## Data Demo
# ### 1. AURORA Data

# In[4]:


import os
import pandas as pd
os.chdir('/Users/alinaxu/Documents/CDM/CausalDM')
AURORA_CEL = pd.read_csv('./causaldm/data/Survey_red.csv')


# In[5]:


AURORA_CEL.columns


# In[6]:


import matplotlib.pyplot as plt
plt.hist(AURORA_CEL['pre-trauma insomnia (cont)'],bins=10)


# In[4]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


n = len(AURORA_CEL)


# In[5]:


state = AURORA_CEL[['Female', 'Age', 'Non-Hispanic White', 'Education',
       'pre-trauma physical health', 'pre-trauma mental health',
       'Chronic Perception of Severity of Stress', 'Neuroticism',
       'Childhood trauma']]
action = AURORA_CEL['pre-trauma insomnia (cont)']
mediator = AURORA_CEL[['peritraumatic distress', 'W2 acute stress disorder', 'W2 ptsd',
       'W2 Depression']]
reward = AURORA_CEL['3 month ptsd']

AURORA_CEL_MD = {'state':state,'action':action,'mediator':mediator,'reward':reward}


# In[6]:


action[np.where(action>=0)[0]] = 1
action[np.where(action<0)[0]] = 0


# In[7]:


from causaldm.causaldm.learners.Causal_Effect_Learning.Mediation_Analysis.ME_Single import ME_Single


# In[8]:


# Control Policy
def control_policy(state = None, dim_state=None, action=None, get_a = False):
    if get_a:
        action_value = np.array([0])
    else:
        state = np.copy(state).reshape(-1,dim_state)
        NT = state.shape[0]
        if action is None:
            action_value = np.array([0]*NT)
        else:
            action = np.copy(action).flatten()
            if len(action) == 1 and NT>1:
                action = action * np.ones(NT)
            action_value = 1-action
    return action_value

def target_policy(state, dim_state = 1, action=None):
    state = np.copy(state).reshape((-1, dim_state))
    NT = state.shape[0]
    pa = 1 * np.ones(NT)
    if action is None:
        if NT == 1:
            pa = pa[0]
            prob_arr = np.array([1-pa, pa])
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            raise ValueError('No random for matrix input')
    else:
        action = np.copy(action).flatten()
        action_value = pa * action + (1-pa) * (1-action)
    return action_value


# In[9]:


problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
Direct_est = ME_Single(AURORA_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 9, dim_mediator = 4, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'Direct')

Direct_est.estimate_DE_ME()
Direct_est.est_DE, Direct_est.est_ME, Direct_est.est_TE,


# In[10]:


IPW_est = ME_Single(AURORA_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 9, dim_mediator = 4, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'IPW')

IPW_est.estimate_DE_ME()
IPW_est.est_DE, IPW_est.est_ME, IPW_est.est_TE,


# In[38]:


Robust_est = ME_Single(AURORA_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 9, dim_mediator = 4, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'Robust')

Robust_est.estimate_DE_ME()
Robust_est.est_DE, Robust_est.est_ME, Robust_est.est_TE,


# In[54]:


Robust_DE = np.zeros(5)
Robust_IE = np.zeros(5)
Robust_TE = np.zeros(5)

Robust_DE[0] = Robust_est.est_DE
Robust_IE[0] = Robust_est.est_ME
Robust_TE[0] = Robust_est.est_TE


# In[64]:


for i in range(1,5):
    mediator_1d = mediator.iloc[:,i-1]
    AURORA_CEL_1D = {'state':state,'action':action,'mediator':mediator_1d,'reward':reward}
    
    Robust_est = ME_Single(AURORA_CEL_1D, r_model = 'OLS',
                         problearner_parameters = problearner_parameters,
                         truncate = 50, 
                         target_policy=target_policy, control_policy = control_policy, 
                         dim_state = 9, dim_mediator = 1, 
                         expectation_MCMC_iter = 50,
                         nature_decomp = True,
                         seed = 10,
                         method = 'Robust')

    Robust_est.estimate_DE_ME()
    Robust_DE[i] = Robust_est.est_DE
    Robust_IE[i] = Robust_est.est_ME
    Robust_TE[i] = Robust_est.est_TE


# In[73]:


Mediators_index = ["Overall"]
Mediators_index.append(mediator.columns.values)
print(Mediators_index)


# In[77]:


df = pd.DataFrame()
df['Mediators'] = np.array(['Four mediators in Total','peritraumatic distress', 'W2 acute stress disorder', 'W2 ptsd','W2 Depression'])


df['DE'] = np.round(Robust_DE.reshape(-1, 1), 3)
df['IE'] = np.round(Robust_IE.reshape(-1, 1), 3)
df['TE'] = np.round(Robust_TE.reshape(-1, 1), 3)

df


# #### treatment effect

# In[12]:


n = len(AURORA_CEL)
#userinfo_index = np.array([3,5,6,7,8,9,10])
SandA = AURORA_CEL[['Female', 'Age', 'Non-Hispanic White', 'Education',
       'pre-trauma physical health', 'pre-trauma mental health',
       'Chronic Perception of Severity of Stress', 'Neuroticism',
       'Childhood trauma','pre-trauma insomnia (cont)']]


# In[16]:


SandA


# In[15]:


# S-learner
np.random.seed(0)
S_learner = GradientBoostingRegressor(max_depth=5)
S_learner.fit(SandA, reward)


# In[17]:


SandA_all1 = SandA.copy()
SandA_all0 = SandA.copy()
SandA_all1['pre-trauma insomnia (cont)']=np.ones(n)
SandA_all0['pre-trauma insomnia (cont)']=np.zeros(n)

ATE_DM = np.sum(S_learner.predict(SandA_all1) - S_learner.predict(SandA_all0))/n


# In[18]:


ATE_DM


# In[19]:


# propensity score model fitting
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression()
ps_model.fit(state,  action)


# In[20]:


pi_S = ps_model.predict_proba(state)
ATE_IS = np.sum((action/pi_S[:,1] - (1-action)/pi_S[:,0])*reward)/n


# In[21]:


ATE_IS


# In[22]:


np.sum(action*(reward-S_learner.predict(SandA_all1))/pi_S[:,1] - (1-action)*(reward-S_learner.predict(SandA_all0))/pi_S[:,0])/n


# In[23]:


# combine the DM estimator and IS estimator
ATE_DR = ATE_DM + np.sum(action*(reward-S_learner.predict(SandA_all1))/pi_S[:,1] - (1-action)*(reward-S_learner.predict(SandA_all0))/pi_S[:,0])/n
ATE_DR


# In[52]:


# mediation effect
df = pd.DataFrame(columns=['DE','IE','TE'],index=['four mediators in total'])
df.iloc[0,] = [round(Robust_est.est_DE,3), round(Robust_est.est_ME,3), round(Robust_est.est_TE,3)]
df


# In[53]:


# treatment effect
df = pd.DataFrame(columns=['DM','IS','DR'],index=['treatment effect'])
df.iloc[0,] = [round(ATE_DM,3), round(ATE_IS,3), round(ATE_DR,3)]
df


# ### 2. Covid19 Data

# In[7]:


import os
import pandas as pd
Covid19_CEL = pd.read_csv('./causaldm/data/covid19.csv')


# In[8]:


Covid19_CEL


# In[6]:


# import related packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt;
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


n = len(Covid19_CEL)


# In[38]:


state = np.zeros(n).reshape(-1, 1)
#state = np.array(Covid19_CEL['Beijing']).reshape(-1, 1)
action = np.array(Covid19_CEL['A'])
mediator = np.array(Covid19_CEL['Y'])
reward = np.array(Covid19_CEL['Tianjin'])
MovieLens_CEL_MD = {'state':state,'action':action,'mediator':mediator,'reward':reward}


# In[39]:


from causaldm.causaldm.learners.Causal_Effect_Learning.Mediation_Analysis.ME_Single import ME_Single


# In[40]:


# Control Policy
def control_policy(state = None, dim_state=None, action=None, get_a = False):
    if get_a:
        action_value = np.array([0])
    else:
        state = np.copy(state).reshape(-1,dim_state)
        NT = state.shape[0]
        if action is None:
            action_value = np.array([0]*NT)
        else:
            action = np.copy(action).flatten()
            if len(action) == 1 and NT>1:
                action = action * np.ones(NT)
            action_value = 1-action
    return action_value

def target_policy(state, dim_state = 1, action=None):
    state = np.copy(state).reshape((-1, dim_state))
    NT = state.shape[0]
    pa = 1 * np.ones(NT)
    if action is None:
        if NT == 1:
            pa = pa[0]
            prob_arr = np.array([1-pa, pa])
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            raise ValueError('No random for matrix input')
    else:
        action = np.copy(action).flatten()
        action_value = pa * action + (1-pa) * (1-action)
    return action_value


# In[41]:


problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,50)},
Direct_est = ME_Single(MovieLens_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 1, dim_mediator = 1, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'Direct')

Direct_est.estimate_DE_ME()
Direct_est.est_DE, Direct_est.est_ME, Direct_est.est_TE,


# In[42]:


IPW_est = ME_Single(MovieLens_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 1, dim_mediator = 1, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'IPW')

IPW_est.estimate_DE_ME()
IPW_est.est_DE, IPW_est.est_ME, IPW_est.est_TE,


# In[44]:


Robust_est = ME_Single(MovieLens_CEL_MD, r_model = 'OLS',
                     problearner_parameters = problearner_parameters,
                     truncate = 50, 
                     target_policy=target_policy, control_policy = control_policy, 
                     dim_state = 1, dim_mediator = 1, 
                     expectation_MCMC_iter = 50,
                     nature_decomp = True,
                     seed = 10,
                     method = 'Robust')

Robust_est.estimate_DE_ME()
Robust_est.est_DE, Robust_est.est_ME, Robust_est.est_TE,


# In[49]:


Robust_DE = np.zeros(30)
Robust_IE = np.zeros(30)
Robust_TE = np.zeros(30)


for i in range(1,31):
    state = np.zeros(n).reshape(-1, 1)
    #state = np.array(Covid19_CEL['Beijing']).reshape(-1, 1)
    action = np.array(Covid19_CEL['A'])
    mediator = np.array(Covid19_CEL['Y'])
    reward = np.array(Covid19_CEL.iloc[:,i])
    MovieLens_CEL_MD = {'state':state,'action':action,'mediator':mediator,'reward':reward}
    
    MovieLens_CEL_MD
    Robust_est = ME_Single(MovieLens_CEL_MD, r_model = 'OLS',
                         problearner_parameters = problearner_parameters,
                         truncate = 50, 
                         target_policy=target_policy, control_policy = control_policy, 
                         dim_state = 1, dim_mediator = 1, 
                         expectation_MCMC_iter = 50,
                         nature_decomp = True,
                         seed = 10,
                         method = 'Robust')

    Robust_est.estimate_DE_ME()
    Robust_DE[i-1] = Robust_est.est_DE
    Robust_IE[i-1] = Robust_est.est_ME
    Robust_TE[i-1] = Robust_est.est_TE


# In[51]:


# Analysis of causal effects of 2020 Hubei lockdowns on reducing the COVID-19 spread in China regulated by Chinese major cities outside Hubei
df = pd.DataFrame()
df['cities'] = np.array(Covid19_CEL.columns.values[1:31])


df['DE'] = np.round(Robust_DE.reshape(-1, 1), 3)
df['IE'] = np.round(Robust_IE.reshape(-1, 1), 3)
df['TE'] = np.round(Robust_TE.reshape(-1, 1), 3)

df


# In[ ]:





# ## References
# 1. Hicks, Raymond and Dustin Tingley (2011). “Causal mediation analysis”. In: The Stata Journal 11.4, pp. 605–619.
# 
# 2. Hong, Guanglei et al. (2010). “Ratio of mediator probability weighting for estimating natural direct and indirect effects”. In: Proceedings of the American Statistical Association, biometrics section. Alexandria, VA, USA, pp. 2401–2415.
# 
# 3. Imai, Kosuke, Luke Keele, and Dustin Tingley (2010). “A general approach to causal mediation analysis.”. In: Psychological methods 15.4, p. 309.
# 
# 4. Pearl, Judea (2022). “Direct and indirect effects”. In: Probabilistic and causal inference: The works of Judea Pearl, pp. 373–392.
# 
# 5. Tchetgen, Eric J Tchetgen and Ilya Shpitser (2012). “Semiparametric theory for causal mediation analysis: efficiency bounds, multiple robustness, and sensitivity analysis”. In: Annals of statistics 40.3, p. 1816.
# 

# In[ ]:




