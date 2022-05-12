#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def generate_test_case(setup, N, seed = 0, phi1=None, phi2=None,psi1=None,psi2=None):
    if setup == 'random_binary':
        np.random.seed(seed)
        s1 = np.random.normal(450,150, N)
        A1 = np.random.binomial(1,np.exp(phi1[0]+phi1[1]*s1)/(1+np.exp(phi1[0]+phi1[1]*s1)),size = N)
        s2 = np.random.normal(1.25*s1,60, N)
        A2 = np.random.binomial(1,np.exp(phi2[0]+phi2[1]*s2)/(1+np.exp(phi2[0]+phi2[1]*s2)),size = N)
        A = {}
        A[0] = A1
        A[1] = A2
        mu1 = (psi1[0]+psi1[1]*s1)*((psi1[0]+psi1[1]*s1 >0).astype(int)-A1)
        mu2 = (psi2[0]+psi2[1]*s2)*((psi2[0]+psi2[1]*s2 >0).astype(int)-A2)
        Y_opt = np.random.normal(400+1.6*s1,60)
        Y = Y_opt - mu1 - mu2
        opt_true = {}
        opt_true[0] = (psi1[0]+psi1[1]*s1 >0).astype(int)
        opt_true[1] = (psi2[0]+psi2[1]*s2 >0).astype(int)
        X = np.hstack([np.ones(N)[:, np.newaxis], s1[:, np.newaxis], A1[:, np.newaxis], (s1*A1)[:, np.newaxis], s2[:, np.newaxis]])
        
        instance = {
            'X' : X, 
            'A' : A, 
            'Y' : Y, 
            'optimal_A' : opt_true, 
            'optimal_V' : np.mean(Y_opt),
            'XAY' : [X, A, Y]
        }
        return instance


# In[2]:


instance = generate_test_case('random_binary', 1000, seed = 0,  phi1=[2,-.006],phi2=[.8,-.004],psi1=[250,-1],psi2=[720,-2])
X,A,Y = instance['XAY']


# In[3]:


instance['optimal_A'][0].sum()


# In[4]:


instance['optimal_V']


# In[5]:


# TODO: there might be something wrong with the multiple step as the difference between A-learning and Q-learning is large

# A demo with code on how to use the package
from causaldm.learners import ALearning
from causaldm.test import shared_simulation
import numpy as np

ALearn = ALearning.ALearning()
model_info = [{'X_prop': list(range(2)),
              'X_q0': list(range(2)),
               'X_C':{1:list(range(2))},
              'action_space': [0,1]},
             {'X_prop': [0,4],
              'X_q0': list(range(5)),
               'X_C':{1:[0,4]},
              'action_space': [0,1]}]
ALearn.train(X, A, Y, model_info, T=2, bootstrap = True, n_bs = 100)
fitted_params,fitted_value,value_avg,value_std,params=ALearn.estimate_value_boots()
print('Value_hat:',value_avg,'Value_std:',value_std)
##estimated contrast model at t = 0
print('estimated_contrast:',params[0]['contrast'])
print('estimated_contrast:',params[1]['contrast'])
print('estimated_prop:',params[0]['prop'])
print('estimated_prop:',params[1]['prop'])

# recommend action
opt_d = ALearn.recommend().head()
# get the estimated value of the optimal regime
V_hat = ALearn.estimate_value()
print("opt regime:",opt_d)
print("opt value:",V_hat)


# In[50]:


ALearn.recommend().sum(axis=0)


# In[51]:


np.array(params[0]['contrast'][1]['Mean'])


# In[52]:


# TODO: feasible set
from causaldm.learners import QLearning
from causaldm.test import shared_simulation
import numpy as np


# In[53]:


import pandas as pd
X = pd.DataFrame(X, columns=['intercept','S1','A1','S1A1','S2'])
del X['A1']
A = pd.DataFrame({'A1':A[0],'A2':A[1]})
Y = pd.DataFrame(Y,columns=['Y'])

# Optional: we also provide a bootstrap standard deviaiton of the optimal value estimation
# Warning: results amay not be reliable
QLearn = QLearning.QLearning()
model_info = [{"model": "Y~S1+A1+A1*S1",
              'action_space':{'A1':[0,1]}},
             {"model": "Y~S1+A1+A1*S1+S2+A2+A2*S2",
              'action_space':{'A2':[0,1]}}]
QLearn.train(X, A, Y, model_info, T=2, bootstrap = True, n_bs = 200)
fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()
print('Value_hat:',value_avg,'Value_std:',value_std)
print(params)


# In[54]:


QLearn.recommend().sum()


# In[57]:


A_est = {'C0':[],'C1':[],'Vhat':[]}
Q_est = {'Q0':[],'Q1':[],'Vhat':[]}
opt_V = []
for rep in range(100):
    instance = generate_test_case('random_binary', 1000, seed = rep,  phi1=[2,-.006],phi2=[.8,-.004],psi1=[250,-1],psi2=[720,-2])
    X,A,Y = instance['XAY']
    opt_V.append(instance['optimal_V'])
    
    ALearn = ALearning.ALearning()
    model_info = [{'X_prop': list(range(2)),
                  'X_q0': list(range(2)),
                   'X_C':{1:list(range(2))},
                  'action_space': [0,1]},
                 {'X_prop': [0,4],
                  'X_q0': list(range(5)),
                   'X_C':{1:[0,4]},
                  'action_space': [0,1]}]
    ALearn.train(X, A, Y, model_info, T=2, bootstrap = True, n_bs = 100)
    fitted_params,fitted_value,value_avg,value_std,params=ALearn.estimate_value_boots()
    # recommend action
    opt_d = ALearn.recommend().head()
    # get the estimated value of the optimal regime
    V_hat = ALearn.estimate_value()
    A_est['C0'].append(np.array(params[0]['contrast'][1]['Mean']))
    A_est['C1'].append(np.array(params[1]['contrast'][1]['Mean']))
    A_est['Vhat'].append(V_hat)

    X = pd.DataFrame(X, columns=['intercept','S1','A1','S1A1','S2'])
    del X['A1']
    A = pd.DataFrame({'A1':A[0],'A2':A[1]})
    Y = pd.DataFrame(Y,columns=['Y'])

    # Optional: we also provide a bootstrap standard deviaiton of the optimal value estimation
    # Warning: results amay not be reliable
    QLearn = QLearning.QLearning()
    model_info = [{"model": "Y~S1+A1+A1*S1",
                  'action_space':{'A1':[0,1]}},
                 {"model": "Y~S1+A1+A1*S1+S2+A2+A2*S2",
                  'action_space':{'A2':[0,1]}}]
    QLearn.train(X, A, Y, model_info, T=2, bootstrap = True, n_bs = 200)
    fitted_params,fitted_value,value_avg,value_std,params=QLearn.estimate_value_boots()

    # recommend action
    opt_d = QLearn.recommend().head()
    # get the estimated value of the optimal regime
    V_hat = QLearn.estimate_value()
    
    Q_est['Q0'].append(np.array(params[0]['Mean']))
    Q_est['Q1'].append(np.array(params[1]['Mean']))
    Q_est['Vhat'].append(V_hat)
    print(rep)


# In[58]:


sum(A_est['C0'])/100


# In[59]:


sum(A_est['C1'])/100


# In[60]:


sum(A_est['Vhat'])/100


# In[61]:


sum(Q_est['Q0'])/100


# In[62]:


sum(Q_est['Q1'])/100


# In[63]:


sum(Q_est['Vhat'])/100


# In[64]:


sum(opt_V)/100


# # Test A-Learning Single

# In[ ]:


def generate_test_case(setup, N, seed = 0):
    if setup == 'random_binary':
        np.random.seed(seed)
        s1 = np.random.normal(0,1, N)
        A1 = np.random.binomial(1,np.exp(-2*s1)/(1+np.exp(-2*s1)),size = N)
        Y = np.random.normal(1+s1+A1*(1+.5*s1),3)
        opt_true = (1+.5*s1 >0).astype(int)
        X = np.hstack([np.ones(N)[:, np.newaxis], s1[:, np.newaxis]])
        instance = {
            'X' : X, 
            'A' : A1, 
            'Y' : Y, 
            'optimal_A' : opt_true, 
            'XAY' : [X, A1, Y]
        }
        return instance
    
instance = generate_test_case('random_binary', 10000, seed = 0)
X,A1,Y = instance['XAY']
A = {}
A[0] = A1


# In[ ]:


# initialize the learner
ALearn = ALearning.ALearning()
p = X.shape[1]
model_info = [{'X_prop': list(range(p)),
              'X_q0': list(range(p)),
               'X_C':{1:list(range(p))},
              'action_space': [0,1]}] #A in [0,1,2]
# train the policy
ALearn.train(X, A, Y, model_info, T=1)
# Fitted Model
ALearn.fitted_model['prop'][0].params

