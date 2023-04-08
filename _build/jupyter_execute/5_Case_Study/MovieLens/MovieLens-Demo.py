#!/usr/bin/env python
# coding: utf-8

# ## Causal Effect Learning

# In[1]:


import numpy as np
import causaldm.causaldm.learners.Online.CMAB._env_realCMAB as env
data = env.get_movielens()


# In[2]:


data.keys()


# In[4]:


data_ML = data['Individual']


# In[5]:


data_ML.keys()


# In[6]:


data_ML[48].keys()


# In[7]:


data_ML[48]['Comedy']


# In[8]:


data_ML[48]['Drama']['complete']


# In[118]:


userinfo_index = np.array([3,9,11,12,13,14])
data_ML[48]['Drama']['complete'].iloc[0,userinfo_index]


# In[119]:


data_ML[48]['Drama']['Reward']


# In[120]:


users_index = data_ML.keys()
movie_generes = ['Comedy', 'Drama', 'Action', 'Thriller', 'Sci-Fi']


# In[121]:


data_CEL = {}
 
# initialize the final data we'll use in Causal Effect Learning
for i in movie_generes:
    data_CEL[i] = None   


# In[122]:


import pandas as pd
for movie_genere in movie_generes:
    for user in users_index:
        data_CEL[movie_genere] = pd.concat([data_CEL[movie_genere] , data_ML[user][movie_genere]['complete']])


# In[123]:


len(data_CEL['Comedy'])


# In[66]:


len(data_CEL['Drama'])


# In[67]:


len(data_CEL['Action'])


# In[68]:


len(data_CEL['Thriller'])


# In[69]:


len(data_CEL['Sci-Fi'])


# In[70]:


len(np.unique(data_CEL['Comedy']['user_id'])) # the total number of users


# In[71]:


users_index


# In[124]:


data_CEL['Comedy']


# ### the entire dataset of interst

# In[125]:


data_CEL_all = data_CEL['Comedy']
for movie_genere in movie_generes[1:5]:
    data_CEL_all = pd.concat([data_CEL_all, data_CEL[movie_genere]])


# In[126]:


len(data_CEL_all)


# In[ ]:





# In[127]:


data_CEL_all.iloc[np.where(data_CEL_all['Comedy']==1)[0],userinfo_index]


# In[ ]:





# ### sample 1% of the data by users

# In[108]:


len(data_CEL['Comedy'])


# In[128]:


np.where(data_CEL['Action']['user_id'] == user)[0]


# In[135]:


int(np.ceil(10.5))


# In[139]:


random.sample(np.array([1,2,3]),2)


# In[137]:


import random
random.sample(np.where(data_CEL['Action']['user_id'] == user)[0], int(np.ceil(len(np.where(data_CEL['Action']['user_id'] == user)[0]))))


# In[116]:


# randomly select 1% of samples from all users
data_CEL_sample = data_CEL['Comedy']
for movie_genere in movie_generes[1:5]:

    data_CEL_sample = pd.concat([data_CEL_sample, data_CEL[movie_genere]])


for user in users_index:       
    len(np.where(data_CEL['Action']['user_id'] == user)[0])
    


# In[115]:


for user in users_index:
    print(user)


# ### nonlinear model fitting

# In[76]:



models_CEL = {}
 
# initialize the models we'll fit in Causal Effect Learning
for i in movie_generes:
    models_CEL[i] = None   


# In[77]:


from lightgbm import LGBMRegressor
for movie_genere in movie_generes: 
    models_CEL[movie_genere] = LGBMRegressor(max_depth=3)
    models_CEL[movie_genere].fit(data_CEL_all.iloc[np.where(data_CEL_all[movie_genere]==1)[0],userinfo_index],data_CEL_all.iloc[np.where(data_CEL_all[movie_genere]==1)[0],2] )


# In[78]:


models_CEL['Comedy'].predict(data_CEL_all.iloc[np.where(data_CEL_all['Comedy']==1)[0],userinfo_index])


# In[79]:


min(data_CEL_all['age'])


# In[80]:


max(data_CEL_all['age'])


# In[81]:


# record thev estimated expected reward for each movie genere, under each possible combination of state variable
age_range = np.linspace(min(data_CEL_all['age']),max(data_CEL_all['age']),int(max(data_CEL_all['age'])-min(data_CEL_all['age'])+1)).astype(int)
print(age_range)


# In[82]:


import itertools

gender = np.array([0,1])
occupation_college = np.array([0,1])
occupation_executive = np.array([0,1])
occupation_other  = np.array([0,1])
occupation_technician = np.array([0,1])

# result contains all possible combinations.
combinations = pd.DataFrame(itertools.product(age_range,gender,occupation_college,
                                              occupation_executive,occupation_other,occupation_technician))


# In[83]:


combinations.columns =['age','gender','occupation_college', 'occupation_executive','occupation_other','occupation_technician']
combinations


# In[84]:


len(models_CEL['Comedy'].predict(combinations))


# In[85]:


movie_genere


# In[86]:


models_CEL['Comedy'].predict(combinations)


# In[87]:


values = np.zeros((5,1312))
i=0
for movie_genere in movie_generes:
    values[i,:] = models_CEL[movie_genere].predict(combinations)
    i=i+1
    print(values)


# In[88]:


result_CEL = combinations
i=0
for movie_genere in movie_generes:
  #values = models_CEL[movie_genere].predict(combinations)
  result_CEL.insert(len(result_CEL.columns), movie_genere, values[i,:])
  i=i+1


# In[89]:


result_CEL


# In[ ]:





# In[90]:


result_CEL.to_csv('/Users/alinaxu/Documents/CDM/Causal-Decision-Making/5_Case_Study/MovieLens/result_CEL_nonlinear.csv')


# In[91]:


# read the result file
result_CEL_nonlinear = pd.read_csv('/Users/alinaxu/Documents/CDM/Causal-Decision-Making/5_Case_Study/MovieLens/result_CEL_nonlinear.csv')
result_CEL_nonlinear = result_CEL_nonlinear.drop(result_CEL_nonlinear.columns[0], axis=1)
result_CEL_nonlinear


# #### Analysis

# In[92]:


np.where(result_CEL_nonlinear['gender']==0)[0]


# In[93]:


result_CEL_nonlinear.iloc[np.where(result_CEL_nonlinear['gender']==0)[0],6:11]


# In[94]:


# calculate the expected reward of Comedy for female
TE_female=result_CEL_nonlinear.iloc[np.where(result_CEL_nonlinear['gender']==0)[0],6:11]/(41*(2**4))
TE_female=pd.DataFrame(TE_female.sum(axis=0))
TE_female.columns =['Expected Rating']


# In[95]:


TE_female


# In[ ]:





# In[96]:


# calculate the expected reward of Comedy for female
TE_male=result_CEL_nonlinear.iloc[np.where(result_CEL_nonlinear['gender']==1)[0],6:11]/(41*(2**4))
TE_male=pd.DataFrame(TE_male.sum(axis=0))
TE_male.columns =['Expected Rating']


# In[97]:


TE_male


# In[ ]:





# ### linear model fitting

# In[308]:



models_CEL_linear = {}
 
# initialize the models we'll fit in Causal Effect Learning
for i in movie_generes:
    models_CEL_linear[i] = None   


# In[309]:


from sklearn.linear_model import LinearRegression
for movie_genere in movie_generes: 
    models_CEL_linear[movie_genere] = LinearRegression()
    models_CEL_linear[movie_genere].fit(data_CEL_all.iloc[np.where(data_CEL_all[movie_genere]==1)[0],userinfo_index],data_CEL_all.iloc[np.where(data_CEL_all[movie_genere]==1)[0],2] )


# In[310]:


models_CEL_linear['Comedy'].predict(data_CEL_all.iloc[np.where(data_CEL_all['Comedy']==1)[0],userinfo_index])


# In[313]:


# record thev estimated expected reward for each movie genere, under each possible combination of state variable
age_range = np.linspace(min(data_CEL_all['age']),max(data_CEL_all['age']),int(max(data_CEL_all['age'])-min(data_CEL_all['age'])+1)).astype(int)
print(age_range)


# In[314]:


import itertools

gender = np.array([0,1])
occupation_college = np.array([0,1])
occupation_executive = np.array([0,1])
occupation_other  = np.array([0,1])
occupation_technician = np.array([0,1])

# result contains all possible combinations.
combinations = pd.DataFrame(itertools.product(age_range,gender,occupation_college,
                                              occupation_executive,occupation_other,occupation_technician))


# In[315]:


combinations.columns =['age','gender','occupation_college', 'occupation_executive','occupation_other','occupation_technician']
combinations


# In[316]:


values = np.zeros((5,1312))
i=0
for movie_genere in movie_generes:
  values[i,:] = models_CEL_linear[movie_genere].predict(combinations)
  i=i+1
  print(values)


# In[317]:


result_CEL_linear = combinations
i=0
for movie_genere in movie_generes:
  #values = models_CEL[movie_genere].predict(combinations)
  result_CEL_linear.insert(len(result_CEL_linear.columns), movie_genere, values[i,:])
  i=i+1


# In[318]:


result_CEL_linear


# In[ ]:





# In[319]:


result_CEL_linear.to_csv('drive/MyDrive/Causal-Decision-Making/Data-Demo/result_CEL_linear.csv')


# In[320]:


# read the result file
result_CEL_linear = pd.read_csv('drive/MyDrive/Causal-Decision-Making/Data-Demo/result_CEL_linear.csv')
result_CEL_linear = result_CEL_linear.drop(result_CEL_linear.columns[0], axis=1)
result_CEL_linear


# #### Analysis

# In[322]:


result_CEL_linear.iloc[np.where(result_CEL_linear['gender']==0)[0],6:11]


# In[323]:


# calculate the expected reward of Comedy for female
TE_female_linear=result_CEL_linear.iloc[np.where(result_CEL_linear['gender']==0)[0],6:11]/(41*(2**4))
TE_female_linear=pd.DataFrame(TE_female_linear.sum(axis=0))
TE_female_linear.columns =['Expected Rating']


# In[324]:


TE_female_linear


# In[326]:


# calculate the expected reward of Comedy for female
TE_male_linear=result_CEL_linear.iloc[np.where(result_CEL_linear['gender']==1)[0],6:11]/(41*(2**4))
TE_male_linear=pd.DataFrame(TE_male_linear.sum(axis=0))
TE_male_linear.columns =['Expected Rating']


# In[327]:


TE_male_linear


# In[ ]:




