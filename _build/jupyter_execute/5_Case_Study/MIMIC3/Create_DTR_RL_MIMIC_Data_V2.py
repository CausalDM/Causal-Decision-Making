#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
mimic3 = pd.read_csv("subset_rl_data_final_cont.csv")


# In[2]:


mimic3_base = mimic3[['icustayid','bloc', 'Glucose','PaO2_FiO2',
                           'iv_input', 'SOFA','died_within_48h_of_out_time']]
mimic3_base['died_within_48h_of_out_time'] = - 2 * np.array(mimic3_base['died_within_48h_of_out_time']) + 1
mimic3_base.columns = ['icustayid', 'bloc','Glucose','PaO2_FiO2','IV_Input', 'SOFA','Died_within_48H']
mimic3_base['IV_Input'] = mimic3_base['IV_Input'].apply(lambda x: 1 if x >=1 else 0)
mimic3_base


# In[3]:


mimic_final = mimic3_base[mimic3_base['SOFA']<=12]


# In[4]:


mimic_final


# ## Generating 3-stage-DTR dataset

# In[5]:


selected_IDs = mimic_final.icustayid.value_counts()
selected_IDs = selected_IDs[selected_IDs>=3].index.tolist()


# In[6]:


mimic_final[mimic_final.icustayid==1006].iloc[:3,:]


# In[7]:


DTR_data = [np.concatenate(np.array(mimic_final[mimic_final.icustayid==ind].iloc[:3,:])) for ind in selected_IDs]


# In[8]:


varname = mimic_final.columns.tolist()


# In[9]:


varname_formatted = []
for bloc in [1,2,3]:
    varname_formatted+=[i+'_'+str(bloc) for i in varname]
DTR_data = pd.DataFrame(DTR_data, columns = varname_formatted)
DTR_data = DTR_data.drop(columns = ['bloc_1','bloc_2','bloc_3','Died_within_48H_1','Died_within_48H_2','icustayid_2','icustayid_3'])
DTR_data.rename(columns = {'Died_within_48H_3':'Died_within_48H'}, inplace = True)


# In[10]:


DTR_data.to_csv (r'mimic3_DTR_3stage_V2.csv', index = False, header=True)


# In[11]:


DTR_data.IV_Input_3.value_counts()


# In[12]:


DTR_data.head()


# ## Generating 2-stage-mediated DTR dataset

# In[13]:


selected_IDs = mimic_final.icustayid.value_counts()
selected_IDs = selected_IDs[selected_IDs>=4].index.tolist()
mrl_data = mimic_final.copy()


# In[14]:


mediator = [np.array(mrl_data[mrl_data.icustayid==ind].SOFA)[0:3] for ind in selected_IDs]
state = [np.concatenate(np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','PaO2_FiO2']])[:3,:]) for ind in selected_IDs]
action = [np.array(mrl_data[mrl_data.icustayid==ind].IV_Input)[0:3] for ind in selected_IDs]
reward = [mrl_data[mrl_data.icustayid==ind].Died_within_48H.unique().tolist() for ind in selected_IDs]


# In[15]:


mediator = pd.DataFrame(mediator, columns = ['SOFA_1','SOFA_2','SOFA_3'], index = selected_IDs)


# In[16]:


state = pd.DataFrame(state, columns = ['Glucose_1','PaO2_FiO2_1','Glucose_2','PaO2_FiO2_2','Glucose_3','PaO2_FiO2_3'], index = selected_IDs)


# In[17]:


action = pd.DataFrame(action, columns = ['IV_Input_1','IV_Input_2','IV_Input_3'], index = selected_IDs)


# In[18]:


reward = pd.DataFrame(reward, columns = ['Died_within_48H'], index = selected_IDs)


# In[19]:


MDTR_data = {'state':state, 'action': action, 'mediator': mediator, 'reward': reward}
import pickle
with open('mimic3_MDTR_data_dict_3stage_V2.pickle', 'wb') as handle:
    pickle.dump(MDTR_data, handle)
    


# In[20]:


MDTR_df = pd.concat([state, action, mediator, reward], axis = 1)
MDTR_df = MDTR_df[['Glucose_1', 'PaO2_FiO2_1','IV_Input_1', 'SOFA_1', 'Glucose_2',
                   'PaO2_FiO2_2', 'IV_Input_2', 'SOFA_2', 'Glucose_3', 
                   'PaO2_FiO2_3', 'IV_Input_3', 'SOFA_3','Died_within_48H']]
MDTR_df.head() 


# In[21]:


MDTR_df.to_csv (r'mimic3_MDTR_3stage_V2.csv', index = False, header=True)


# ## Generating MRL dataset

# In[22]:


selected_IDs = mimic_final.icustayid.value_counts()
selected_IDs = selected_IDs[selected_IDs>=2].index.tolist()
mrl_data = mimic_final.copy()


# In[23]:


mrl_data[mrl_data.icustayid==1006]


# In[24]:


mediator = [np.array(mrl_data[mrl_data.icustayid==ind].SOFA)[:] for ind in selected_IDs]
next_state = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','PaO2_FiO2']])[:,:] for ind in selected_IDs]
state = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','PaO2_FiO2']])[:,:] for ind in selected_IDs]
action = [np.array(mrl_data[mrl_data.icustayid==ind].IV_Input)[:] for ind in selected_IDs]
reward = [np.array([0]*(len(np.array(mrl_data[mrl_data.icustayid==ind].IV_Input)[:])-1)+mrl_data[mrl_data.icustayid==ind].Died_within_48H.unique().tolist()) for ind in selected_IDs]
ID =  [np.array(mrl_data[mrl_data.icustayid==ind][['icustayid','bloc']])[:,:] for ind in selected_IDs]
s0 = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','PaO2_FiO2']])[0,:] for ind in selected_IDs]
time_idx = [np.array(mrl_data[mrl_data.icustayid==ind].bloc)[:] for ind in selected_IDs]

s0 = np.array(s0)
state = np.vstack(state)
next_state = np.vstack(next_state)
mediator = np.hstack(mediator).reshape((-1,1))
action = np.hstack(action)
reward =np.hstack(reward)
ID = np.vstack(ID)
time_idx = np.hstack(time_idx)


# In[25]:


MRL_data = {'s0': s0,'state':state, 'action': action, 'mediator': mediator, 'reward': reward, 'next_state': next_state, 'time_idx': time_idx}
import pickle
with open('mimic3_MRL_data_dict_V2.pickle', 'wb') as handle:
    pickle.dump(MRL_data, handle)
    


# In[26]:


state = pd.DataFrame(state, columns = ['Glucose','PaO2_FiO2'])
action = pd.DataFrame(action, columns = ['IV_Input'])
mediator = pd.DataFrame(mediator, columns = ['SOFA'])
next_state = pd.DataFrame(next_state, columns = ['next_Glucose','next_PaO2_FiO2'])
reward = pd.DataFrame(reward, columns = ['Died_within_48H'])
ID = pd.DataFrame(ID, columns = ['icustayid','bloc'])
MRL_df = pd.concat([ID, state, action, mediator, next_state, reward], axis = 1)
MRL_df.to_csv (r'mimic3_MRL_df_V2.csv', index = False, header=True)


# In[27]:


MRL_df[MRL_df.icustayid==1006]


# ## Generate RL data

# In[28]:


mrl_data = mimic_final.copy()
#for var in ['Glucose','paO2','PaO2_FiO2','SOFA']:
#    mrl_data[var] = (mrl_data[var] - mrl_data[var].mean()) / mrl_data[var].std()

next_state = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','paO2','PaO2_FiO2','SOFA']])[1:,:] for ind in selected_IDs]
state = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','paO2','PaO2_FiO2','SOFA']])[:-1,:] for ind in selected_IDs]
action = [np.array(mrl_data[mrl_data.icustayid==ind].IV_Input)[:-1] for ind in selected_IDs]
reward = [np.array([0]*(len(np.array(mrl_data[mrl_data.icustayid==ind].IV_Input)[:-1])-1)+mrl_data[mrl_data.icustayid==ind].Died_within_48H.unique().tolist()) for ind in selected_IDs]
ID =  [np.array(mrl_data[mrl_data.icustayid==ind][['icustayid','bloc']])[:-1,:] for ind in selected_IDs]
s0 = [np.array(mrl_data[mrl_data.icustayid==ind][['Glucose','paO2','PaO2_FiO2','SOFA']])[0,:] for ind in selected_IDs]
time_idx = [np.array(mrl_data[mrl_data.icustayid==ind].bloc)[:-1] for ind in selected_IDs]

s0 = np.array(s0)
state = np.vstack(state)
next_state = np.vstack(next_state)
action = np.hstack(action)
reward =np.hstack(reward)
ID = np.vstack(ID)
time_idx = np.hstack(time_idx)


# In[52]:


state = pd.DataFrame(state, columns = ['Glucose','paO2','PaO2_FiO2','SOFA'])
action = pd.DataFrame(action, columns = ['IV_Input'])
next_state = pd.DataFrame(next_state, columns = ['next_Glucose','next_paO2','next_PaO2_FiO2','next_SOFA'])
reward = pd.DataFrame(reward, columns = ['Died_within_48H'])
ID = pd.DataFrame(ID, columns = ['icustayid','bloc'])
RL_df = pd.concat([ID, state, action, next_state, reward], axis = 1)
RL_df.to_csv (r'mimic3_RL_df.csv', index = False, header=True)


# In[53]:


RL_df


# ## Generating 2-stage-mediated DTR dataset

# In[ ]:


DTR_data = [np.concatenate(np.array(mimic_final[mimic_final.icustayid==ind].iloc[:2,:])) for ind in selected_IDs]


# ## Create single stage data

# In[ ]:


mimic3 = pd.read_csv("subset_rl_data_final_cont.csv")
mimic3_base = mimic3[['icustayid', 'Glucose', 'PaO2_FiO2',
                           'iv_input', 'SOFA','died_within_48h_of_out_time']]
mimic3_base['died_within_48h_of_out_time'] = - 2 * np.array(mimic3_base['died_within_48h_of_out_time']) + 1
mimic3_base.columns = ['icustayid', 'Glucose', 'PaO2_FiO2',
                           'IV Input', 'SOFA','Died within 48H'] 
mimic3_base.head(6)

plt.hist(mimic3_base['SOFA'])
mimic_final = mimic3_base[mimic3_base['SOFA']<=12]
plt.hist(mimic_final['SOFA'])

with open('mimic3_multi_stages.pickle', 'wb') as handle:
    pickle.dump(mimic_final, handle)
    
mimic_final.to_csv (r'mimic3_multi_stages.csv', index = False, header=True)

mimic_final


# ----------- Set lag data
lag_k = 1
    
#     new_sofa = list(np.array(mimic_final['SOFA'][lag_k:]) - np.array(mimic_final['SOFA'][:-lag_k]))

new_sofa = np.array(mimic_final['SOFA'][:-lag_k])
mimic3_sample = mimic_final.iloc[lag_k:]
mimic3_sample['SOFA Post'] = new_sofa 
mimic3_data = mimic3_sample.groupby('icustayid').mean().reset_index() 


with open('mimic3_single_stage.pickle', 'wb') as handle:
    pickle.dump(mimic3_data, handle)
    
mimic3_data.to_csv (r'mimic3_single_stage.csv', index = False, header=True)
 
mimic3_data.head(6)

