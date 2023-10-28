#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import copy


# # Read in Data

# In[2]:


User = pd.read_csv('users.dat', sep='::', names = ["user_id","gender","age","occupation","zip_code"]  , encoding='latin-1')
Occupation = ["other","academic/educator","artist","clerical/admin","college/grad student","customer service","doctor/health care","executive/managerial","farmer"
                 ,"homemaker","K-12 student","lawyer","programmer","retired","sales/marketing","scientist","self-employed","technician/engineer"
                 ,"tradesman/craftsman","unemployed","writer"]
User = np.array(User)
for row in range(User.shape[0]):
    User[row,3] = Occupation[User[row,3]]
User = pd.DataFrame(User,columns = ["user_id","gender","age","occupation","zip_code"])


# In[3]:


data = pd.read_csv('ratings.dat', sep='::', names = ["user_id","movie_id","rating","timestamp"]  , encoding='latin-1')


# In[4]:


Item = pd.read_csv('movies.dat', sep='::', names = ["movie_id","movie_title","Genres"]  , encoding='latin-1')
Genres = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir"
          ,"Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
New = []
for row in range(Item.shape[0]):
    temp = [0]*len(Genres)
    idx_of_Genres = [Genres.index(item) for item in Item.iloc[row,:].Genres.split('|')]
    for idx in idx_of_Genres:
        temp[idx] = 1
    A = [Item.iloc[row,:].movie_id,Item.iloc[row,:].movie_title]+temp
    New.append(A)
Item = pd.DataFrame(New, columns = ['movie_id','movie_title']+Genres)


# In[5]:


User = User.drop(["zip_code"], axis=1)
data = data.drop(["timestamp"], axis=1)
Final_Data = pd.merge(data, User, on="user_id")
Final_Data = pd.merge(Final_Data, Item, on="movie_id")
Final_Data.head()


# In[6]:


len(Final_Data)


# In[7]:


with open("MovieLens_1M.txt", "wb") as fp:
     pickle.dump(Final_Data, fp)


# # Keep only Top 5 Occupations and Top 5 Genres

# In[8]:


with open("MovieLens_1M.txt", "rb") as fp:
    MovieLens_1M = pickle.load(fp) 


# In[9]:


#Top 5 popular genere
popular_genre = list(pd.DataFrame(MovieLens_1M.iloc[:, 7:].sum(axis = 0)).sort_values(0, ascending=0).iloc[:5].index)
popular_genre


# In[10]:


#Top 5 popular occupation
popular_occupation = list(MovieLens_1M.occupation.value_counts()[:6].index)
popular_occupation = ['college/grad student', 'executive/managerial', 'academic/educator', 'technician/engineer', 'writer']


# In[11]:


col = list(MovieLens_1M.columns[:6]) + popular_genre
MovieLens_1M_popular = MovieLens_1M[MovieLens_1M.occupation.isin(popular_occupation)][col]
MovieLens_1M_popular = MovieLens_1M_popular[MovieLens_1M_popular.iloc[:,6:12].sum(axis=1)>0][col]
MovieLens_1M_popular.head() 


# In[12]:


MovieLens_1M_popular.occupation.value_counts()


# In[13]:


len(MovieLens_1M_popular)


# # Convert Gender/Occupation to Dummy Variables

# In[14]:


MovieLens_1M_popular = pd.get_dummies(MovieLens_1M_popular, columns=['gender','occupation'])
MovieLens_1M_popular = MovieLens_1M_popular.drop(columns=['gender_F'])


# # Convert Observations with Movies under Multiple Genres

# In[15]:


Single_Genre = MovieLens_1M_popular[MovieLens_1M_popular.iloc[:,4:9].sum(axis=1)==1]
Multi_Genre = MovieLens_1M_popular[MovieLens_1M_popular.iloc[:,4:9].sum(axis=1)>1]


# In[16]:


Multiple = []
B = np.array(Multi_Genre)
for row in range(B.shape[0]):
    for i in range(5):
        if B[row,4:9][i] == 1:
            temp = np.zeros((5,))
            temp[i] = 1
            Multiple.append(list(B[row,:4])+list(temp)+list(B[row,9:]))
Multiple = np.array(Multiple)
Multiple = pd.DataFrame(Multiple)
Multiple.columns = Single_Genre.columns


# In[17]:


MovieLens = pd.concat([Single_Genre, Multiple], axis = 0)


# # Subset Users with at Least 500 Observations

# In[18]:


temp = MovieLens.groupby("user_id").nunique()
users = temp.index[temp.movie_id > 500]
MovieLens = MovieLens[MovieLens.user_id.isin(users)]


# In[19]:


import pickle
with open("MovieLens_Cleaned_1M.txt", "wb") as fp:
     pickle.dump(MovieLens, fp)


# # Formating the Dataset

# In[20]:


from scipy.linalg import block_diag
task = MovieLens.user_id.unique()
arm = ['Comedy', 'Drama', 'Action', 'Thriller', 'Sci-Fi']
MovieLens_Bandit = dict()
MovieLens_Bandit['Individual'] = dict()
for ind in task:
    MovieLens_Bandit['Individual'][ind] = dict() 
    temp = MovieLens[MovieLens.user_id == ind]
    a = [1]+list(temp[['age','gender_M','occupation_college/grad student', 'occupation_executive/managerial', 'occupation_academic/educator', 'occupation_technician/engineer']].iloc[0,:])
    a = block_diag(a,a,a,a,a)
    MovieLens_Bandit['Individual'][ind]['Phi'] = a
    for act in arm:
        MovieLens_Bandit['Individual'][ind][act] = dict()
        MovieLens_Bandit['Individual'][ind][act]['complete'] = temp[temp[act] == 1]
        ratings = temp[temp[act] == 1].rating.value_counts().keys().tolist()
        counts = temp[temp[act] == 1].rating.value_counts().tolist()
        MovieLens_Bandit['Individual'][ind][act]['Reward'] = [ratings,counts]

Xs = np.zeros((len(task),a.shape[0],a.shape[1]))
for i in range(len(task)):
    Xs[i,:,:] = MovieLens_Bandit['Individual'][task[i]]['Phi']
MovieLens_Bandit['Xs'] = Xs


# In[21]:


def get_sum_r(Reward):
    return np.dot(np.array(Reward[0]),np.array(Reward[1]))
denom = [sum([sum(MovieLens_Bandit['Individual'][key][act]['Reward'][1]) for key in MovieLens_Bandit['Individual'].keys()]) for act in arm]
A = [sum([get_sum_r(MovieLens_Bandit['Individual'][key][act]['Reward']) for key in MovieLens_Bandit['Individual'].keys()]) for act in arm]
mean_ri = np.array([int(a) / int(b) for a,b in zip(A, denom)])
MovieLens_Bandit['mean_ri'] = mean_ri
mean_ri


# In[22]:


with open("MovieLens_Cleaned_1M.txt", "rb") as fp:
     MovieLens = pickle.load(fp) 


# In[23]:


N = MovieLens_Bandit['Xs'].shape[0]
K = MovieLens_Bandit['Xs'].shape[1]
p = MovieLens_Bandit['Xs'].shape[2]
TASK = list(MovieLens_Bandit['Individual'].keys())
New = np.zeros((MovieLens.shape[0],p))
for row in range(MovieLens.shape[0]):
    user= MovieLens.iloc[row,:].user_id
    a = np.where(np.array(MovieLens.iloc[row,4:9]))[0][0]
    New[row,:] = MovieLens_Bandit['Individual'][user]['Phi'][a,:]
MovieLens_Bandit['standardized_Xs'] = copy.deepcopy(MovieLens_Bandit['Xs'])
for i in range(MovieLens_Bandit['Xs'].shape[0]):
    MovieLens_Bandit['standardized_Xs'][i] /= np.std(New, axis=0)


# In[24]:


MovieLens_Bandit['Xs'].shape


# In[32]:


with open("MovieLens_MTTS_1M_Gaussian.txt", "wb") as fp:
     pickle.dump(MovieLens_Bandit, fp)


# In[31]:


MovieLens_Bandit['standardized_Xs']


# ### Bernoulli Bandit

# In[33]:


MovieLens.rating = (MovieLens.rating > 3.0).astype(float)


# In[34]:


from scipy.linalg import block_diag
task = MovieLens.user_id.unique()
arm = ['Comedy', 'Drama', 'Action', 'Thriller', 'Sci-Fi']
MovieLens_Bandit = dict()
MovieLens_Bandit['Individual'] = dict()
for ind in task:
    MovieLens_Bandit['Individual'][ind] = dict() 
    temp = MovieLens[MovieLens.user_id == ind]
    a = [1]+list(temp[['age','gender_M','occupation_college/grad student', 'occupation_executive/managerial', 'occupation_academic/educator', 'occupation_technician/engineer']].iloc[0,:])
    a = block_diag(a,a,a,a,a)
    MovieLens_Bandit['Individual'][ind]['Phi'] = a
    for act in arm:
        MovieLens_Bandit['Individual'][ind][act] = dict()
        MovieLens_Bandit['Individual'][ind][act]['complete'] = temp[temp[act] == 1]
        ratings = temp[temp[act] == 1].rating.value_counts().keys().tolist()
        counts = temp[temp[act] == 1].rating.value_counts().tolist()
        MovieLens_Bandit['Individual'][ind][act]['Reward'] = [ratings,counts]

Xs = np.zeros((len(task),a.shape[0],a.shape[1]))
for i in range(len(task)):
    Xs[i,:,:] = MovieLens_Bandit['Individual'][task[i]]['Phi']
MovieLens_Bandit['Xs'] = Xs


# In[35]:


N = MovieLens_Bandit['Xs'].shape[0]
K = MovieLens_Bandit['Xs'].shape[1]
p = MovieLens_Bandit['Xs'].shape[2]
TASK = list(MovieLens_Bandit['Individual'].keys())
New = np.zeros((MovieLens.shape[0],p))
for row in range(MovieLens.shape[0]):
    user= MovieLens.iloc[row,:].user_id
    a = np.where(np.array(MovieLens.iloc[row,4:9]))[0][0]
    New[row,:] = MovieLens_Bandit['Individual'][user]['Phi'][a,:]
MovieLens_Bandit['standardized_Xs'] = copy.deepcopy(MovieLens_Bandit['Xs'])
for i in range(MovieLens_Bandit['Xs'].shape[0]):
    MovieLens_Bandit['standardized_Xs'][i] /= np.std(New, axis=0)


# In[36]:


with open("MovieLens_MTTS_1M_Bernoulli.txt", "wb") as fp:
     pickle.dump(MovieLens_Bandit, fp)


# In[37]:


MovieLens_Bandit['standardized_Xs']


# In[32]:


MovieLens.age.min()


# In[33]:


MovieLens.age.max()


# In[34]:


MovieLens.iloc[:,-4:].sum(axis = 1).unique()


# In[ ]:




