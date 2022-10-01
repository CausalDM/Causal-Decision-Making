#!/usr/bin/env python
# coding: utf-8

# # Meta Bandits
# 
# Recently, a new class of bandit learning algorithms, named meta Bandits, has been developed to utilize learned information to accelerate learning new tasks and share information efficiently across different tasks from a perspective of meta-learning, which is also known as \textit{learning how to learn} \citep{kveton2021meta, hong2021hierarchical, wan2021metadata, basu2021no}. 
# 
# ## Problem Setting
# Most related literature on meta Bandits is TS-based, and all focus on sharing knowledge across a large number of relatively simple bandit tasks, such as MAB. For example, consider the following medical application scenario. Patients need to learn their own optimal treatment among a few (i,e., $K$) treatments. Here each patient is a task, and the learning process of each patient is simply a $K$-Armed Bandit. Slightly different from the notation in other sections of this report, here we denote $\boldsymbol{\mu}_j$ is a $K$-dimensional vector, where $\mu_{j,a}$ is the expected reward of action $a$ for task $j$.
# 
# 

# ## Real Data
# **1. MovieLens**
# 
# Movie Lens is a website that helps users find the movies they like and where they will rate the recommended movies. [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) is a dataset including the observations collected in an online movie recommendation experiment and is widely used to generate data for online bandit simulation studies. The goal of the simulation studies below is to learn the reward distribution of different movie genres and hence to recommend the optimal movie genres to the users to optimize the cumulative user satisfaction. In other words, every time a user visits the website, the agent will recommend a movie genre ($A_t$) to the user, and then the user will give a rating ($R_t$) to the genre recommended. We assume that users' satisfaction is fully reflected through the ratings. Therefore, the ultimate goal of the bandit algorithms is to optimize the cumulative ratings received by finding and recommending the optimal movie genre that will receive the highest rating. In this chapter, we mainly focus on the top 5 Genres, including 
# 
# - **Comedy**: $a=0$,
# - **Drama**: $a=1$,
# - **Action**: $a=2$,
# - **Thriller**: $a=3$,
# - **Sci-Fi**: $a=4$.
# 
# Therefore, $K=5$. For each user, feature information, including age, gender and occupation, are available:
# 
# - **age**: numerical, from 18 to 56,
# - **gender**: binary, =1 if male,
# - **college/grad student**: binary, =1 if a college/grad student,
# - **executive/managerial**: binary, =1 if a executive/managerial,
# - **academic/educator**: binary, =1 if an academic/educator,
# - **technician/engineer**: binary, =1 if a technician/engineer,
# - **writer**: if a writer, then all the previous occupation-related variables = 0 (baseline).
# 
# Furthermore, there are two different types of the reward $R_t$:
# 
# - **Gaussian Bandit**: $R_t$ is a numerical variable, taking the value of $\{1,2,3,4,5\}$, where 1 is the least satisfied and 5 is the most satisfied.
# - **Bernoulli Bandit**: $R_t$ is a binary variable, =1 if the rating is higher than 3.
# 
# In the following, we evaluated the empirical performance of the supported algorithms on the MovieLens dataset under either the Gaussian bandit or Bernoulli bandit settings.

# In[ ]:




