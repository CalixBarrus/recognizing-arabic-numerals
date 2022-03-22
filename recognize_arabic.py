#!/usr/bin/env python
# coding: utf-8

# In[136]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle

import gmmhmm


# In[64]:


def initialize(n_states):
    transmat = np.ones((n_states,n_states))/float(n_states)
    for i in range(n_states):
        transmat[i,:] += sp.random.uniform(-1./n_states,1./n_states,n_states)
        transmat[i,:] /= sum(transmat[i,:])
    startprob = np.ones(n_states)/float(n_states) + sp.random.uniform(-1./n_states,1./n_states,n_states)
    startprob /= sum(startprob)
    return startprob, transmat

def predict(mfcc_array, models):
    M = -np.inf
    ind = 0
    for i,model in enumerate(models):
        if model.score(mfcc_array) > M:
            ind = i
            M = val
    return ind


# # Load Data

# In[134]:


test = pd.read_csv("Test_Arabic_Digit.txt", header=None, sep=' ')
test = np.split(test, test[test.isnull().all(1)].index)
test = [df[1:].reset_index().drop('index', axis=1).values for df in test[1:]]
test = np.array([arr[:4] for arr in test])

train = pd.read_csv("Train_Arabic_Digit.txt", header=None, sep=' ')
train = np.split(train, train[train.isnull().all(1)].index)
train = [df[1:].reset_index().drop('index', axis=1).values for df in train[1:]]
train = np.array([arr[:4] for arr in train])

test_target_number = [i for i in range(10) for _ in range(220)]
test_target_gender = [gender for _ in range(10) for gender in ['m', 'f'] for _ in range(110)]

train_target_number = np.array([i for i in range(10) for _ in range(660)])
train_target_gender = np.array([gender for _ in range(10) for gender in ['m', 'f'] for _ in range(330)])


# In[ ]:


# If train() takes data as single argument instead of 2
# train_number = np.hstack((train_data, train_target_number.reshape(-1,1)))
# train_gender = np.hstack((train_data, train_target_gender.reshape(-1,1)))


# In[128]:


plt.hist([len(df) for df in test+train])
plt.title("Block Lengths")
plt.show()


# # Train HMMs

# In[135]:


numbers = np.array([train[train_target_number == num,:] for num in np.unique(train_target_number)])
number_models = []
gender_models = []

for single_number_data in numbers:
    # Train GMMHMM on single_number_data
    startprob, transmat = initialize(5)
    hmm = gmmhmm.GMMHMM(n_components=5, n_mix=3, transmat=transmat, startprob=startprob, cvtype='diag')
    print(single_number_data.shape)
    hmm.covars_prior = 0.01
    hmm.fit(single_number_data, init_params='mc', var=0.1)
    number_models.append(hmm)


# In[139]:


for i,model in enumerate(number_models):
    with open(f"model_digit_{i}.pk", 'wb') as fout:
        pickle.dump(model,fout)


# # Test HMMs

# In[ ]:




