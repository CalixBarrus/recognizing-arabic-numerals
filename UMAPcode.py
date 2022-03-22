# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tZUw9zzvL3ZC99YVFSG1ZMq5BBC43Yb0
"""

# pip install umap-learn

from umap import UMAP
from datetime import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Do the same thing for the Training dataset 
train = pd.read_csv("Train_Arabic_Digit.txt", header=None, sep=' ')
df_list_train = np.split(train, train[train.isnull().all(1)].index)
df_list_train = [df[1:].reset_index() for df in df_list_train]

# block_lens = [len(df) for df in df_list_train]

train_data = [df_list_train[i].iloc[0] for i in range(1, len(df_list_train))]
train_data = np.array(train_data)
# removing the first column, which is the index of each of the sequences
train_data = train_data[:, 1:]

train_target_number = [i for i in range(10) for _ in range(660)]
train_target_gender = [gender for _ in range(10) for gender in ['m', 'f'] for _ in range(330)]
train_target_gender_bool = [t == 'm' for t in train_target_gender]




#reading test set
test = pd.read_csv("Test_Arabic_Digit.txt", header=None, sep=' ')
df_list_test = np.split(test, test[test.isnull().all(1)].index)
df_list_test = [df[1:].reset_index() for df in df_list_test]

# block_lens = [len(df) for df in df_list]
# plt.hist(block_lens)

test_data = [df_list_test[i].iloc[0] for i in range(1, len(df_list_test))]
test_data = np.array(test_data)
# preserving only the first column, which is the index of each of the sequences
test_data = test_data[:, 1:]

test_target_number = [i for i in range(10) for _ in range(220)]
test_target_gender = [gender for _ in range(10) for gender in ['m', 'f'] for _ in range(110)]
test_target_gender_bool = [t == 'm' for t in test_target_gender]

train_data = np.array([df_list_train[i].iloc[:4] for i in range(1, len(df_list_train))])
train_data = np.array([chunk[:,1:].flatten().flatten() for chunk in train_data])

test_data = np.array([df_list_test[i].iloc[:4] for i in range(1, len(df_list_test))])
test_data = np.array([chunk[:,1:].flatten().flatten() for chunk in test_data])

#training umap
umap_trained = UMAP(n_neighbors = 10,n_components = 2).fit(train_data)
train_data_transformed = umap_trained.embedding_
test_data_transformed = umap_trained.transform(test_data)

plt.figure(figsize=(10, 8))
males = train_data_transformed[train_target_gender_bool]
females = train_data_transformed[np.invert(train_target_gender_bool)]
plt.scatter(males[:,0],males[:,1], c='b', s = .8, label = "Train Males")
plt.scatter(females[:,0],females[:,1], c='y', s = .8, label = "Train Females")
test_males = test_data_transformed[test_target_gender_bool]
plt.scatter(test_males[:,0], test_males[:,1], c = 'r', s = .5, label = "Test Males")
plt.title('UMAP mapping by Gender')
plt.legend()
plt.show()
#implement legend

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data_transformed, train_target_gender_bool)
y_pred = knn.predict(test_data_transformed)
gender_acc = metrics.accuracy_score(y_pred, test_target_gender_bool)
print(f'Gender Accuracy: {gender_acc}')

#basic plot of dimensionality reduction
plt.figure(figsize=(10, 7))
colors = ['b','g','c','m','y', 'orange','teal', 'slateblue','deepskyblue','mediumspringgreen']
for i in range(10):
  j,k = i*220, (i+1)*220
  plt.scatter(train_data_transformed[j:k,0],train_data_transformed[j:k,1], c=colors[i], s = 1.3, label = f'Digit {i}')
plt.legend(loc = 'best')
plt.title('UMAP Mapping by Digit')
plt.show()
#implement legend

#plotting to specifically point out zero digit
#basic plot of dimensionality reduction
plt.figure(figsize=(10, 7))
plt.scatter(train_data_transformed[:660,0],train_data_transformed[:660,1], c='b', s = .8, label = 'Digit 0 Train Data')
plt.scatter(train_data_transformed[660:,0],train_data_transformed[660:,1], c='y', s = .4, label = 'Other Digits Train Data')
plt.scatter(test_data_transformed[:220,0], test_data_transformed[:220,1], c = 'r', s = .8, label = "Digit 0 Test Data")
plt.title('UMAP comparison of Digit 0')
plt.legend(loc = 'best')
plt.show()

#using knn to predict number said
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(train_data_transformed, train_target_number)
y_pred = knn.predict(test_data_transformed)
num_acc = metrics.accuracy_score(y_pred, test_target_number)
print(f'Digit Accuracy: {num_acc}')

