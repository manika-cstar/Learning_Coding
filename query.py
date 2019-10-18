#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:58:48 2019

@author: miny
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)

#gradient descent

alpha=0.1
target_dimesion=2
iterations=1000

t=np.ones([784,2])/np.sqrt(784)
train_data_transpose=np.transpose(train_data)
covariance_matrix=np.cov(train_data_transpose)

for i in range(iterations):
    t[:,0] = t[:,0] + alpha * (train_data_transpose.dot(train_data.dot(t[:,0])))
    t[:,0] = t[:,0] / np.linalg.norm(t[:,0])

for i in range(iterations):
    t[:,1] = t[:,1] + alpha * (train_data_transpose.dot(train_data.dot(t[:,1])))
    t[:,1] = t[:,1] / np.linalg.norm(t[:,1])

train_data_reduced=np.transpose(t).dot(np.transpose(train_data))
train_data_reduced=np.transpose(train_data_reduced)

plt.plot(train_data_reduced[:,0], train_data_reduced[:,1], '.')
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Gradient descent PCA")    

plt.show()

#normal pca

train_data = train_data - np.mean(train_data,axis = 0)
train_data_transpose=np.transpose(train_data)
covariance_matrix=np.cov(train_data_transpose)



eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvectors = eigenvectors[:,:target_dimesion]

train_data_reduced=np.transpose(eigenvectors).dot(np.transpose(train_data))
train_data_reduced=np.transpose(train_data_reduced)


plt.plot(train_data_reduced[:,0],train_data_reduced[:,1],'.')
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Normal PCA")

plt.show()