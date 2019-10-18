#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:53:54 2019

@author: miny
"""

'''Linear Perceptron'''

import numpy
import matplotlib.pyplot as plt 

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# train predictions
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1]]

weights = [1,1,1]

print("Initial weights", weights)


test_data=numpy.transpose(dataset)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [2,2,2]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        '''print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print(weights)'''
    return weights

l_rate = 0.2
n_epoch = 1000
weights = train_weights(dataset, l_rate, n_epoch)
print("Final weights", weights)

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# test predictions
dataset = [[3.396561688,4.400293529,0],
    [1.465489372,2.3621250760,0],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1]]

for row in dataset:
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))
    
'''Logistic regression'''

import numpy
import matplotlib.pyplot as plt 
import math

# Make a prediction with weights
def predict(features, weights):
  z = np.dot(features, weights)
  return 1.0 if sigmoid(z) >= .5 else -1.0

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# train predictions
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1]]

weights = [1,1,1]

print("Initial weights", weights)


test_data=numpy.transpose(dataset)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [1,1,1]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        '''print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print(weights)'''
    return weights

l_rate = 0.2
n_epoch = 1000
gamma=0.1
weights = train_weights(dataset, l_rate, n_epoch)
print("Final weights", weights)

# test predictions
dataset = [[.3396561688,.4400293529,-1],
    [.1465489372,.23621250760,-1],
    [.6922596716,.177106367,1],
    [.8675418651,-.0242068655,1]]


for row in dataset:
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))