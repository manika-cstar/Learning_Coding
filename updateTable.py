#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:59:14 2019

@author: miny
"""

from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu()

from __future__ import print_function
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

mnist_train, mnist_trainlabels = transform(train_data, train_labels)
mnist_test, mnist_testlabels = transform(test_data, test_labels)

print(mnist_train.shape, mnist_test.shape)
print(mnist_trainlabels.shape, mnist_testlabels.shape)

mnist_train = mnist_train.reshape(-1,28,28,1)
mnist_test = mnist_test.reshape(-1,28,28,1)

image, label = mnist_train[0], mnist_trainlabels[0]
print(label.shape, label)

num_inputs = 784
num_outputs = 10
num_examples = 6000

c = [[[0 for col in range(28)]for row in range(28)] for x in range(6000)]

c=mx.nd.array(mnist_train)

    
image=mx.nd.array(c[0])
im = mx.nd.tile(image, (1,1,3))
print(im.shape)

d = [[[0 for col in range(28)]for row in range(28)] for x in range(1000)]

d=mx.nd.array(mnist_test)

e = [0 for col in range(1000)]

e=mx.nd.array(mnist_trainlabels)
    


f = [0 for col in range(1000)]

f=mx.nd.array(mnist_testlabels)

import matplotlib.pyplot as plt
plt.imshow(im.asnumpy())
plt.show()

batch_size = 64
train_data = mx.gluon.data.DataLoader(c, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(d, batch_size, shuffle=False)

W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
b = nd.random_normal(shape=num_outputs,ctx=model_ctx)

params = [W, b]

for param in params:
    param.attach_grad()
    
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

sample_y_linear = nd.random_normal(shape=(2,10))
sample_yhat = softmax(sample_y_linear)
print(sample_yhat)

print(nd.sum(sample_yhat, axis=1))

def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
        
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        print(data)
        a=[0]*64
        a=mnist_testlabels[i:i+64]
        
        m = [0 for col in range(64)]
        
        m=mx.nd.array(a)
            
        label = m.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        print(predictions)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 5
learning_rate = .005

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
    
# Define the function to do prediction
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break