import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import datasets

from __future__ import print_function

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
print(train_labels.shape, test_labels.shape)

'''create a new train dataset'''
count=0
for i in range(6000):
    if(train_labels[i]==1 or train_labels[i]==2):
        count=count+1
        '''print(train_data[i])'''
       
num_points = count
dim_points = 28 * 28
new_data = np.empty((num_points, dim_points))
new_labels = np.empty(num_points)

j=0
for i in range(6000):
    if(train_labels[i]==1 or train_labels[i]==2):
        new_labels[j] = train_labels[i]
        new_data[j] = train_data[i]
        j=j+1
        
X_tr = pd.DataFrame(new_data)
X_tr = X_tr.iloc[:,0:]
y_tr = pd.DataFrame(new_labels)
y_tr = y_tr.iloc[:, 0]
X_test=pd.DataFrame(test_data)
X_test = X_test.iloc[:,0:]
y_test=pd.DataFrame(test_labels)
y_test = y_test.iloc[:, 0]

sns.countplot(new_labels)
plt.show()

# Create support vector classifier object
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)

svc = SVC(kernel='linear', random_state=0,C=0.001)

# Train classifier
model = svc.fit(X_tr, y_tr)
y_pred = model.predict(X_test)
print(y_pred)


'''svc = LinearSVC(C=5)
model = svc.fit(X_tr, y_tr)
y_pred = model.predict(X_test)
print(y_pred)'''

print("Starting 5 samples of predictions:")
print("\n")
print("Predicted values:")
print(y_pred[0:5])
print("\n")
print("Actual values:")
print(y_test[0:5])
print("\n")
error=0
for i in range(1000):
    if(y_pred[i]-y_test[i]!=0):
        error=error+1

accuracy_percentage=((1000-error)/1000)*100
#Acuuracy with C=0.001
print("Accuracy percentage= %3.2f percent" %(accuracy_percentage))

#plot with first two eigen vectors of data
cov = np.cov(train_data.T)
eigvals, eigvec = LA.eig(cov)
eigvals = np.real(eigvals)
sorter = eigvals.argsort()
A = np.real(eigvec[:,sorter[-1]])
B = np.real(eigvec[:,sorter[-2]])
a = np.array([A,B])
tran = train_data.dot(a.T)
data = tran.T

color = ['yellow' if c == 0 else 'green' for c in train_labels]
print(data[1:])
plt.scatter(data[0,:],data[1,:], c=color)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(6.6,6.8, num=20)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy,'k-')
plt.axis("off")
plt.show()

#support vectors
s_v=model.support_vectors_
plt.scatter(s_v[:,0], s_v[:,1], color = 'blue')
plt.show()