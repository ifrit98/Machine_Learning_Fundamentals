import numpy as np
import pandas as pd #For easy data reading

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

df = pd.read_csv('blood_fat.csv')
N = len(df.index)

df.head(4)

X = df.iloc[:,0:2].as_matrix()
y = df.iloc[:,2].as_matrix().reshape(len(df.index),1)
print(X.shape,':',X[1])
print(y.shape,':',y[1])


def layer():
    l1 = sigmoid(np.dot(X, weight))
    return l1

def loss(predicted):
    l1_loss = y - predicted
    return l1_loss

def backprop_L2Regularization(l1, loss, weight, lamb=10):
    # L2 Regularization expression
    L2 = lamb / (2 * N) + (weight ** 2 / N)

    l1_delta = loss * sigmoid(l1, True)
    weight += L2 * np.dot(X.T, l1_delta)

    return weight

weight = np.random.normal(size=(2,1),loc=0.5,scale=0.5)

epochs = 100
loss_list = []

for step in range(epochs):
    l1_out = layer()
    _loss = loss(l1_out)
    av_loss = abs(np.average(_loss))
    loss_list.append(av_loss)

    weight = backprop_L2Regularization(l1_out,_loss,weight)

    print('Loss at epoch %d: %.3f' %(step,av_loss))