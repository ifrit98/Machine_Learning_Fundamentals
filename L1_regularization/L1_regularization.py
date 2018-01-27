import numpy as np
import pandas as pd #For easy data reading
import matplotlib.pyplot as plt

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def layer():
    l1 = sigmoid(np.dot(X, weight))
    return l1

def loss(predicted):
    l1_loss = y - predicted
    return l1_loss

def backprop_L1Regularization(l1, loss, weight, lamb=10):
    # L1 Regularization expression
    L1 = lamb / (2 * N) + (weight / N)
    l1_delta = loss * sigmoid(l1, True)
    weight += L1 * np.dot(X.T, l1_delta)

    return weight


datafile = pd.read_csv('blood_fat.csv')
N = len(datafile.index)
datafile.head(4)

X = datafile.iloc[:,0:2].as_matrix()
y = datafile.iloc[:,2].as_matrix().reshape(len(datafile.index),1)
print(X.shape,':',X[1])
print(y.shape,':',y[1])

weight = np.random.normal(size=(2,1),loc=0.5,scale=0.5)

epochs = 100
loss_list = []

for step in range(epochs):
    l1_out = layer()
    _loss = loss(l1_out)
    av_loss = abs(np.average(_loss))
    loss_list.append(av_loss)

    weight = backprop_L1Regularization(l1_out,_loss,weight)
    print('Loss at epoch %d: %.2f' %(step,av_loss))

plt.title('Loss vs epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.scatter(np.arange(0,len(loss_list)),loss_list)
plt.show()
