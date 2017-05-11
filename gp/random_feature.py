import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py

def data(n=100):
    X = 4*np.pi*np.random.rand(n,1)
    y = 2*np.sin(X)+np.cos(np.sqrt(2)*X)+np.random.randn(n,1)*0.2
    return X, y


def random_feature(X, m, sgm=1., seed=1):
    # RBF kernel
    np.random.seed(seed)
    n, d = X.shape
    W = np.random.randn(d, m)*sgm
    b = 2*np.pi*np.random.rand(1,m)
    return np.sqrt(2./m)*np.cos(np.matmul(X,W)+b)

def rbf(X, sgm=1.):
    n, d = X.shape
    return np.exp(-np.sum((X.reshape(1,n,d)-X.reshape(n,1,d))**2,axis=2)/2)

def main():
    X,y = data()
    py.iplot([{'x':X[:,0], 'y':y[:,0], 'mode':'markers'}])
