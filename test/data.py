import os
import numpy as np
import pandas as pd


def data1(n=50, sgm=0.3):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.sin(X)[:, 0] + np.random.randn(n) * sgm
    return X, y


def syn(n=50):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.cos(X[:, 0] / 6) + 0.3 * np.cos(X[:, 0] / 3) * np.random.randn(n)
    return X, y


def data2(n=50):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.cos(X)[:, 0]
    return X, y


def twin(n=100):
    theta = np.random.rand(n) * np.pi
    theta[:n//2] += np.pi
    eps = np.random.randn(n)*0.1
    X = np.stack([(1+eps)*np.cos(theta), (1+eps)*np.sin(theta)], 1)
    X[n//2:] += [[0.4, -0.2]]
    X[:n//2] += [[-0.4, 0.2]]
    y = np.ones(n)
    y[n//2:] *= -1
    return X, y

def twoGaus(n=100):
    X = np.random.randn(n, 2)
    X[:n//2] += [0,1]
    X[n//2:] += [0,-1]
    y = np.ones(n)
    y[n//2:] *= -1
    return X, y

def threeGaus(n=100):
    X = np.random.randn(n, 2)*0.3
    X[:n//4] += [-1, 1]
    X[n//4:n//2] += [-1, -1]
    X[n//2:] += [1, 0]
    y = np.ones(n)
    y[n//2:] *= -1
    return X, y

def download(filename, url):
    from pycurl import Curl
    c = Curl()
    with open(filename, 'wb') as f:
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()

def australian():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale'
    filename = "australian_scale"
    if filename not in os.listdir(os.path.dirname(__file__)):
        download(filename, url)

    with open(filename) as f:
        data = [[v.split(':') for v in line.split()] for line in f.readlines()]

    y, x = zip(*[(int(line[0][0]), line[1:]) for line in data])
    x = pd.DataFrame([{int(v[0]): float(v[1]) for v in line} for line in x])
    x = x.loc[:, x.notnull().all(axis=0)]
    return np.array(x), np.array(y)
