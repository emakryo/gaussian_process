import numpy as np


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
    y[:n//2] *= -1
    return X, y
