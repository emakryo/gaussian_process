import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py


def data(n=100):
    X = 4 * np.pinp.random.rand(n, 1)
    y = 2 * np.sin(x) + np.cos(np.sqrt(2) * x) + np.random.randn(n, 1) * 0.2
    return X, y


def main():
    X, y = data()
    py.iplot([{'x': X[:, 0], 'y':y[:, 0], 'mode':'markers'}])
