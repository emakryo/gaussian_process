import numpy as np
import matplotlib.pyplot as plt

try:
    from gp import Regression
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import Regression


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


def main():
    X, y = data1(100, 0.2)
    model = Regression()
    model.fit(X, y)
    for i in range(10):
        model.optimize(True)

    model.plot((-1, 11), "output.png")
    plt.show()
    print(model.grad_log_marginal_likelihood())
    print(model.log_marginal_likelihood())
    print(model.param_array)
    print(model.k_params)

#    x = np.linspace(0.002, 0.01, 10)
#    y = np.zeros(10)
#    yg = np.zeros(10)
#    for i,xn in enumerate(x):
#        param_array = model.param_array
#        param_array[0] = xn
#        model.param_array = param_array
#        y[i]  = model.log_marginal_likelihood()
#        yg[i] = model.grad_log_marginal_likelihood()[0]
#
#    plt.clf()
#    plt.plot(x,y)
#    plt.savefig('obj.png')
#    plt.clf()
#    plt.plot(x,yg)
#    plt.savefig('grad.png')


if __name__ == "__main__":
    main()
