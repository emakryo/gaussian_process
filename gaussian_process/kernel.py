import numpy as np
from collections import namedtuple
from sklearn.base import BaseEstimator

KernelValue = namedtuple('KernelValue', ['K', 'dtheta'])

class RBF(BaseEstimator):
    """
    Radial basis function:
        $ k(x_i, x_j) = \sigma \exp{ \beta \|x_i - x_j\|^2 } $
    """

    param_bounds = [(1e-5, None), (1e-5, None)]

    def __init__(self, sigma=1, beta=1):
        self.sigma = sigma
        self.beta = beta

    def __call__(self, X1, X2=None, diag=False):
        assert X1.ndim == 2

        if X2 is None:
            X2 = X1
        else:
            assert X2.ndim == 2

        n1, dim1 = X1.shape
        n2, dim2 = X2.shape
        assert dim1 == dim2

        if not diag:
            X1sq = np.sum(X1 ** 2, 1).reshape(-1, 1)
            X2sq = np.sum(X2 ** 2, 1).reshape(-1, 1)
            diff = X1sq + X2sq.T - 2 * X1 @ X2.T
            K = self.sigma * np.exp(-0.5 * self.beta * diff)
            dbeta = - 0.5 * K * diff
            dsigma = K/self.sigma
        else:
            K = self.sigma * np.ones(min(n1, n2))
            dbeta = np.zeros(min(n1, n2))
            dsigma = np.ones(min(n1, n2))

        dtheta = np.stack([dbeta, dsigma], axis=2)
        return KernelValue(K, dtheta)


class ARD():
    def __init__(self, X1, X2=None, ls=1., sigma=1., diag=False):
        assert X1.ndim == 2
        self.X1 = X1
        if X2 is None:
            self.X2 = X1
        else:
            self.X2 = X2

        assert self.X1.shape[1] == self.X2.shape[1]
        self.n1, self.dim = self.X1.shape
        self.n2 = self.X2.shape[0]

        ls = ls * self.ones(self.dim)
        self.params = {'ls': ls, 'sgm': sigma}

        if not diag:
            X1sq = np.sum((self.X1*ls.reshape(1, -1))**2, 1).reshape(-1, 1)
            X2sq = np.sum((self.X2*ls.reshape(1, -1))**2, 1).reshape(-1, 1)
            self._diff = X1sq+X2sq.T-2*self.X1@self.X2.T

    @property
    def sigma(self):
        return self.params['sgm']

    @sigma.setter
    def sigma(self, x):
        self.params['sgm'] = x

    @property
    def ls(self):
        return self.parmas['ls']

    @ls.setter
    def ls(self, x):
        self.params['ls'] = x*np.ones(self.dim)

    def __call__(self):
        if self.diag:
            return self.sigma*np.ones(min(self.n1, self.n2))
        else:
            return self.sigma * np.exp(-0.5*self._diff)
