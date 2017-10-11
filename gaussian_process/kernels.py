import numpy as np
from collections import namedtuple
from sklearn.base import BaseEstimator


KernelValue = namedtuple('KernelValue', ['K', 'dtheta'])


class RBF(BaseEstimator):
    """
    Radial basis function:
        $ k(x_i, x_j) = \sigma \exp{ -0.5 \beta \|x_i - x_j\|^2 } $
    """

    param_bounds = {'beta':(1e-5, None), 'sigma':(1e-5, None)}

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
            dbeta = -0.5 * diff * K
            dsigma = K / self.sigma
        else:
            K = self.sigma * np.ones(min(n1, n2))
            dbeta = np.zeros(min(n1, n2))
            dsigma = np.ones(min(n1, n2))

        dtheta = {'sigma':dsigma, 'beta':dbeta}
        return KernelValue(K, dtheta)


class ARD(BaseEstimator):
    """
    Automatic relevence determination kernel:
        $ k(x_i, x_j) = \sigma \exp{ -0.5 \sum_k \beta_k |x_i^(k) - x_j^(k)|^2 } $
    """

    def __init__(self, sigma=1, beta=np.ones(3)):
        self.sigma = sigma
        self.beta = np.array(beta)
        self.dim = len(beta)
        self.param_bounds = {'sigma':(1e-5, None), 'beta':len(beta)*[(1e-5, None)]}

    def __call__(self, X1, X2=None, diag=False):
        self.beta = np.array(self.beta)
        assert self.beta.ndim == 1
        self.dim = len(self.beta)

        assert X1.ndim == 2
        if X2 is None:
            X2 = X1
        else:
            assert X2.ndim == 2

        n1, dim1 = X1.shape
        n2, dim2 = X2.shape
        assert dim1 == dim2 == self.dim

        if not diag:
            diff = (X1.reshape(-1, 1, self.dim) - X2.reshape(1, -1, self.dim))**2
            summed = np.sum(self.beta * diff, axis=2)
            K = self.sigma * np.exp(-0.5 * summed)
            dbeta = -0.5 * np.rollaxis(diff, 2) * K
            dsigma = K / self.sigma
        else:
            K = self.sigma * np.ones(min(n1, n2))
            dbeta = np.zeros(self.dim, min(n1, n2))
            dsigma = np.ones(min(n1, n2))

        dtheta = {'sigma':dsigma, 'beta':dbeta}
        return KernelValue(K, dtheta)
