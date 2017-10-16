import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y
from scipy.stats import norm
from scipy.optimize import minimize
from .base import BayesEstimator
from .expectation_propagation import GaussianProcessExpectationPropagation


class GaussianProcessDistillation(GaussianProcessExpectationPropagation):
    """Gaussian process distillation with expectation propagation algorithm"""

    def __init__(self, cov, rho, sigma):
        self.cov = cov
        self.rho = rho
        self.sigma = sigma

    def fit(self, X, y, s):
        self.Xtr, self.ytr = check_X_y(X, y)
        _, self.str = check_X_y(X, s, y_numeric=True)
        self.n, self.dim = self.Xtr.shape
        # Kernel matrix between data points
        K_ = self.cov(self.Xtr).K
        # Kernel matrix of whole tasks (regression and classification)
        K = np.block([[K_, self.rho * K_],
                      [self.rho * K_, K_]])
        self._ep(K, fix_index=np.arange(self.n, 2 * self.n),
                 nu_fixed=self.str / self.sigma,
                 tau_fixed=np.ones(self.n) / self.sigma)

    def posterior(self, X):
        Ks_ = self.cov(self.Xtr, X).K
        Ks = np.vstack([Ks_, self.rho * Ks_])
        Kss = self.cov(X, diag=True).K
        return self._posterior(Ks, Kss)
