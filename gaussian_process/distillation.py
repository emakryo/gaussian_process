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
        self.param_bounds = {'rho': (0, 1),
                             'sigma': (1e-5, None)}
        for k, v in self.cov.param_bounds.items():
            self.param_bounds['cov__'+k] = v

    def fit(self, X, y, s):
        self.Xtr, self.ytr = check_X_y(X, y)
        _, self.str = check_X_y(X, s, y_numeric=True)
        self.n, self.dim = self.Xtr.shape
        # Kernel matrix between data points
        K_ = self.cov(self.Xtr).K
        # Task kernel
        Kt = np.array([[1, self.rho],
                       [self.rho, 1]])
        # Kernel matrix of whole tasks (regression and classification)
        K = np.kron(Kt, K_)
        self._ep(K, fix_index=np.arange(self.n, 2 * self.n),
                 nu_fixed=self.str / self.sigma,
                 tau_fixed=np.ones(self.n) / self.sigma)

    def refit(self):
        self.fit(self.Xtr, self.ytr, self.str)

    def posterior(self, X):
        Ks_ = self.cov(self.Xtr, X).K
        Ks = np.vstack([Ks_, self.rho * Ks_])
        Kss = self.cov(X, diag=True).K
        return self._posterior(Ks, Kss)

    def log_marginal_likelihood(self, eps=1e-7, splitted_terms=False):
        tau_bar = 1 / np.diag(self.Sigma) - self.tau_tilde
        nu_bar = self.mu / np.diag(self.Sigma) - self.nu_tilde
        t0 = -np.sum(np.log(np.trace(self.L)))
        t1 = np.sum(np.log(self.S))
        w = np.linalg.solve(self.L @ np.diag(self.S.flatten()), self.nu_tilde)
        t2 = -0.5 * w.dot(w)
        t3 = np.sum(norm.logcdf(self.ytr * nu_bar[self.n:] /
                    np.sqrt(tau_bar[self.n:] * (1 + tau_bar[self.n:]))))
        assert np.all(tau_bar[self.n:] > eps)
        assert np.all(self.tau_tilde[self.n:] > eps)
        t4 = 0.5 * np.sum(np.log((1 / tau_bar + 1 / self.tau_tilde)[self.n:]))
        t5 = 0.5 * np.sum(((nu_bar / tau_bar + self.nu_tilde / self.tau_tilde)**2 *
                           (1 / tau_bar + 1 / self.tau_tilde))[self.n:])
        if splitted_terms:
            return t0, t1, t2, t3, t4, t5
        else:
            return sum((t0, t1, t2, t3, t4, t5))

    def grad_log_marginal_likelihood(self, splitted_terms=False):
        kernel = self.cov(self.Xtr)
        K_ = kernel.K
        Kt = np.array([[1, self.rho],
                       [self.rho, 1]])
        K = np.kron(Kt, K_)
        K_grads = {'rho': np.kron([[0., 1.],
                                   [1., 0.]], K_),
                   'sigma': np.block([[np.zeros((self.n, self.n)),
                                       np.zeros((self.n, self.n))],
                                      [np.zeros((self.n, self.n)),
                                       np.eye(self.n)]])}
        for k, v in kernel.dtheta.items():
            K_grads['cov__'+k] = np.kron(Kt, v)

        return {k: v if splitted_terms else sum(v) for k, v
                in self._grad_log_marginal_likelihood(K, K_grads).items()}
