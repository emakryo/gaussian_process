import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y
from scipy.stats import norm
from .base import BayesEstimator

eps = 1e-8

class GaussianProcessExpectationPropagation(BayesEstimator, ClassifierMixin):
    """Gaussian process classification with expectation propagation algorithm"""

    def __init__(self, cov):
        self.cov = cov
        self.param_bounds = {'cov__'+k: v for k, v in cov.param_bounds.items()}

    def fit(self, X, y):
        self.Xtr, self.ytr = check_X_y(X, y)
        self.n, self.dim = self.Xtr.shape
        K = self.cov(self.Xtr).K
        self._ep(K)

    def refit(self):
        self.fit(self.Xtr, self.ytr)

    def _ep(self, K, fix_index=None, nu_fixed=None, tau_fixed=None, shuffle=True):
        if fix_index is not None:
            fix_index = np.array(fix_index, dtype=int)
            nu_fixed = np.array(nu_fixed, dtype=float)
            tau_fixed = np.array(tau_fixed, dtype=float)
            assert len(fix_index) == len(nu_fixed) == len(tau_fixed)

        else:
            fix_index = np.array([], dtype=int)
            nu_fixed = np.array([])
            tau_fixed = np.array([])

        n = K.shape[0]
        K = K + eps * np.eye(n)
        is_fixed = np.zeros(n, dtype=bool)
        is_fixed[fix_index] = True
        # Natural parameters of site approximation:
        # $ \tilde{\tau} = 1 / \sigma^2 $
        # $ \tilde{\nu} = \tilde{\tau} \tilde{\mu} = \tilde{sigma}^{-2} \tilde{\mu} $
        # where site approximation is
        # $ \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^{-2} $
        # `tau_tilde = $ \tilde{\tau} $
        # `nu_tilde` = $ \tilde{\nu} $
        tau_tilde = np.zeros(n)
        nu_tilde = np.zeros(n)
        tau_tilde[fix_index] = tau_fixed
        nu_tilde[fix_index] = nu_fixed
        # Posterior covariance `Sigma` and mean `mu`
        # $ \Sigma = (K^{-1} + \text{diag}(\tilde{\sigma}^{-2}) \\
        #          = K - K (K + S^2)^{-1} K \\
        #          = K - K (K + S^{-2})^{-1} K \\
        #          = K - K S L L^\top S K $
        # \mu = \Sigma \tilde{\nu}
        # where
        # $ S = \text{diag}(\tilde{\sigma}) $
        # $ L = \text{cholesky}(I + S K S) $
        S = np.sqrt(tau_tilde).reshape(-1, 1)
        L = np.linalg.cholesky(np.eye(n) + S * K * S.T)
        V = np.linalg.solve(L, S * K)
        Sigma = K - V.T @ V
        mu = Sigma @ nu_tilde
        order = np.arange(n)
        for c in range(100):
            tau_tilde_old = np.copy(tau_tilde)
            nu_tilde_old = np.copy(nu_tilde)
            if shuffle:
                order = np.random.permutation(n)
            for i in order:
                if is_fixed[i]:
                    continue
                # cavity distribution
                tau_bar = 1.0 / Sigma[i, i] - tau_tilde[i]
                nu_bar = mu[i] / Sigma[i, i] - nu_tilde[i]
                dom = tau_bar ** 2 + tau_bar
                z = self.ytr[i] * nu_bar / np.sqrt(dom)
                ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))
                mu_hat = nu_bar / tau_bar + self.ytr[i] * ratio / np.sqrt(dom)
                sigma_hat = 1 / tau_bar - ratio / dom * (z + ratio)
                dtau_tilde = 1 / sigma_hat - tau_bar - tau_tilde[i]
                tau_tilde[i] += dtau_tilde
                nu_tilde[i] = mu_hat / sigma_hat - nu_bar
                Sigma -= (dtau_tilde / (1 + dtau_tilde * Sigma[i, i]) *
                          np.outer(Sigma[i], Sigma[i]))
                mu = Sigma @ nu_tilde

            S = np.sqrt(tau_tilde).reshape(-1, 1)
            L = np.linalg.cholesky(np.eye(n)+ S * K * S.T)
            V = np.linalg.solve(L, S * K)
            Sigma = K - V.T @ V
            mu = Sigma@nu_tilde
            tau_diff = np.sum((tau_tilde_old - tau_tilde) ** 2)
            nu_diff = np.sum((nu_tilde_old - nu_tilde) ** 2)
            if tau_diff < eps and nu_diff < eps:
                break

        self.nu_tilde = nu_tilde
        self.tau_tilde = tau_tilde
        self.mu = mu
        self.Sigma = Sigma
        self.K = K
        self.L = L
        self.S = S
        w = np.linalg.solve(L, S * K @ nu_tilde)
        self.z = S.reshape(-1) * np.linalg.solve(L.T, w)

    def posterior(self, X):
        Ks = self.cov(self.Xtr, X).K
        Kss = self.cov(X, diag=True).K
        return self._posterior(Ks, Kss)

    def _posterior(self, Ks, Kss):
        mean = Ks.T @ (self.nu_tilde - self.z)
        v = np.linalg.solve(self.L, self.S * Ks)
        var = Kss - np.sum(v**2, 0)
        return mean, var

    def decision_function(self, X):
        mean, var = self.posterior(X)
        return norm.cdf(mean/np.sqrt(1+var))

    def predict(self, X):
        pi = self.decision_function(X)
        return np.sign(pi-0.5)

    def log_marginal_likelihood(self, splitted_terms=False):
        # cavity distribution
        tau_bar = 1 / np.diag(self.Sigma) - self.tau_tilde
        nu_bar = self.mu / np.diag(self.Sigma) - self.nu_tilde
        t0 = -np.sum(np.log(np.diag(self.L)))
        t1 = 0.5 * np.sum(np.log(1 + self.tau_tilde / tau_bar))
        t2 = 0.5 * self.nu_tilde.dot(self.K @ self.nu_tilde)
        w = np.linalg.solve(self.L, self.S * self.K @ self.nu_tilde)
        t3 = -0.5 * w.dot(w)
        t4 = -0.5 * (self.nu_tilde ** 2).dot(1 / (tau_bar + self.tau_tilde))
        mu_bar = nu_bar / tau_bar
        t5 = 0.5 * nu_bar.dot((self.tau_tilde * mu_bar - 2 * self.nu_tilde) /
                              (tau_bar + self.tau_tilde))
        t6 = np.sum(norm.logcdf(self.ytr * mu_bar / np.sqrt(1 + 1 / tau_bar)))
        # for debugging
        t7 = 0.5 * np.sum(np.log(self.tau_tilde))
        t8 = -0.5 * self.nu_tilde.dot(self.nu_tilde / self.tau_tilde)

        if splitted_terms:
            return (t0, t1, t2, t3, t4, t5, t6)
        else:
            return sum((t0, t1, t2, t3, t4, t5, t6))

    def grad_log_marginal_likelihood(self, splitted_terms=False):
        kernel = self.cov(self.Xtr)
        K = kernel.K
        K_grads = {}
        for k, v in kernel.dtheta.items():
            K_grads['cov__'+k] = v
        return {k: v if splitted_terms else np.sum(v) for k, v in
                self._grad_log_marginal_likelihood(K, K_grads).items()}

    def _grad_log_marginal_likelihood(self, K, K_grads):
        U = np.linalg.solve(self.L, np.diag(self.S.reshape(-1)))
        K += eps * np.eye(self.n)
        b = (self.nu_tilde - U.T @ U @ K @ self.nu_tilde).reshape(-1, 1)
        R = b @ b.T - U.T @ U
        return {k: [0.5 * np.trace(b @ b. T @ v), 0.5 * np.trace(U.T @ U @ v)]
                for k, v in K_grads.items()}
