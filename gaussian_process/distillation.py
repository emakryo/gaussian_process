import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y
from scipy.stats import norm
from scipy.optimize import minimize
from .base import BayesEstimator


class GaussianProcessDistillation(BayesEstimator, ClassifierMixin):
    """Gaussian process distillation with expectation propagation algorithm"""

    def __init__(self, cov, rho, sigma):
        self.cov = cov
        self.rho = rho
        self.sigma = sigma

    def fit(self, X, y, s):
        self.Xtr, self.ytr = check_X_y(X, y)
        _, self.str = check_X_y(X, s, y_numeric=True)
        self.n, self.dim = self.Xtr.shape
        self._fit()

    def _fit(self):
        # Kernel matrix between data points
        K_ = self.cov(self.Xtr).K
        # Kernel matrix of whole tasks (regression and classification)
        K = np.block([[K_, self.rho * K_],
                      [self.rho * K_, K_]])
        # Natural parameters of site approximation:
        # $ \tilde{\tau} = 1 / \sigma^2 $
        # $ \tilde{\nu} = \tilde{\tau} \tilde{\mu} = \tilde{sigma}^{-2} \tilde{\mu} $
        # where site approximation is
        # $ \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^{-2} $
        # `tau_tilde = $ \tilde{\tau} $
        # `nu_tilde` = $ \tilde{\nu} $
        tau_tilde = np.concatenate([np.zeros(self.n), np.ones(self.n) / self.sigma])
        nu_tilde = np.concatenate([np.zeros(self.n), self.str / self.sigma])
        # Posterior covariance `Sigma` and mean `mu`
        # where
        # $ S = \text{diag}(\tilde{\sigma}) $
        # $ L = \text{cholesky}(I + S K S) $
        S = np.sqrt(tau_tilde).reshape(-1, 1)
        L = np.linalg.cholesky(np.eye(2 * self.n) + S * K * S.T)
        V = np.linalg.solve(L, S * K)
        Sigma = K - V.T @ V
        mu = Sigma @ nu_tilde
        eps = 1e-5
        for c in range(100):
            tau_tilde_old = np.copy(tau_tilde)
            nu_tilde_old = np.copy(nu_tilde)
            for i in range(self.n):
                # cavity distribution
                tau_bar = 1.0 / Sigma[i, i] - tau_tilde[i]
                nu_bar = mu[i] / Sigma[i, i] - nu_tilde[i]
                dom = tau_bar**2+tau_bar
                z = self.ytr[i] * nu_bar / np.sqrt(dom)
                ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))
                mu_hat = nu_bar / tau_bar + self.ytr[i] * ratio / np.sqrt(dom)
                sigma_hat = 1 / tau_bar - ratio / dom * (z + ratio)
                dtau_tilde = 1 / sigma_hat - tau_bar - tau_tilde[i]
                #tau_tildeNext = tau_tilde[i] + dTauTilde
                #nuTildeNext = muHat / sigmaHat - nuBar
                tau_tilde[i] += dtau_tilde
                nu_tilde[i] = mu_hat / sigma_hat - nu_bar
                Sigma -= (dtau_tilde / (1 + dtau_tilde * Sigma[i, i]) *
                          np.outer(Sigma[i], Sigma[i]))
                mu = Sigma @ nu_tilde

            S = np.sqrt(tau_tilde).reshape(-1, 1)
            L = np.linalg.cholesky(np.eye(2 * self.n)+ S * K * S.T)
            V = np.linalg.solve(L, S * K)
            Sigma = K - V.T @ V
            mu = Sigma@nu_tilde
            diff = (np.sum((tau_tilde_old - tau_tilde) ** 2) +
                    np.sum((nu_tilde_old - nu_tilde) ** 2))
            if diff < eps:
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
        Ks_ = self.cov(self.Xtr, X).K
        Ks = np.block([[Ks_], [self.rho * Ks_]])
        Kss = self.cov(X, diag=True).K
        return self._posterior(Ks, Kss)

    def _posterior(self, Ks, Kss):
        mean = Ks.T @ (self.nu_tilde - self.z)
        v = np.linalg.solve(self.L, self.S * Ks)
        var = Kss - np.sum(v ** 2, 0)
        return mean, var

    def decision_function(self, X):
        mean, var = self.posterior(X)
        return norm.cdf(mean/np.sqrt(1+var))

    def predict(self, X):
        pi = self.decision_function(X)
        return np.sign(pi-0.5)

    def log_ml(self):
        raise NotImplementedError()
        return sum(self.log_ml_terms())

    def log_ml_terms(self):
        raise NotImplementedError()
        tauBar = 1.0/np.diag(self.Sigma) - self.tauTilde
        nuBar = self.mu/np.diag(self.Sigma) - self.nuTilde
        t0 = -np.sum(np.log(np.diag(self.L)))
        t1 = 0.5*np.sum(np.log(1+self.tauTilde/tauBar))
        t2 = 0.5*self.nuTilde.dot(self.K@self.nuTilde)
        w = np.linalg.solve(self.L, self.Ssq*self.K@self.nuTilde)
        t3 = -0.5*w.dot(w)
        t4 = -0.5*(self.nuTilde**2).dot(1/(tauBar+self.tauTilde))
        muBar = nuBar/tauBar
        t5 = 0.5*nuBar.dot((self.tauTilde*muBar-2*self.nuTilde)/(tauBar+self.tauTilde))
        t6 = np.sum(norm.logcdf(self.ytr*muBar/np.sqrt(1+1/tauBar)))
        # for debugging
        t7 = 0.5*np.sum(np.log(self.tauTilde))
        t8 = -0.5*self.nuTilde.dot(self.nuTilde/self.tauTilde)
        return (t0,t1,t2,t3,t4,t5,t6,t7,t8,-t7,-t8)

    def dlog_mldtheta(self):
        raise NotImplementedError()
        nuTilde = self.nuTilde.reshape(-1, 1)
        U = np.linalg.solve(self.L, np.diag(self.Ssq.reshape(-1)))
        b = nuTilde - U.T@U@self.K@self.nuTilde
        R = b@b.T - U.T@U
        dKdtheta = self.kern(self.X, **self.k_params).dtheta()
        return 0.5*np.trace(np.einsum('ij,jkl->ikl', R, dKdtheta))

    def empirical_bayes(self, opt_param_names=None, verbose=False):
        raise NotImplementedError()
        if opt_param_names is None:
            opt_param_names = [k for k, v in self.get_params().items()
                               if np.isscalar(v) or isinstance(v, np.ndarray)]

        def obj(x):
            self.set_opt_params(x, opt_param_names)
            self._fit()
            return -self.log_ml()

        before_obj = -self.log_ml()
        before_params = self.get_opt_params(opt_param_names)
        init = before_params
        bounds = []
        for k, v in self.cov.param_bounds.items():
            if isinstance(v[0], tuple):
                bounds.extend(v)
            else:
                bounds.append(v)

        res = minimize(obj, init, bounds=bounds)

        if verbose:
            print(res.message)

        if res.fun < before_obj:
            self.set_opt_params(res.x, opt_param_names)
        else:
            self.set_opt_params(before_params, opt_param_names)

        self._fit()
