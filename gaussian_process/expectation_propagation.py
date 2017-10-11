import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from scipy.stats import norm
from scipy.optimize import minimize


class GaussianProcessExpectationPropagation(BaseEstimator, ClassifierMixin):
    """Gaussian process classification with expectation propagation algorithm"""

    def __init__(self, cov):
        self.cov = cov

    def fit(self, X, y):
        self.Xtr, self.ytr = check_X_y(X, y)
        self.n, self.dim = self.Xtr.shape

        self._fit()

    def _fit(self):
        K = self.cov(self.Xtr).K
        nuTilde = np.zeros(self.n)
        tauTilde = np.zeros(self.n)
        Sigma = np.copy(K)
        mu = np.zeros(self.n)
        eps = 1e-5
        for c in range(100):
            tauTildeOld = np.copy(tauTilde)
            nuTildeOld = np.copy(nuTilde)
            for i in range(self.n):
                tauBar = 1.0/Sigma[i, i] - tauTilde[i]
                nuBar = mu[i]/Sigma[i, i] - nuTilde[i]
                dom = tauBar**2+tauBar
                z = self.ytr[i] * nuBar / np.sqrt(dom)
                ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))
                muHat = nuBar / tauBar + self.ytr[i]*ratio/np.sqrt(dom)
                sigmaHat = 1/tauBar - ratio / dom * (z+ratio)
                dTauTilde = 1/sigmaHat - tauBar - tauTilde[i]
                tauTildeNext = tauTilde[i] + dTauTilde
                nuTildeNext = muHat / sigmaHat - nuBar
                tauTilde[i] = tauTildeNext
                nuTilde[i] = nuTildeNext
                Sigma -= (dTauTilde/(1+dTauTilde*Sigma[i, i]) *
                          np.outer(Sigma[i], Sigma[i]))
                mu = Sigma@nuTilde

            Ssq = np.sqrt(tauTilde).reshape(-1, 1)
            L = np.linalg.cholesky(np.eye(self.n)+Ssq*K*Ssq.T)
            V = np.linalg.solve(L, Ssq*K)
            Sigma = K - V.T@V
            mu = Sigma@nuTilde

            diff = (np.sum((tauTildeOld-tauTilde)**2) +
                    np.sum((nuTildeOld-nuTilde)**2))
            if diff < eps:
                break

        self.nuTilde = nuTilde
        self.tauTilde = tauTilde
        self.mu = mu
        self.Sigma = Sigma
        self.K = K
        self.L = L
        self.Ssq = Ssq
        w = np.linalg.solve(L, Ssq*K@nuTilde)
        self.z = Ssq.reshape(-1)*np.linalg.solve(L.T, w)

    def posterior(self, X):
        Ks = self.cov(self.Xtr, X).K
        Kss = self.cov(X, diag=True).K
        return self._posterior(Ks, Kss)

    def _posterior(self, Ks, Kss):
        mean = Ks.T @ (self.nuTilde - self.z)
        v = np.linalg.solve(self.L, self.Ssq*Ks)
        var = Kss - np.sum(v**2, 0)
        return mean, var

    def decision_function(self, X):
        mean, var = self.posterior(X)
        return norm.cdf(mean/np.sqrt(1+var))

    def predict(self, X):
        pi = self.decision_function(X)
        return np.sign(pi-0.5)

    def log_ml(self):
        return sum(self.log_ml_terms())

    def log_ml_terms(self):
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
        nuTilde = self.nuTilde.reshape(-1, 1)
        U = np.linalg.solve(self.L, np.diag(self.Ssq.reshape(-1)))
        b = nuTilde - U.T@U@self.K@self.nuTilde
        R = b@b.T - U.T@U
        dKdtheta = self.kern(self.X, **self.k_params).dtheta()
        return 0.5*np.trace(np.einsum('ij,jkl->ikl', R, dKdtheta))

    def empirical_bayes(self, opt_param_names=None, verbose=False):
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

    def get_opt_params(self, param_names):
        params = self.get_params()
        opt_params = []
        for k in param_names:
            if np.isscalar(params[k]):
                opt_params.append(params[k])
            else:
                opt_params.extend(params[k].flatten())

        return np.array(opt_params)

    def set_opt_params(self, opt_params, param_names):
        params = self.get_params()
        index = 0
        set_params = {}
        for k in param_names:
            if np.isscalar(params[k]):
                set_params[k] = opt_params[index]
                index += 1
            else:
                shape = params[k].shape
                val = opt_params[index:index+np.prod(shape)]
                set_params[k] = np.array(val).reshape(*shape)

        self.set_params(**set_params)
