import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from . import kernel


class EPClassification():
    def __init__(self, kernel=kernel.RBF, sigma=1, beta=1):
        self.kernel = kernel
        self.k_params = {'sigma': sigma, 'beta': beta}

    def fit(self, X, y):
        self.X = X
        self.n, self.dim = self.X.shape
        assert self.n == len(y)
        self.y = y

        K = self.kernel(X, **self.k_params)()
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
                z = y[i] * nuBar / np.sqrt(dom)
                ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))
                muHat = nuBar / tauBar + y[i]*ratio/np.sqrt(dom)
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

    def posterior(self, Xs):
        Ks = self.kernel(self.X, Xs, **self.k_params)()
        mean = Ks.T @ (self.nuTilde - self.z)
        v = np.linalg.solve(self.L, self.Ssq*Ks)
        Kss = np.diag(self.kernel(Xs, Xs, **self.k_params)())
        var = Kss - np.sum(v**2, 0)
        return mean, var

    def decision_function(self, Xs):
        mean, var = self.posterior(Xs)
        return norm.cdf(mean/np.sqrt(1+var))

    def predict(self, Xs):
        pi = self.decision_function(Xs)
        return pi > 0.5

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
        t6 = np.sum(norm.logcdf(self.y*muBar/np.sqrt(1+1/tauBar)))

        # for debugging
        t7 = 0.5*np.sum(np.log(self.tauTilde))
        t8 = -0.5*self.nuTilde.dot(self.nuTilde/self.tauTilde)
        return (t0,t1,t2,t3,t4,t5,t6,t7,t8,-t7,-t8)

    def dlog_mldtheta(self):
        nuTilde = self.nuTilde.reshape(-1, 1)
        U = np.linalg.solve(self.L, np.diag(self.Ssq.reshape(-1)))
        b = nuTilde - U.T@U@self.K@self.nuTilde
        R = b@b.T - U.T@U
        dKdtheta = self.kernel(self.X, **self.k_params).dtheta()
        return 0.5*np.trace(np.einsum('ij,jkl->ikl', R, dKdtheta))

    def empiricalBayes(self):
        def obj(x):
            self.k_params = self.kernel.param_dict(x)
            self.fit(self.X, self.y)
            return -self.log_ml()

        res = minimize(obj, [1, 1], bounds=[(1e-5, None), (1e-5, None)])
        self.k_params = self.kernel.param_dict(res.x)

