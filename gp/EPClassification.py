import numpy as np
from scipy.stats import norm
from gp import kernel


class EPClassification():
    def __init__(self, kernel=kernel.RBF, k_params={'sgm': 1, 'beta': 10}):
        self.kernel = kernel
        self.k_params = k_params

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
            print(c)
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
            L = np.linalg.cholesky(np.eye(self.n) + Ssq * K * Ssq.T)
            V = np.linalg.solve(L, Ssq * K)
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
