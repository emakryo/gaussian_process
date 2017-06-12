import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from gp import kernel

def dzdg(g, mf, vf, mg, vg, y):
    sf = np.sqrt(vf+np.exp(g))
    sg = np.sqrt(vg)
    return norm.cdf(y*mf/sf)*norm.pdf(g, mg, sg)

def d2zdgdmf(g, mf, vf, mg, vg, y):
    sf = np.sqrt(vf+np.exp(g))
    sg = np.sqrt(vg)
    return norm.pdf(y*mf/sf)*y/sf*norm.pdf(g, mg, sg)

def d3zdzdmf2(g, mf, vf, mg, vg, y):
    sf = np.sqrt(vf+np.exp(g))
    sg = np.sqrt(vg)
    return -0.5*norm.pdf(y*mf/sf)*sg**-3*norm.pdf(g, mg, sg)

def d2zdgdmg(g, mf, vf, mg, vg, y):
    sf = np.sqrt(vf+np.exp(g))
    sg = np.sqrt(vg)
    return norm.cdf(y*mf/sf)*norm.pdf(g, mg, sg)*(-(g-mg)/sg**2)

def d3zdgdmg2(g, mf, vf, mg, vg, y):
    sf = np.sqrt(vf+np.exp(g))
    sg = np.sqrt(vg)
    return norm.cdf(y*mf/sf)*norm.pdf(g, mg, sg)*(sg**-2-(g-mg)**2*sg**-4)

class EPClassification():
    def __init__(self, kernelF=kernel.RBF, k_paramsF={'sigma': 1, 'beta': 10},
                 kernelG=kernel.RBF, k_paramsG={'sigma':1, 'beta':10}):
        self.kernel = kernel
        self.k_params = k_params

    def fit(self, X, y, Xstar):
        self.X = X
        self.Xstar = Xstar
        self.n, self.dim = self.X.shape
        self.dimStar = self.Xstar.shape[1]
        assert self.n == len(y)
        self.y = y

        KF = self.kernel(X, **self.k_paramsF)
        KG = self.kernel(Xstar, **self.k_paramsG)
        nuTildeF = np.zeros(self.n)
        tauTildeF = np.zeros(self.n)
        nuTildeG = np.zeros(self.n)
        tauTildeG = np.zeros(self.n)
        SigmaF = np.copy(KF())
        muF = np.zeros(self.n)
        SgiamG = np.copy(KG())
        muG = np.zeros(self.n)
        eps = 1e-5
        for c in range(100):
            print(c)
            tauTildeFOld = np.copy(tauTildeF)
            nuTildeFOld = np.copy(nuTildeF)
            tauTildeGOld = np.copy(tauTildeG)
            nuTildeGOld = np.copy(nuTildeG)
            for i in range(self.n):
                tauBarF = 1.0/SigmaF[i, i] - tauTildeF[i]
                nuBarF = muF[i]/SigmaF[i, i] - nuTildeF[i]
                tauBarG = 1.0/SigmaG[i, i] - tauTildeG[i]
                nuBarG = muG[i]/SigmaG[i, i] - nuTildeG[i]

                mf = nuBarF/tauBarF
                vf = 1/tauBarF
                mg = nuBarG/tauBarG
                vg = 1/tauBarG
                z = quad(dzdg, -np.inf, np.inf, (mf, vf, mg, vg, y))
                dzdmf = quad(d2zdgdmf, -np.inf, np.inf, (mf, vf, mg, vg, y))
                d2zdmf2 = quad(d3zdgdmf2, -np.inf, np.inf, (mf, vf, mg, vg, y))
                dzdmg = quad(d2zdgdmg, -np.inf, np.inf, (mf, vf, mg, vg, y))
                d2zdmg2 = quad(d3zdgdmg2, -np.inf, np.inf, (mf, vf, mg, vg, y))
                af = dzdmf/z
                bf = d2zdmf2/z-af/z
                ag = dzdmg/z
                bg = d2zdmg2/z-ag/z
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
            L = np.linalg.cholesky(np.eye(self.n) + Ssq * K() * Ssq.T)
            V = np.linalg.solve(L, Ssq * K())
            Sigma = K() - V.T@V
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
        w = np.linalg.solve(L, Ssq*K()@nuTilde)
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
