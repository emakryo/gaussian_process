import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from .gfitc import GFITC

class PrivClassification():
    def __init__(self, m):
        self.m = m
        self.sigmaF = 0
        self.sigmaG = 0
        self.sigma0F = -1
        self.sigma0G = -1
        self.m0F = 0
        self.m0G = -1
        self.lF = 0
        self.lG = 0

    def fit(self, X, y, Xstar):
        self.X = X
        self.y = y
        self.Xstar = Xstar
        self.n, self.dim = self.X.shape
        self.dimStar = self.Xstar.shape[1]
        self.lF = self.lF * np.ones(self.dim)
        self.lG = self.lG * np.ones(self.dimStar)
        XbarF = self.X[:self.m]
        XbarG = self.Xstar[:self.m]
        self.internal(XbarF, XbarG)

    def internal(self, XbarF, XbarG):
        self.f1HatF1 = np.zeros(self.n)
        self.f1HatF2 = np.zeros(self.n)
        self.f1HatG1 = np.zeros(self.n)
        self.f1HatG2 = np.zeros(self.n)
        self.gfitcF = GFITC(self.X, XbarF, self.lF, self.sigmaF,
                            self.sigma0F, self.m0F)
        self.gfitcG = GFITC(self.Xstar, XbarG, self.lG, self.sigmaG,
                            self.sigma0G, self.m0G)

        damping = 0.5
        for i in range(100):
            print(i)
            f1, f2, g1, g2 = self.process_likelihood_factors(damping)
            assert all(np.isfinite(f1))
            assert all(np.isfinite(f2))
            assert all(np.isfinite(g1))
            assert all(np.isfinite(g2))
            diff = np.mean((self.f1HatF1-f1)**2)
            diff += np.mean((self.f1HatF2-f2)**2)
            diff += np.mean((self.f1HatG1-g1)**2)
            diff += np.mean((self.f1HatG2-g2)**2)
            if diff < 1e-5:
                break
            self.f1HatF1 = f1
            self.f1HatF2 = f2
            self.f1HatG1 = g1
            self.f1HatG2 = g2
            damping *= 0.99


    def process_likelihood_factors(self, damping):
        meanMarginalF, varMarginalF = self.gfitcF.titledDistribution(self.f1HatF2,
                                                                     self.f1HatF1)
        meanMarginalG, varMarginalG = self.gfitcG.titledDistribution(self.f1HatG2,
                                                                     self.f1HatG1)

        assert all(varMarginalF>0)
        assert all(varMarginalG>0)

        vOldF = 1/(1/varMarginalF - self.f1HatF2)
        mOldF = vOldF * (meanMarginalF / varMarginalF - self.f1HatF1)
        vOldG = 1/(1/varMarginalG - self.f1HatG2)
        mOldG = vOldG * (meanMarginalG / varMarginalG - self.f1HatG1)

        assert all(vOldF>0)
        assert all(vOldG>0)

        Z = self.Z(mOldF, vOldF, mOldG, vOldG)
        logZ = np.log(Z)

        alphaF = self.dZdmOldF(mOldF, vOldF, mOldG, vOldG) / Z
        betaF = self.d2ZdmOldF2(mOldF, vOldF, mOldG, vOldG) / Z - alphaF**2

        f1HatF2 = -betaF / (1+betaF*vOldF)
        f1HatF1 = - (alphaF-mOldF*betaF)/(1+betaF*vOldF)

        alphaG = self.dZdmOldG(mOldF, vOldF, mOldG, vOldG) / Z
        betaG = self.d2ZdmOldG2(mOldF, vOldF, mOldG, vOldG) / Z - alphaG**2

        f1HatG2 = -betaG / (1+betaG*vOldG)
        f1HatG1 = - (alphaG-mOldG*betaG)/(1+betaG*vOldG)

        f1HatF1 = damping * self.f1HatF1 + (1-damping) * f1HatF1
        f1HatF2 = damping * self.f1HatF2 + (1-damping) * f1HatF2
        f1HatG1 = damping * self.f1HatG1 + (1-damping) * f1HatG1
        f1HatG2 = damping * self.f1HatG2 + (1-damping) * f1HatG2

        return f1HatF1, f1HatF2, f1HatG1, f1HatG2

    def Z(self, mOldF, vOldF, mOldG, vOldG):
        z = np.empty(self.n)
        for i in range(self.n):
            def fun(g):
                y = self.y[i]
                mf, sf = mOldF[i], np.sqrt(np.exp(g)+vOldF[i])
                mg, sg = mOldG[i], np.sqrt(vOldG[i])
                return norm.pdf(g, mg, sg) * norm.cdf(y * mf / sf)

            z[i] = quad(fun, -np.inf, np.inf)[0]

        return z

    def dZdmOldF(self, mOldF, vOldF, mOldG, vOldG):
        z = np.empty(self.n)
        for i in range(self.n):
            def fun(g):
                y = self.y[i]
                mf, sf = mOldF[i], np.sqrt(np.exp(g)+vOldF[i])
                mg, sg = mOldG[i], np.sqrt(vOldG[i])
                return norm.pdf(g, mg, sg) * norm.pdf(y * mf / sf) * y / sf

            z[i] = quad(fun, -np.inf, np.inf)[0]

        return z

    def d2ZdmOldF2(self, mOldF, vOldF, mOldG, vOldG):
        z = np.empty(self.n)
        for i in range(self.n):
            def fun(g):
                y = self.y[i]
                mf, sf = mOldF[i], np.sqrt(np.exp(g)+vOldF[i])
                mg, sg = mOldG[i], np.sqrt(vOldG[i])
                return norm.pdf(g, mg, sg)*norm.pdf(y*mf/sf)*sf**-3*(-mf)*y

            z[i] = quad(fun, -np.inf, np.inf)[0]

        return z

    def dZdmOldG(self, mOldF, vOldF, mOldG, vOldG):
        z = np.empty(self.n)
        for i in range(self.n):
            def fun(g):
                y = self.y[i]
                mf, sf = mOldF[i], np.sqrt(np.exp(g)+vOldF[i])
                mg, sg = mOldG[i], np.sqrt(vOldG[i])
                return norm.pdf(g, mg, sg)*(g-mg)*sg**-2*norm.cdf(y*mf/sf)

            z[i] = quad(fun, -np.inf, np.inf)[0]

        return z

    def d2ZdmOldG2(self, mOldF, vOldF, mOldG, vOldG):
        z = np.empty(self.n)
        for i in range(self.n):
            def fun(g):
                y = self.y[i]
                mf, sf = mOldF[i], np.sqrt(np.exp(g)+vOldF[i])
                mg, sg = mOldG[i], np.sqrt(vOldG[i])
                return (norm.pdf(g, mg, sg)*((g-mg)**2*sg**-4-sg**-2)*norm.cdf(y*mf/sf))

            z[i] = quad(fun, -np.inf, np.inf)[0]

        return z

    def predict(self, Xtest):
        mf, vf = self.gfitcf.predict(Xtest, self.f1HatF2, self.f1HatF1)
        noise = self.predictNoise(Xtest)
        return norm.cdf(mf/np.sqrt(vf))
