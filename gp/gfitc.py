import numpy as np
from scipy.linalg import solve_triangular

def cholInv(M):
    """return lower triangular L s.t. M^(-1) = L @ L.T"""
    return solve_triangular(np.linalg.cholesky(M), np.eye(M.shape[0]), lower=True)

class GFITC():
    def __init__(self, X, Xbar, l, sigma=0, sigma0=-1, m0=0):
        assert X.shape[1] == Xbar.shape[1] == len(l)
        self.X = X
        self.Xbar = Xbar
        self.n, self.d = X.shape
        self.m, self.d = Xbar.shape
        self.sigma = np.exp(sigma) + 1e-3
        self.sigma0 = np.exp(sigma0)
        self.l = np.exp(l).reshape(1,-1)

        self.Xbarl = Xbar * self.l
        self.Xl = X * self.l

        self.Xbar2 = np.sum(self.Xbarl**2, 1).reshape(-1,1)
        self.dmm = self.Xbar2 + self.Xbar2.T - 2*self.Xbarl@self.Xbarl.T
        self.Kmm = self.sigma*np.exp(-0.5*self.dmm) + (self.sigma0+1e-6)*np.eye(self.m)

        self.X2 = np.sum(self.Xl**2, 1).reshape(-1,1)
        self.dnm = self.X2 + self.Xbar2.T - 2*self.Xl@self.Xbarl.T
        self.Knm = self.sigma*np.exp(-0.5*self.dnm)

        self.P = self.Knm
        self.R = cholInv(self.Kmm)
        self.PRt = self.P @ self.R.T

        self.diagKnn = (self.sigma+self.sigma0+1e-6)*np.ones(self.n)
        self.D = self.diagKnn - np.sum(self.PRt**2, 1)

        L0 = cholInv(np.eye(self.m)+(self.PRt.T/self.D)@self.PRt)
        V = L0@self.PRt.T*(1/self.D)
        self.AinvM0 = m0/self.D - (m0*V.T@V.sum(1)).reshape(-1)

        self.m0 = m0
        self.RtRPAInvM0 = self.R.T@self.PRt.T@self.AinvM0

    def predict(self, Xtest, tauTilde, muTilde):
        r = Xtest.shape[0]
        Xtest = Xtest * self.l
        Xtest2 = np.sum(Xtest**2, 1).reshape(-1,1)
        dtm = Xtest2 + self.Xbar2.T - 2*Xtest@self.Xbar.T
        pStar = self.sigma*np.exp(-0.5*dtm)
        dStar = (self.sigma+self.sigma0+1e-6)*np.ones(r) - np.sum((pStar@self.R.T)**2, 1)
        Dnew = self.D / (1+self.D*tauTilde)
        Pnew = self.P / (1+self.D*tauTilde).reshape(-1, 1)
        chol = np.linalg.cholesky(np.eye(self.m)+(self.PRt.T*tauTilde/(1+self.D*tauTilde))@self.PRt)
        Rnew = solve_triangular(chol, self.R, lower=True)
        gammaNew = Rnew.T@(Rnew@Pnew.T@(muTilde+self.AinvM0))
        mPrediction = pStar@gammaNew+self.m0-pStar@self.RtRPAInvM0
        vPrediction = dStar+np.sum((pStar@Rnew.T)**2, 1)
        return mPrediction, vPrediction

    def evidence(self, tauTilde, muTilde):
        Dnew = self.D / (1+self.D*tauTilde)
        Pnew = self.P / (1+self.D*tauTilde)
        chol = np.linalg.cholesky(np.eye(self.m)+(self.PRt.T/(1+self.D*tauTilde))@self.PRt)
        Rnew = solve_triangular(chol, self.R, lower=True)
        upsilon = self.AInvM0 + muTilde
        aNew = Dnew * upsilon
        gammaNew = Rnew.T@(Rnew@Pnew.T@upsilon)
        mNew = aNew + (Pnew@gammaNew).reshape(-1)
        t1 = -0.5*self.n*np.log(2*np.pi)+np.sum(np.log(np.diag(Rnew)))
        t2 = -np.sum(np.log(np.diag(self.R)))-0.5*np.sum(np.log(1+tauTilde*self.D))
        t3 = 0.5*np.sum((muTilde+self.AInvM0)*mNew)-0.5*np.sum(muTilde**2/tauTilde)
        t4 = -0.5*np.sum(self.m0*self.AInvM0)
        return t1+t2+t3+t4

    def titledDistribution(self, tauTilde, muTilde):
        d = (1+self.D*tauTilde)
        DNew = self.D / d
        PNew = self.P / d
        chol = np.linalg.cholesky(np.eye(self.m)+(self.PRt.T/d)@self.PRt)
        RNew = solve_triangular(chol, self.R, lower=True)
        upsilon = self.AinvM0 + muTilde
        aNew = DNew * upsilon
        gammaNew = RNew.T@(RNew@PNew.T@upsilon)
        vNew = DNew + np.sum((PNew@RNew.T)**2, 1)
        mNew = aNew + (PNew@gammaNew).reshape(-1)
        return mNew, vNew

    def derivEvidence(self, tauTilde, muTilde):
        mHat = muTilde / tauTilde - self.m0
        A = 1(1/tauTilde + self.D)
        B = cholInv(np.eye(self.m) + (self.PRt.T/A)@self.PRt)
        C = (self.PRt@B.T)*A
        e = A*mHat - C@C.T@mHat
        PRtR = self.PRt@self.R
        M1 = PRtR.T@C@C.T - PRtR.T*A
        M2 = 0.5*(M1@PRtR)
        v1 = 1/np.sqrt(2)*(np.sum(C**2, 1)-A)
        v2 = 1/np.sqrt(2)*np.ones(self.n)
        v3 = PRtR.T@e
        e2 = 1/np.sqrt(2)*e
        M1prima = M1-2*PRtR.T*(v1*v2+e2**2)
        M2prima = PRtR.T*(v1*v2+e2**2)@PRtR-M2
        dPdSigma = self.P/self.sigma*(self.sigma-1e-3)
        dKmmdSigma = (self.Kmm-self.sigma0*np.eye(self.m)-1e-6*np.eye(self.m))/self.sigma*(self.sigma-1e-3)
        dDiagKnnSigma = (self.diagKnn-self.sigma0-1e-6)/self.sigma*(self.sigma-1e-3)

        term1 = np.sum((v1*v2+e2**2)*dDiagKnnSigma)
        term2 = np.sum(M1prima*dPdSigma.T)
        term3 = np.sum(M2prima*dKmmdSigma)
        term4 = np.sum(e.T@dPdSigma@v3)
        term5 = np.sum(-0.5*v3.T@dKmmdSigma@v3)
        dLogZdSigma = term1+term2+term3+term4+term5

        dPdSigma0 = np.zeros(self.n, self.m)
        dKmmdSigma0 = self.sigma0*np.eye(self.m)
        dDiagKnndSigma0 = self.sigma0*np.ones(self.n)

        dLogZdl = np.zeros_like(self.l)
        for i in range(len(self.l)):
            Q = self.Xl[:,i].reshape(-1, 1)
            Qbar = self.Xbarl[:,i].reshape(-1, 1)
            dist = Q**2+Qbar.T**2-2*Q@Qbar.T
            dPdl = self.P*dist

            dist = Qbar**2+Qbar.T**2-2*Qbar@Qbar.T
            dKmmdl = -(self.Kmm-(self.sigma0+1e-6)*np.eye(self.m))*dist
            dDiagKnndl = np.zeros(self.n)

            term1 = np.sum((v1*v2+e2**2)*dDiagKnndl)
            term2 = np.sum(M1prima*dPdl.T)
            term3 = np.sum(M2prima*dKmmdl)
            term4 = np.sum(e.T@dPdl@v3)
            term5 = np.sum(-0.5*v3.T@dKmmdl@v3)
            dLogZdl[i] = term1+term2+term3+term4+term5

        dLogZdXbar = np.zeros(self.m, self.d)
        for i in range(len(self.l)):
            Q = self.Xl[:,i].reshape(-1, 1)
            Qbar = self.Xbarl[:,i].reshape(-1, 1)
            dist = Q**2+Qbar.T**2-2*Q@Qbar.T
            dPdXbar = self.P*dist*self.l[i]

            dist = Qbar**2+Qbar.T**2-2*Q@Qbar.T
            dKmmdXbar = (self.Kmm-(self.sigma0+1e-6)*np.eye(self.m))*dist*self.l[i]

            dDiagKnndXbar = np.zeros(self.n)

            term1 = np.sum((v1*v2+e2**2)*dDiagKnndXbar)*np.ones(self.m)
            term2 = np.sum(M1prima*dPdXbar.T, 1)
            term3 = -2*np.sum(M2prima*dKmmdXbar, 1)
            term4 = np.sum(dPdXbar.T@e*v3)
            term5 = -dKmmdXbar.T@v3*v3
            dLogZdZbar[:,i] = term1+term2+term3+term4*term5

        dLogZdm0 = np.sum(e)

        return {"sigma": dLogZdSigma,
                "sigma0": dLogZdSigma0,
                "l" : dLogZdl,
                "Xbar": dLogZdXbar,
                "m0": dLogZdm0}

if __name__ == '__main__':
    import pandas as pd
    X = np.array(pd.read_csv('data'))
    Xbar = X[1::2]
    model = GFITC(X, Xbar, [0,0])
    print(model.evidence(np.ones(100), np.ones(100)))

try:
    if __IPYTHON__:
        import pandas as pd
        X = np.array(pd.read_csv('data'))
        Xbar = X[1::2]
        model = GFITC(X, Xbar, [0,0])
except:
    pass
