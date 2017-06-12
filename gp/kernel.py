import numpy as np


class RBF():
    """
    k(x_i, x_j) = \Sigma exp{ \beta (x_i - x_j)^T (x_i - x_j) }
    """

    def __init__(self, X1, X2=None, sigma=1, beta=1, diag=False):
        assert X1.ndim == 2
        self.X1 = X1
        if X2 is None:
            self.X2 = X1
        else:
            assert X2.ndim == 2
            self.X2 = X2

        self.n1, self.dim1 = self.X1.shape
        self.n2, self.dim2 = self.X2.shape
        self.params = {'sigma': sigma, 'beta': beta}
        if not diag:
            X1sq = np.sum(self.X1**2, 1).reshape(-1, 1)
            X2sq = np.sum(self.X2**2, 1).reshape(-1, 1)
            self._diff = X1sq + X2sq.T - 2*self.X1@self.X2.T
            self._K = self.sigma * np.exp(-0.5*self.beta*self._diff)
        else:
            self._K = self.sigma * np.ones(min(self.n1, self.n2))

        self.diag = diag

    def __call__(self):
        return self._K

    def dbeta(self, diag=False):
        if self.diag:
            return np.zeros(min(self.n1, self.n2))
        else:
            return - 0.5 * self._K * self._diff

    def dsigma(self, diag=False):
        if self.diag:
            return np.ones(min(self.n1, self.n2))
        else:
            return self._K/self.sigma

    def dtheta(self):
        return np.stack([self.dbeta(), self.dsigma()], axis=2)

    @property
    def param_array(self):
        return np.array([self.params['beta'], self.params['sigma']])

    @param_array.setter
    def param_array(self, arr):
        self.params['beta'], self.params['sigma'] = arr

    @property
    def sigma(self):
        return self.params['sigma']

    @sigma.setter
    def sigma(self, x):
        self.params['sigma'] = x

    @property
    def beta(self):
        return self.params['beta']

    @beta.setter
    def beta(self, x):
        self.params['beta'] = x

    @staticmethod
    def bounds():
        return [(1e-5, None), (1e-5, None)]

    @staticmethod
    def param_dict(arr):
        assert len(arr) == 2
        return dict(beta=arr[0], sigma=arr[1])

    @staticmethod
    def param_array(dic):
        return np.array([dic['beta'], dic['sigma']])

class PrivAddRBF():
    def __init__(self, XZ1, XZ2=None, Zsample=None, alpha=0.5,
            sigmax=1, betax=1, sigmaz=1, betaz=1):
        """
        X1, X2: input
        Z1, Z2: privileged input
        """

        if XZ2 is None:
            XZ2 = XZ1

        self.X1 = XZ1[0]
        self.Z1 = XZ1[1]
        self.X2 = XZ2[0]
        self.Z2 = XZ2[1]

        kx = RBF(self.X1, self.X2, sigmax, betax)
        if self.Z1 is None and self.Z2 is None:
            kz = RBF(Zsample, Zsample, sigmaz, betaz)
        elif self.Z2 is None:
            kz = RBF(self.Z1, Zsample, sigmaz, betaz)
        else:
            kz = RBF(self.Z1, self.Z2, sigmaz, betaz)

        self.kx = kx
        self.kz = kz
        self.alpha = alpha

    def __call__(self):
        alpha = self.alpha
        if self.Z1 is None and self.Z2 is None:
            return alpha*self.kx() + (1-alpha)*np.mean(self.kz())
        elif self.Z2 is None:
            return alpha*self.kx() + (1-alpha)*np.mean(self.kz(), axis=1).reshape(-1, 1)
        else:
            return alpha*self.kx() + (1-alpha)*self.kz()

class PrivMultRBF():
    def __init__(self, XZ1, XZ2=None, Zsample=None,
            sigmax=1, betax=1, sigmaz=1, betaz=1):
        """
        X1, X2: input
        Z1, Z2: privileged input
        """

        if XZ2 is None:
            XZ2 = XZ1

        self.X1 = XZ1[0]
        self.Z1 = XZ1[1]
        self.X2 = XZ2[0]
        self.Z2 = XZ2[1]

        kx = RBF(self.X1, self.X2, sigmax, betax)
        if self.Z1 is None and self.Z2 is None:
            kz = RBF(Zsample, Zsample, sigmaz, betaz)
        elif self.Z2 is None:
            kz = RBF(self.Z1, Zsample, sigmaz, betaz)
        else:
            kz = RBF(self.Z1, self.Z2, sigmaz, betaz)

        self.kx = kx
        self.kz = kz

    def __call__(self):
        if self.Z1 is None and self.Z2 is None:
            return self.kx()*np.mean(self.kz())
        elif self.Z2 is None:
            return self.kx()*np.mean(self.kz(), axis=1).reshape(-1, 1)
        else:
            return self.kx()*self.kz()

class IncompleteRBF():
    def __init__(self, X1, X2=None, Zsample=None, alpha=0.5,
            sigma=1, beta=1):
        """
        X1, X2: input
        Zsample: privileged information
        """

        if X2 is None:
            X2 = X1

        if Zsample is None or X1.shape[1] == X2.shape[1] == Zsample.shape[1]:
            kx = RBF(X1, X2, sigma, beta)
            kz = None
            self.incomplete = 0
        elif Zsample.shape[1] == X1.shape[1] > X2.shape[1]:
            dim = X2.shape[1]
            kx = RBF(X1[:, :dim], X2[:, :dim], sigma, beta)
            kz = RBF(X1[:, dim:], Zsample[:, dim:], sigma, beta)
            self.incomplete = 1
        else:
            dim = min(X1.shape[1], X2.shape[1])
            kx = RBF(X1[:, :dim], X2[:, :dim], sigma, beta)
            kz = RBF(Zsample[:, dim:], Zsample[:, dim:], sigma, beta)
            self.incomplete = 2

        self.kx = kx
        self.kz = kz
        self.alpha = alpha

    def __call__(self):
        alpha = self.alpha
        if self.incomplete == 0:
            return self.kx()
        elif self.incomplete == 1:
            return self.kx()*np.mean(self.kz(), axis=1).reshape(-1, 1)**alpha
        else:
            return self.kx()*np.mean(self.kz())**alpha

class ARD():
    def __init__(self, X1, X2=None, ls=1., sigma=1., diag=False):
        assert X1.ndim == 2
        self.X1 = X1
        if X2 is None:
            self.X2 = X1
        else:
            self.X2 = X2

        assert self.X1.shape[1] == self.X2.shape[1]
        self.n1, self.dim = self.X1.shape
        self.n2 = self.X2.shape[0]

        ls = ls * self.ones(self.dim)
        self.params = {'ls': ls, 'sgm': sigma}

        if not diag:
            X1sq = np.sum((self.X1*ls.reshape(1, -1))**2, 1).reshape(-1, 1)
            X2sq = np.sum((self.X2*ls.reshape(1, -1))**2, 1).reshape(-1, 1)
            self._diff = X1sq+X2sq.T-2*self.X1@self.X2.T

    @property
    def sigma(self):
        return self.params['sgm']

    @sigma.setter
    def sigma(self, x):
        self.params['sgm'] = x

    @property
    def ls(self):
        return self.parmas['ls']

    @ls.setter
    def ls(self, x):
        self.params['ls'] = x*np.ones(self.dim)

    def __call__(self):
        if self.diag:
            return self.sigma*np.ones(min(self.n1, self.n2))
        else:
            return self.sigma * np.exp(-0.5*self._diff)
