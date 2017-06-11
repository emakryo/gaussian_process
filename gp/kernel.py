import numpy as np


class RBF():
    def __init__(self, X1, X2=None, sigma=1, beta=1, diag=False):
        """
        k(x_i, x_j) = \Sigma exp{ \beta (x_i - x_j)^T (x_i - x_j) }
        """
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

    def ___dK_dZ(self):
        """
        __dK_dZ[i,j,d] = \frac{\patial k(z_i, z_j)}{\partial z_{i,d}}
        """
        return self.__K[:, :, np.newaxis] * self.params['beta'] * self.diff / 2

    def dK_dZi(self, ix):
        i, j = self.__K.shape
        d = np.zeros((i, j, self.dim))
        if self.X2 is not None:
            d[ix, :, :] = self.__dK_dZ[ix]
        else:
            d[ix, :, :] = self.__dK_dZ[ix]
            d[:, ix, :] = -self.__dK_dZ[ix]

        return d

    def dK_dZj(self, jx):
        i, j = self.__K.shape
        d = np.zeros((i, j, self.dim))
        if self.X2 is not None:
            d[:, jx, :] = -self.__dK_dZ[jx]
        else:
            d[jx, :, :] = self.__dK_dZ[jx]
            d[:, jx, :] = -self.__dK_dZ[jx]

        return d

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
        return [(1e-10, None), (1e-10, None)]

    @staticmethod
    def param_dict(arr):
        assert len(arr) == 2
        return dict(beta=arr[0], sigma=arr[1])

    @staticmethod
    def param_array(dic):
        return np.array([dic['beta'], dic['sigma']])

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
