import numpy as np

class Kernel():
    def __init__(self, cls, **params):
        self.cls = cls
        self.params = params
        self.bounds = cls.bounds()

    def __call__(self, X1, X2=None, **kwargs):
        return self.cls(X1, X2, **self.params)

    @property
    def param_array(self):
        return np.array([self.params[k] for k in sorted(self.params.keys())])

    @param_array.setter
    def param_array(self, arr):
        for i, k in enumerate(sorted(self.params.keys())):
            self.params[k] = arr[i]


class RBF():
    def __init__(self, X1, X2=None, sgm=1, beta=1, param_X1=False, param_X2=False):
        """
        k(x_i, x_j) = \sgm exp{ \beta (x_i - x_j)^T (x_i - x_j) }
        """
        if X1.ndim == 2: self.X1 = X1
        elif X1.ndim == 1: self.X1 = X1.reshape(-1,1)
        else:
            raise ValueError(
                "The number of dimension of X1 higher than 2: X1.ndim=%d" % X1.ndim)

        if X2 is None: self.X2 = None
        elif X2.ndim == 2: self.X2 = X2
        elif X2.ndim == 1: self.X2 = X2.reshape(-1,1)
        else:
            raise ValueError(
                "The number of dimension of X2 higher than 2: X2.ndim=%d" % X2.ndim)

        assert X2 is None or X1.shape[1] == X2.shape[1],\
                     ("The dimension of data does not match: X1.shape=%s, X2.shape=%s" %
                      (str(X1.shape), str(X2.shape)))

        self.dim = X1.shape[1]

        self.params = {'sgm':sgm, 'beta':beta}

        self.__K = self.___K()
        self.__dK_dZ = self.___dK_dZ()


    def __call__(self):
        return self.__K

    def ___K(self):

        i = self.X1.shape[0]
        if self.X2 is None: j = i
        else: j = self.X2.shape[0]

        X1 = np.repeat(self.X1.reshape(i,1,self.dim), j, 1)
        if self.X2 is None: X2 = np.repeat(self.X1.reshape(1,j,self.dim), i, 0)
        else: X2 = np.repeat(self.X2.reshape(1,j,self.dim), i, 0)

        self.diff = X1-X2
        self.norm2 = np.sum(self.diff**2, 2)

        return self.params['sgm'] * np.exp(- 0.5 * self.params['beta'] * self.norm2)


    def ___dK_dZ(self):
        """
        __dK_dZ[i,j,d] = \frac{\patial k(z_i, z_j)}{\partial z_{i,d}}
        """
        return self.__K[:,:,np.newaxis] * self.params['beta'] * self.diff / 2


    def dK_dZi(self, ix):
        i, j = self.__K.shape
        d = np.zeros((i,j,self.dim))
        if self.X2 is not None:
            d[ix,:,:] = self.__dK_dZ[ix]
        else:
            d[ix,:,:] = self.__dK_dZ[ix]
            d[:,ix,:] = -self.__dK_dZ[ix]

        return d

    def dK_dZj(self, jx):
        i, j = self.__K.shape
        d = np.zeros((i,j,self.dim))
        if self.X2 is not None:
            d[:,jx,:] = -self.__dK_dZ[jx]
        else:
            d[jx,:,:] = self.__dK_dZ[jx]
            d[:,jx,:] = -self.__dK_dZ[jx]

        return d

    def dK_dbeta(self):
        return - 0.5 * self.__K * self.norm2

    def dK_dsgm(self):
        return np.exp(- 0.5 * self.params['beta'] * self.norm2)

    def dK_dtheta(self):
        return self.dK_dbeta(), self.dK_dsgm()

    @staticmethod
    def bounds():
        return [(1e-10,None), (1e-10,None)]

if __name__ == "__main__":
    k = Kernel(RBF, sgm=1, beta=1)

    x = np.random.rand(2,3)
    print(x)
    kxx = k(x)
    print(kxx())
    print(kxx.norm2)
    print(kxx.dK_dsgm())
    print(kxx.dK_dbeta())
