import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular, cholesky
from . import kernel


def cholInv(A):
    """return upper triangular U s.t. U*U.T = A^-1"""
    return solve_triangular(cholesky(A), np.eye(A.shape[0]))


class FITCRegression():

    def __init__(self, X, y, m, sgm=0.1, kern=kernel.RBF,
                 k_params=dict(sgm=1., beta=1.)):

        assert X.ndim == 2, \
            "The number of dimension of X must be 2: X.ndim=%d" % X.ndim
        self.X = X
        assert y.ndim in [1, 2], \
            "The number of dimension of y is higher than 2: y.ndim=%d" % y.ndim

        self.y = np.array(y).reshape(-1, 1)

        assert X.shape[0] == y.shape[0], "data size does not match"

        self.n, self.dim = X.shape
        self.m = m
        self.Z = X[np.random.permutation(self.n)[:m]]
        self.params = {'sgm': sgm}
        self.kernel = kern
        self.k_params = k_params

    def predict(self, Xp):
        assert Xp.shape[1] == self.dim
        K = self.kernel(self.X, **self.k_params, diag=True)()
        Kxm = self.kernel(self.X, self.Z, **self.k_params)()
        Kmm = self.kernel(self.Z, self.Z, **self.k_params)()
        cholInvKmm = cholInv(Kmm+1e-6*np.eye(self.m))
        Q = np.sum((Kxm@cholInvKmm)**2, 1)
        gamma = K-Q+self.sigma
        SigmaInv = Kmm + (Kxm.T/gamma)@Kxm
        Sigmasq = cholInv(SigmaInv+1e-6*np.eye(self.m))

        Kpp = self.kernel(Xp, **self.k_params, diag=True)()
        Kpm = self.kernel(Xp, self.Z, **self.k_params)()
        Qpp = np.sum((Kpm@cholInvKmm)**2, 1)
        P = Kpm@Sigmasq
        mean = P@Sigmasq.T@Kxm.T@(self.y.reshape(-1)/gamma)
        var = Kpp-Qpp+np.sum(P**2, 1)

        return mean, var

    def plot(self, output=None):
        assert self.dim == 1
        dom = self.X.max()-self.X.min()
        Xp = np.linspace(self.X.min()-0.1*dom,
                         self.X.max()+0.1*dom)
        mean, var = self.predict(Xp.reshape(-1, 1))

        fig = plt.figure()
        plt.plot(self.X[:, 0], self.y[:, 0], "bx")
        mean = mean.reshape(-1)
        std = np.sqrt(var).reshape(-1)
        plt.plot(Xp, mean, "k-")
        plt.fill_between(Xp, mean - std, mean + std,
                         alpha=0.3, facecolor="black")
        mean, _ = self.predict(self.Z)
        plt.plot(self.Z[:, 0], mean.reshape(-1), "ro")

        if output:
            plt.savefig(output)

        return fig

    def log_marginal_likelihood_lower_bound(self):
        return 0

    def grad_log_marginal_likelihood_lower_bound(self):
        return np.zeros(len(self.param_array))

    @property
    def sigma(self):
        return self.params['sgm']

    @sigma.setter
    def sigma(self, x):
        self.params['sgm'] = x

    @property
    def param_array(self):
        return np.concatenate(([self.params['beta']],
                               self.kernel.param_array,
                               self.Z.flatten()))

    @param_array.setter
    def param_array(self, arr):
        self.params['beta'] = arr[0]
        self.kernel.param_array = arr[:self.kernel.num_param + 1]
        self.Z = arr[self.kernel.num_param + 1:].reshape(self.m, -1)
