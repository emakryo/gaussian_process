import numpy as np
from scipy.linalg import solve
import kernel


class SparseGP():

    def __init__(self, X, y, m, k=kernel.Kernel(kernel.RBF, sgm=1., beta=1.), beta=1.):

        assert X.ndim == 2, "The number of dimension of X must be 2: X.ndim=%d" % X.ndim
        self.X = X
        assert y.ndim in [1, 2], \
            "The number of dimension of y is higher than 2: y.ndim=%d" % y.ndim

        if y.ndim == 1:
            self.y = y.reshape(-1, 1)
        else:
            self.y = y

        assert X.shape[0] == y.shape[0], "data size does not match"
        assert y.shape[1] == 1
        self.dim = X.shape[1]
        self.Z = X[np.random.permutation(np.arange(self.dim))[:m]]
        self.params = {'beta': beta}
        self.kernel = k

    def predict(self, Xn):
        Kmm = kernel(self.Z)
        Kxm = kernel(Xn, self.Z)
        Knm = kernel(self.X, self.Z)
        A = Kmm() + matmul(Knm().T, Knm()) * self.params['beta']
        B = matmul(Knm().T, self.y)
        mean = 0
        var = 0

        return mean, var

    def plot(self):
        pass

    def log_marginal_likelihood_lower_bound(self):
        return 0

    def grad_log_marginal_likelihood_lower_bound(self):
        return np.zeros(len(param_array))

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
