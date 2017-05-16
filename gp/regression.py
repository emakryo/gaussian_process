import numpy as np
from . import kernel
from scipy.linalg import cholesky, solve_triangular, solve, inv
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class Regression():
    def __init__(self, sgm=1., kern=kernel.RBF,
                 k_params=dict(sgm=1., beta=1.)):

        self.params = {'sgm': sgm}
        self.kernel = kern
        self.k_params = k_params

    def fit(self, X, y):
        assert X.ndim == 2
        self.X = X
        self.y = y.reshape(-1, 1)

        assert self.X.shape[0] == self.y.shape[0]

        self.n, self.dim = self.X.shape

        self.K = self.kernel(self.X, **self.k_params)

    def log_marginal_likelihood(self):
        Ky = self.K() + self.sigma * np.identity(self.n)
        L = cholesky(Ky, lower=True)
        alpha = solve_triangular(L, self.y, lower=True)[:, 0]

        t1 = - 0.5 * np.dot(alpha, alpha)
        t2 = - np.sum(np.log(np.diagonal(L)))
        t3 = - 0.5 * self.n * np.log(2 * np.pi)
        return t1+t2+t3

    def grad_log_marginal_likelihood(self):
        Ky = self.K() + self.sigma * np.identity(self.n)
        Ky_inv = inv(Ky)
        alpha = solve(Ky, self.y, sym_pos=True)
        A = alpha @ alpha.T - Ky_inv
        K_grads = [np.identity(self.n)] + list(self.K.dK_dtheta())

        return np.array([0.5 * np.sum(A * dK_dtheta) for dK_dtheta in K_grads])

    def optimize(self, random=False, verbose=False):
        before = (self.param_array, -self.log_marginal_likelihood())
        if random:
            init = np.random.rand(*self.param_array.shape) * 3
        else:
            init = self.param_array

        def fun(param_array):
            self.param_array = param_array
            return (-self.log_marginal_likelihood(),
                    -self.grad_log_marginal_likelihood())

        res = minimize(fun, init, method='L-BFGS-B', jac=True,
                       bounds=[(1e-10, None)] + self.kernel.bounds(),
                       options={'disp': verbose})

        if res.fun < before[1]:
            self.param_array = res.x
        else:
            self.param_array = before[0]

    @property
    def param_array(self):
        return np.concatenate(([self.sigma],
                               [self.k_params[k] for k in self.k_params]))

    @param_array.setter
    def param_array(self, arr):
        self.params['sgm'] = arr[0]
        self.k_params = self.kernel.params(arr[1:])
        self.K = self.kernel(self.X, **self.k_params)

    @property
    def sigma(self):
        return self.params['sgm']

    @sigma.setter
    def sigma(self, x):
        self.params['sgm'] = x

    def predict(self, Xn, variance=True):
        """
        Xn: (m, d)
        X: (n, d)
        y: (n, 1)
        """
        k = self.kernel(self.X, Xn, **self.k_params)
        K = self.kernel(self.X, **self.k_params)
        m = Xn.shape[0]

        L = cholesky(K() + self.sigma * np.identity(self.n), lower=True)

        alpha = solve_triangular(L, self.y[:, 0], lower=True)  # (n,)
        V = solve_triangular(L, k(), lower=True)  # (n, m)

        yn = np.dot(alpha, V)  # (m,)
        vn_t1 = self.kernel(Xn, **self.k_params)() - np.dot(V.T, V)
        vn_t2 = self.sigma * np.identity(m)  # (m, m)

        if variance:
            return yn, vn_t1+vn_t2
        else:
            return yn

    def plot(self, output=None):
        """
        data = (X,y) : input and output array.
        range_ = (min, max) : range to draw.
        """
        if self.dim == 1:
            fig = plt.figure()
            range_ = (self.X[:, 0].min(), self.X[:, 0].max())

            plt.xlim()
            plt.plot(self.X[:, 0], self.y[:, 0], "bx")
            x = np.linspace(range_[0], range_[1], 100).reshape(-1, 1)
            yn, vn = self.predict(x)
            std = np.sqrt(vn.diagonal())
            plt.plot(x, yn, "k-")
            plt.fill_between(x.reshape(-1), yn - std, yn + std,
                             alpha=0.3, facecolor="black")
            if output:
                plt.savefig(output)
            return fig
        else:
            assert True, "Only 1 dimension is supported for input"
