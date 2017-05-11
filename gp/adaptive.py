import numpy as np
from gaussian_process import kernel
from scipy import matmul
from scipy.linalg import cholesky, solve_triangular, solve, inv, det
from scipy.optimize import minimize
from matplotlib import pyplot as plt


def data1(n=50, sgm=0.3):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.sin(X)[:, 0] + np.random.randn(n) * sgm
    return (X, y)


def syn(n=50):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.cos(X[:, 0] / 6) + 0.3 * np.cos(X[:, 0] / 3) * np.random.randn(n)
    return (X, y)


def data2(n=50):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10
    y = np.cos(X)[:, 0]
    return (X, y)


class AdaptiveRegression():
    def __init__(self, k=kernel.Kernel(kernel.RBF, sgm=1., beta=1.), sgm=1.):

        self.params = {'sgm': sgm}
        self.kernel = k

    def fit(self, X, y):
        assert X.ndim == 2
        self.X = X

        assert y.ndim in [1, 2]
        if y.ndim == 1:
            self.y = y.reshape(-1, 1)
        else:
            self.y = y
        assert self.y.shape[1] == 1

        assert self.X.shape[0] == self.y.shape[0]

        self.n, self.dim = self.X.shape

        self.K = self.kernel(self.X)

        self.err = self.predict(X) - y

    def log_marginal_likelihood(self):
        raise NotImplementedError()
        #Ky = self.K()+self.params['sgm']*np.identity(self.n)
        #L = cholesky(Ky, lower=True)
        #alpha = solve_triangular(L, self.y, lower=True)[:,0]

        # return (- 0.5 * np.dot(alpha, alpha)
        #        - np.sum(np.log(np.diagonal(L)))
        #        - 0.5 * self.n * np.log(2*np.pi))

    def grad_log_marginal_likelihood(self):
        raise NotImplementedError()
        #Ky = self.K()+self.params['sgm']*np.identity(self.n)
        #Ky_inv = inv(Ky)
        #alpha = solve(Ky, self.y, sym_pos=True)
        #A = matmul(alpha, alpha.T) - Ky_inv
        #K_grads = [np.identity(self.n)]+list(self.K.dK_dtheta())

        # return np.array([0.5 * np.sum(A * dK_dtheta) for dK_dtheta in
        # K_grads])

    def optimize(self, random=False, verbose=False):
        raise NotImplementedError()
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
                       bounds=[(1e-10, None)] + self.kernel.bounds,
                       options={'disp': verbose})

        if res.fun < before[1]:
            self.param_array = res.x
        else:
            self.param_array = before[0]

    @property
    def param_array(self):
        return np.concatenate(([self.params['sgm']], self.kernel.param_array))

    @param_array.setter
    def param_array(self, arr):
        self.params['sgm'] = arr[0]
        self.kernel.param_array = arr[1:]
        self.K = self.kernel(self.X)

    def predict(self, Xn):
        """
        Xn: (m, d)
        X: (n, d)
        y: (n, 1)
        """
        k = self.kernel(self.X, Xn)
        K = self.kernel(self.X)
        m = Xn.shape[0]

        L = cholesky(K() + self.params['sgm'] *
                     np.identity(self.n), lower=True)
        V = solve_triangular(L, k(), lower=True)  # (n, m)

        alpha = solve_triangular(L, self.y[:, 0], lower=True)  # (n,)
        yn = np.dot(alpha, V)  # (m,)

        return yn

    def predict_var(self, Xn):

        k = self.kernel(self.X, Xn)
        K = self.kernel(self.X)
        m = Xn.shape[0]

        L = cholesky(K() + self.params['sgm'] *
                     np.identity(self.n), lower=True)
        V = solve_triangular(L, k(), lower=True)  # (n, m)
        beta = solve_triangular(L, self.err**2, lower=True)
        vn = np.dot(beta, V)

        return vn

    def plot(self, range_=None, output=None):
        """
data = (X,y) : input and output array.
range_ = (min, max) : range to draw.
        """
        if self.dim == 1:
            fig = plt.figure()
            if range_ is None:
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


def main():
    X, y = data1(100, 0.2)
    model = Regression()
    model.fit(X, y)
    for i in range(10):
        model.optimize(True)

    model.plot((-1, 11), "output.png")
    plt.show()
    print(model.grad_log_marginal_likelihood())
    print(model.log_marginal_likelihood())
    print(model.param_array)
    print(model.kernel.params)

#    x = np.linspace(0.002, 0.01, 10)
#    y = np.zeros(10)
#    yg = np.zeros(10)
#    for i,xn in enumerate(x):
#        param_array = model.param_array
#        param_array[0] = xn
#        model.param_array = param_array
#        y[i]  = model.log_marginal_likelihood()
#        yg[i] = model.grad_log_marginal_likelihood()[0]
#
#    plt.clf()
#    plt.plot(x,y)
#    plt.savefig('obj.png')
#    plt.clf()
#    plt.plot(x,yg)
#    plt.savefig('grad.png')


if __name__ == "__main__":
    main()
