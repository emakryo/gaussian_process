import numpy as np
import kernel
from scipy import matmul
from scipy.linalg import cholesky, solve_triangular, solve, inv, det
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def data1(n=50, sgm=0.3):
    np.random.seed(0)
    X = np.random.rand(n,1) * 10
    y = np.sin(X)[:,0] + np.random.randn(n)*sgm
    return (X,y)

def data2(n=50):
    np.random.seed(0)
    X = np.random.rand(n,1) * 10
    y = np.cos(X)[:,0]
    return (X,y)


class Regression():
    def __init__(self, X, y, k=kernel.Kernel(kernel.RBF, sgm=1., beta=1.), sgm=1.):
        assert X.ndim == 2
        self.X = X

        assert y.ndim in [1,2]

        if y.ndim == 1: self.y = y.reshape(-1,1)
        else: self.y = y

        assert self.X.shape[0] == self.y.shape[0]
        assert self.y.shape[1] == 1

        self.n, self.dim = self.X.shape

        self.params = {'sgm':sgm}

        self.kernel = k
        self.K = self.kernel(self.X)

    def log_marginal_likelihood(self):
        Ky = self.K()+self.params['sgm']*np.identity(self.n)
        L = cholesky(Ky, lower=True)
        alpha = solve_triangular(L, self.y, lower=True)[:,0]

        return (- 0.5 * np.dot(alpha, alpha)
                - np.sum(np.log(np.diagonal(L)))
                - 0.5 * self.n * np.log(2*np.pi))

    def grad_log_marginal_likelihood(self):
        Ky = self.K()+self.params['sgm']*np.identity(self.n)
        Ky_inv = inv(Ky)
        alpha = solve(Ky, self.y, sym_pos=True)
        A = matmul(alpha, alpha.T) - Ky_inv
        K_grads = [np.identity(self.n)]+list(self.K.dK_dtheta())

        return np.array([0.5 * np.sum(A * dK_dtheta) for dK_dtheta in K_grads])


    def optimize(self, random=False):
        before = (self.param_array, -self.log_marginal_likelihood())
        if random:
            init = np.random.rand(*self.param_array.shape)*3
        else:
            init = self.param_array

        def fun(param_array):
            self.param_array = param_array
            return (-self.log_marginal_likelihood(),
                    -self.grad_log_marginal_likelihood())

        res = minimize(fun, init, method='L-BFGS-B', jac=True,
                       bounds=[(1e-10,None)]+self.kernel.bounds,
                       options={'disp':True})

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

    def predict(self,Xn):
        """
        Xn: (m, d)
        X: (n, d)
        y: (n, 1)
        """
        k = self.kernel(self.X, Xn)
        K = self.kernel(self.X)
        m = Xn.shape[0]

        L = cholesky(K()+self.params['sgm']*np.identity(self.n), lower=True)

        alpha = solve_triangular(L, self.y[:,0], lower=True) # (n,)
        V = solve_triangular(L, k(), lower=True) # (n, m)

        yn = np.dot(alpha, V) # (m,)
        vn = (self.kernel(Xn)() - np.dot(V.T,V)
              + self.params['sgm'] * np.identity(m)) # (m, m)

        return yn, vn

    def plot(self, range_, output):
        """
data = (X,y) : input and output array.
range_ = (min, max) : range to draw.
mean : function to predicted mean
var; function to predicted variance
        """
        if self.dim == 1:
            plt.xlim()
            plt.plot(self.X[:,0], self.y[:,0], "bx")
            x = np.linspace(range_[0], range_[1], 100).reshape(-1,1)
            yn, vn = self.predict(x)
            std = np.sqrt(vn.diagonal())
            plt.plot(x, yn, "k-")
            plt.fill_between(x.reshape(-1), yn-std, yn+std,
                             alpha=0.3, facecolor="black")
            plt.savefig(output)
        else:
            assert True, "Only 1 dimension is supported for input"


if __name__ == "__main__":
    X,y=data1(50,0.1)
    model = Regression(X,y)
    for i in range(10):
        model.optimize(True)

    model.plot((-1,11), "output.png")
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
