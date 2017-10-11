import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve, inv
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y


class GaussianProcessRegression(BaseEstimator, RegressorMixin):
    """Ordinary Gaussian process regression"""

    def __init__(self, cov, sigma=0.01):
        self.cov = cov
        self.sigma = sigma
        self.param_bounds = {'sigma': (1e-5, None)}
        for k, v in cov.param_bounds.items():
            self.param_bounds['cov__' + k] = v

    def fit(self, X, y, empirical_bayes=False):
        self.Xtr, self.ytr = check_X_y(X, y, y_numeric=True)
        self.n, self.dim = self.Xtr.shape

        if empirical_bayes:
            self.empirical_bayes()

    def predict(self, X):
        return self.predict_with_variance(X)[0]

    def predict_with_variance(self, X):
        K = self.cov(self.Xtr)
        k = self.cov(self.Xtr, X)
        m = X.shape[0]

        L = cholesky(K.K + self.sigma * np.eye(self.n), lower=True)
        alpha = solve_triangular(L, self.ytr, lower=True)  # (n,)
        V = solve_triangular(L, k.K, lower=True)  # (n, m)
        yn = np.dot(alpha, V)  # (m,)

        vn_t1 = self.cov(X).K - np.dot(V.T, V)
        vn_t2 = self.sigma * np.identity(m)  # (m, m)

        return yn, vn_t1 + vn_t2

    def log_marginal_likelihood(self):
        K = self.cov(self.Xtr)
        Ky = K.K + self.sigma * np.eye(self.n)
        L = cholesky(Ky, lower=True)
        alpha = solve_triangular(L, self.ytr, lower=True)

        t1 = - 0.5 * np.dot(alpha, alpha)
        t2 = - np.sum(np.log(np.diagonal(L)))
        t3 = - 0.5 * self.n * np.log(2 * np.pi)
        return t1 + t2 + t3

    def grad_log_marginal_likelihood(self):
        K = self.cov(self.Xtr)
        Ky = K.K + self.sigma * np.eye(self.n)
        Ky_inv = inv(Ky)
        alpha = solve(Ky, self.ytr, assume_a='pos').reshape(-1, 1)
        A = alpha @ alpha.T - Ky_inv
        K_grads = {'sigma':np.eye(self.n)}
        for param in K.dtheta:
            K_grads['cov__'+param] = K.dtheta[param]

        grads = {}
        for k, v in K_grads.items():
            if v.ndim == 2:
                grads[k] = np.array([0.5 * np.sum(A * v)])
            elif v.ndim == 3:
                grads[k] = 0.5 * np.sum(A * v, axis=(1, 2))
            else:
                raise ValueError("Invalid number of dimensional")

        return grads

    def empirical_bayes(self, opt_param_names=None, random=False, verbose=False):
        if opt_param_names is None:
            opt_param_names = [k for k, v in self.get_params().items()
                               if np.isscalar(v) or isinstance(v, np.ndarray)]

        before_x = self.get_opt_params(opt_param_names)
        before_fun = -self.log_marginal_likelihood()
        if random:
            init = np.random.rand(len(before[0])) * 3
        else:
            init = before_x

        def fun(params):
            self.set_opt_params(params, opt_param_names)
            grads = self.grad_log_marginal_likelihood()
            return (-self.log_marginal_likelihood(),
                    -np.concatenate([grads[k].flatten()
                                     for k in opt_param_names]))

        bounds = []
        for k in opt_param_names:
            if type(self.param_bounds[k][0]) is tuple:
                bounds.extend(self.param_bounds[k])
            else:
                bounds.append(self.param_bounds[k])

        res = minimize(fun, init, method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'disp': verbose})

        if verbose:
            print(res.message)

        if res.fun < before_fun:
            self.set_opt_params(res.x, opt_param_names)
        else:
            self.set_opt_params(before_x, opt_param_names)

    def get_opt_params(self, param_names):
        params = self.get_params()
        opt_params = []
        for k in param_names:
            if np.isscalar(params[k]):
                opt_params.append(params[k])
            elif isinstance(params[k], np.ndarray):
                opt_params.extend(params[k].flatten())

        return np.array(opt_params)

    def set_opt_params(self, opt_params, param_names):
        current_params = self.get_params()
        params = {}
        index = 0
        for k in param_names:
            if np.isscalar(current_params[k]):
                params[k] = opt_params[index]
                index += 1
            elif isinstance(current_params[k], np.ndarray):
                param = opt_params[index:index+current_params[k].size]
                params[k] = param.reshape(*current_params[k].shape)
                index += params[k].size

        self.set_params(**params)
