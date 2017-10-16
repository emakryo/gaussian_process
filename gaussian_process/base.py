import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


class BayesEstimator(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        ...

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
            self.fit(self.Xtr, self.ytr)
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
        ...

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
