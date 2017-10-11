import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator


class BayesEstimator(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def empirical_bayes(self):
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
