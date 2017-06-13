import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from . import kernel
from .EPClassification import EPClassification


class privEPClassification(EPClassification):
    def __init__(self, kern=kernel.PrivAddRBF, **k_params):
        self.kern = kern
        self.k_params = k_params

    def fit(self, X, y, Z):
        """
        X: input
        y: output
        Z: privileged input
        """
        self.X = (X, Z)
        self.n = X.shape[0]
        self.y = y
        self._fit()

    def posterior(self, Xs):
        Ks = self.kern(self.X, (Xs, None), Zsample=self.X[1], **self.k_params)()
        Kss = np.diag(self.kern((Xs, None), (Xs, None), Zsample=self.X[1],
            **self.k_params)())
        return self._posterior(Ks, Kss)
