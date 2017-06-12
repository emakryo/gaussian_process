import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from . import kernel
from .EPClassification import EPClassification


class privEPClassification(EPClassification):
    def __init__(self, kernel=kernel.PrivAddRBF,
            sigmax=1, betax=1, sigmaz=1, betaz=1, alpha=0.5):
        self.kernel = kernel
        self.k_params = {'sigmax': sigmax, 'betax': betax,
                'sigmaz': sigmaz, 'betaz': betaz, 'alpha': alpha}

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
        Ks = self.kernel(self.X, (Xs, None), Zsample=self.X[1], **self.k_params)()
        Kss = np.diag(self.kernel((Xs, None), (Xs, None), Zsample=self.X[1],
            **self.k_params)())
        return self._posterior(Ks, Kss)
