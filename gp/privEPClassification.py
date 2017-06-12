import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from . import kernel
from .EPClassification import EPClassification


class privEPClassification(EPClassification):
    def __init__(self, kernel=kernel.eRBF, sigma=1, beta=1):
        self.kernel = kernel
        self.k_params = {'sigma': sigma, 'beta': beta}

    def fit(self, X, y, Z):
        """
        X: privileged input
        y: output
        Z: input
        """
        self.Z = Z
        super().fit(X, y)
