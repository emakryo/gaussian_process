import numpy as np
from sklearn.utils import check_random_state
from .kernels import RBF


def generate_gp(n_samples=50, n_features=1, noise=0.01, range=(-3, 3),
                kernel=RBF(sigma=1.0, beta=1.0), random_state=None):
    generator = check_random_state(random_state)

    X = generator.rand(n_samples, n_features) * (range[1]-range[0]) + range[0]
    K = kernel(X).K
    y = generator.multivariate_normal(np.zeros(n_samples), K + noise * np.eye(n_samples))
    return X, y
