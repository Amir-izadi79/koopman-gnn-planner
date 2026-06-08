import numpy as np
import torch
from scipy.linalg import lstsq


class EDMD:
    """Extended Dynamic Mode Decomposition for learning Koopman operators via RBF observables."""

    def __init__(self, obs_dim=10):
        self.obs_dim = obs_dim
        self.centers = None
        self.K = None

    def rbf_observables(self, X):
        return np.exp(-np.sum((X[:, None] - self.centers) ** 2, axis=2))

    def fit(self, X, Y):
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().numpy()

        self.centers = X[np.random.choice(len(X), self.obs_dim, replace=False)]
        Psi_X = self.rbf_observables(X)
        Psi_Y = self.rbf_observables(Y)
        self.K = lstsq(Psi_X, Psi_Y)[0]
        return self.K

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        psi = self.rbf_observables(x[None, :])
        return psi @ self.K
