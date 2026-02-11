from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import Reducer


class ClassicalPCA(Reducer):
    """Classical PCA baseline (sklearn).

    The paper compares against PCA + SVM and Kernel PCA baselines.
    This reducer gives you a clean baseline that you should always benchmark first.
    """

    def __init__(self, n_components: int = 8, random_state: int = 7):
        self.n_components = n_components
        self.random_state = random_state
        self._pca = PCA(n_components=n_components, random_state=random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(X)
