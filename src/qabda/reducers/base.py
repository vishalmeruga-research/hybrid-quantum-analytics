from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Reducer(ABC):
    """Interface for dimensionality reduction.

    In the paper, this corresponds to the “Quantum-Assisted Dimensionality Reduction”
    module and its classical baseline comparisons.
    """

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...
