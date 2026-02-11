from __future__ import annotations

import numpy as np


def angle_encode(x: np.ndarray, *, n_qubits: int) -> np.ndarray:
    """Simple angle encoding helper.

    We convert a feature vector into `n_qubits` rotation angles.
    For production, you should decide:
    - which features map to which qubits
    - scaling to keep angles within a reasonable range

    This is *not* amplitude encoding (which is more complex to prepare). The paper
    discusses amplitude encoding conceptually; in practice, angle encoding is much
    more NISQ-friendly.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < n_qubits:
        x = np.pad(x, (0, n_qubits - x.size))
    elif x.size > n_qubits:
        x = x[:n_qubits]
    # squashing to [-pi, pi]
    return np.pi * np.tanh(x)


def zz_feature_map_angles(x: np.ndarray, *, n_qubits: int) -> np.ndarray:
    """Return angles for a simple ZZ-style feature map.

    The idea: create entangling structure so the model can represent nonlinear interactions.
    """
    return angle_encode(x, n_qubits=n_qubits)


def z_feature_map_angles(x: np.ndarray, *, n_qubits: int) -> np.ndarray:
    return angle_encode(x, n_qubits=n_qubits)
