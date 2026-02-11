from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import Reducer

# Why a surrogate?
#
# The paper highlights qPCA's asymptotic advantages (poly(log d)) compared to classical O(d^3).
# However, 'true' qPCA requires assumptions that are not satisfied in most production contexts:
# - efficient state preparation (often phrased as QRAM or efficient oracle access)
# - density matrix exponentiation and quantum phase estimation at sufficient depth
# - enough clean qubits to represent the subspace
#
# For a GitHub repo that people can actually run today, we implement a NISQ-friendly surrogate:
# 1) Use a shallow quantum feature map to transform features into a richer representation
# 2) Run classical PCA on the resulting representation to obtain k components
#
# This preserves the "hybrid quantum-classical workflow" architecture and gives you a place to
# swap in improved quantum routines as hardware and libraries mature.

class QPCASurrogate(Reducer):
    """A qPCA-like reducer implemented as:
    quantum feature mapping -> classical PCA.

    Backends:
    - Qiskit Aer: uses statevector simulation to compute kernel-ish features (small qubits only)
    - PennyLane default.qubit: similar purpose

    If quantum deps aren't installed, it falls back to a deterministic random Fourier-like mapping.

    Parameters
    ----------
    n_components:
        Number of PCA components.
    feature_map:
        'zz_feature_map' or 'z_feature_map' (used only for quantum backends).
    reps:
        Depth / repetitions for the feature map (affects expressivity and simulation cost).
    n_qubits:
        Number of qubits used in the feature map representation.
    random_state:
        Seed for fallback mapping.
    """

    def __init__(
        self,
        n_components: int = 8,
        feature_map: str = "zz_feature_map",
        reps: int = 2,
        n_qubits: int | None = None,
        random_state: int = 7,
    ):
        self.n_components = n_components
        self.feature_map = feature_map
        self.reps = reps
        self.n_qubits = n_qubits  # if None, inferred from X at fit time (capped)
        self.random_state = random_state

        self._pca = PCA(n_components=n_components, random_state=random_state)

        self._W: np.ndarray | None = None  # fallback projection
        self._b: np.ndarray | None = None

    def _fallback_map(self, X: np.ndarray, out_dim: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        if self._W is None or self._b is None:
            self._W = rng.normal(size=(X.shape[1], out_dim))
            self._b = rng.uniform(0, 2 * np.pi, size=(out_dim,))
        Z = np.cos(X @ self._W + self._b)
        return Z

    def _quantum_map(self, X: np.ndarray) -> np.ndarray:
        # We produce a compact representation by running a shallow circuit and returning
        # expectation values of PauliZ on each qubit (like a quantum embedding).
        try:
            import pennylane as qml  # type: ignore
            from qabda.quantum.feature_maps import zz_feature_map_angles, z_feature_map_angles
        except Exception:
            # Quantum deps not available; use fallback
            return self._fallback_map(X, out_dim=max(2 * self.n_components, 16))

        n_qubits = self.n_qubits or min(12, X.shape[1], max(self.n_components, 2))
        if self.feature_map == "zz_feature_map":
            angle_fn = zz_feature_map_angles
        else:
            angle_fn = z_feature_map_angles

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(angles: np.ndarray):
            # Simple hardware-efficient feature map:
            # - encode angles with RY rotations
            # - entangle with CZ chain
            for i in range(n_qubits):
                qml.RY(angles[i], wires=i)
            for _ in range(self.reps):
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                for i in range(n_qubits):
                    qml.RY(angles[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        feats = np.zeros((X.shape[0], n_qubits), dtype=float)
        for i in range(X.shape[0]):
            angles = angle_fn(X[i], n_qubits=n_qubits)
            feats[i] = np.array(circuit(angles), dtype=float)
        return feats

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._quantum_map(X)
        return self._pca.fit_transform(Z)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._quantum_map(X)
        return self._pca.transform(Z)
