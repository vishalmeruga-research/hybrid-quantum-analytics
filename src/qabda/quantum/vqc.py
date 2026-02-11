from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VQCConfig:
    """Configuration for a simple Variational Quantum Classifier (binary).

    This matches the paper’s "Variational Quantum Optimization" / VQA loop concept.
    In a real system you will want:
    - better optimizers (Adam, SPSA)
    - batching on device
    - regularization, early stopping
    - multi-class strategies
    """

    backend: str = "pennylane_default_qubit"
    n_qubits: int = 8
    layers: int = 2
    lr: float = 0.2
    epochs: int = 25
    batch_size: int = 32
    seed: int = 7


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class VariationalQuantumClassifier:
    """A small, educational VQC using PennyLane.

    We use angle encoding into RY rotations, then a hardware-efficient ansatz.

    Output:
    - expectation value on wire 0 (PauliZ) mapped to probability.

    Notes
    -----
    This implementation is intentionally simple to keep it maintainable.
    """

    def __init__(self, cfg: VQCConfig):
        self.cfg = cfg
        self.weights: np.ndarray | None = None

    def _build_circuit(self):
        try:
            import pennylane as qml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("pennylane is required for VQC. Install with: pip install -e '.[pennylane]'") from e

        n = self.cfg.n_qubits
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circuit(x: np.ndarray, w: np.ndarray):
            # Encode
            for i in range(n):
                qml.RY(x[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.cfg.layers):
                for i in range(n):
                    qml.RY(w[idx], wires=i)
                    idx += 1
                for i in range(n - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measure wire 0
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.n_qubits
        # squash features into angles
        Xang = np.pi * np.tanh(X[:, :n])

        circuit = self._build_circuit()
        # weights per layer per qubit (RY), plus entanglers (no params)
        n_params = self.cfg.layers * n
        self.weights = rng.normal(scale=0.1, size=(n_params,))

        def predict_proba_batch(Xb: np.ndarray, w: np.ndarray) -> np.ndarray:
            # map expval in [-1, 1] to probability
            out = np.array([circuit(x, w) for x in Xb], dtype=float)
            return (out + 1.0) / 2.0

        # simple SGD on log loss
        bs = self.cfg.batch_size
        for epoch in range(self.cfg.epochs):
            idxs = rng.permutation(Xang.shape[0])
            for start in range(0, Xang.shape[0], bs):
                batch_idx = idxs[start : start + bs]
                Xb = Xang[batch_idx]
                yb = y[batch_idx]

                # finite-difference gradient (simple, slow; replace with parameter-shift for real work)
                eps = 1e-2
                base_p = predict_proba_batch(Xb, self.weights)
                base_p = np.clip(base_p, 1e-6, 1 - 1e-6)
                base_loss = -np.mean(yb * np.log(base_p) + (1 - yb) * np.log(1 - base_p))

                grad = np.zeros_like(self.weights)
                for j in range(self.weights.size):
                    w2 = self.weights.copy()
                    w2[j] += eps
                    p2 = predict_proba_batch(Xb, w2)
                    p2 = np.clip(p2, 1e-6, 1 - 1e-6)
                    loss2 = -np.mean(yb * np.log(p2) + (1 - yb) * np.log(1 - p2))
                    grad[j] = (loss2 - base_loss) / eps

                self.weights -= self.cfg.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("VQC not fitted.")
        circuit = self._build_circuit()
        n = self.cfg.n_qubits
        Xang = np.pi * np.tanh(X[:, :n])
        out = np.array([circuit(x, self.weights) for x in Xang], dtype=float)
        p = (out + 1.0) / 2.0
        return np.vstack([1 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
