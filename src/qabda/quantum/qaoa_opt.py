from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QAOAConfig:
    """Toy QAOA-like optimizer loop.

    In the paper, QAOA is described as a variational optimization method:
        min_theta <psi(theta) | H_C | psi(theta)>

    Implementing full QAOA for clustering is non-trivial and strongly problem-specific.
    This module provides a small scaffold you can extend.

    Use-cases:
    - teaching/demo
    - a placeholder boundary in your architecture so you can slot in real QAOA later
    """

    n_params: int = 8
    lr: float = 0.1
    steps: int = 100
    seed: int = 7


def qaoa_like_optimize(cost_fn, cfg: QAOAConfig) -> tuple[np.ndarray, float]:
    """Minimize `cost_fn(theta)` using a naive finite-difference gradient.

    Parameters
    ----------
    cost_fn:
        Callable(theta)->float (your expected Hamiltonian / objective).
    cfg:
        Optimization hyperparameters.

    Returns
    -------
    theta_best, best_cost
    """
    rng = np.random.default_rng(cfg.seed)
    theta = rng.normal(scale=0.2, size=(cfg.n_params,))
    best_theta = theta.copy()
    best_cost = float(cost_fn(theta))

    eps = 1e-3
    for _ in range(cfg.steps):
        base = float(cost_fn(theta))
        grad = np.zeros_like(theta)
        for j in range(theta.size):
            t2 = theta.copy()
            t2[j] += eps
            grad[j] = (float(cost_fn(t2)) - base) / eps
        theta = theta - cfg.lr * grad

        c = float(cost_fn(theta))
        if c < best_cost:
            best_cost, best_theta = c, theta.copy()

    return best_theta, best_cost
