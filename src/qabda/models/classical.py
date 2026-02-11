from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


@dataclass
class RFModel:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 7

    def build(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )


@dataclass
class SVMModel:
    c: float = 1.0
    kernel: str = "rbf"
    gamma: Optional[str] = "scale"

    def build(self) -> SVC:
        return SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, probability=True)
