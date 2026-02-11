from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class Preprocessor:
    """Local-mode preprocessing for tabular data.

    This mirrors the paper’s “Data ingestion & preprocessing” and “Classical big data processing”
    layers (just without requiring Spark).

    Notes
    -----
    - For ultra-high-dimensional data, you will often do feature pruning upstream
      (e.g., remove near-constant columns, use hashing trick, enforce schema).
    - `max_features` here is a *demo safety valve* to prevent exploding memory.
    """

    normalize: Optional[str] = "standard"
    max_features: Optional[int] = None

    scaler_: Optional[object] = None
    selector_: Optional[VarianceThreshold] = None
    kept_columns_: Optional[list[str]] = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X2 = X.copy()

        # 1) remove non-numeric columns (you can extend with encoders)
        numeric_cols = [c for c in X2.columns if pd.api.types.is_numeric_dtype(X2[c])]
        X2 = X2[numeric_cols].fillna(0.0)

        # 2) variance filter (cheap feature selection)
        self.selector_ = VarianceThreshold(threshold=0.0)
        arr = self.selector_.fit_transform(X2.values)
        kept = [c for c, k in zip(numeric_cols, self.selector_.get_support()) if k]
        self.kept_columns_ = kept

        # 3) optional cap to avoid 'oops we loaded 2M columns into pandas'
        if self.max_features is not None and arr.shape[1] > self.max_features:
            arr = arr[:, : self.max_features]
            self.kept_columns_ = self.kept_columns_[: self.max_features]

        # 4) normalization
        if self.normalize == "standard":
            self.scaler_ = StandardScaler()
        elif self.normalize == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.normalize is None:
            self.scaler_ = None
            return arr.astype(float)

        return self.scaler_.fit_transform(arr).astype(float)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.kept_columns_ is None:
            raise RuntimeError("Preprocessor not fitted.")

        X2 = X[self.kept_columns_].fillna(0.0).astype(float).values
        if self.scaler_ is None:
            return X2
        return self.scaler_.transform(X2)
