from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from qabda.artifacts import ensure_dir, save_json
from qabda.config import PipelineConfig
from qabda.data_io import load_local
from qabda.metrics import classification_metrics
from qabda.preprocessing import Preprocessor
from qabda.reducers.classical_pca import ClassicalPCA
from qabda.reducers.qpca_surrogate import QPCASurrogate
from qabda.models.classical import RFModel, SVMModel
from qabda.quantum.vqc import VariationalQuantumClassifier, VQCConfig


@dataclass
class RunResult:
    metrics: Dict[str, float]
    timings: Dict[str, float]


def _build_reducer(cfg: PipelineConfig):
    if cfg.reducer.name == "classical_pca":
        return ClassicalPCA(n_components=cfg.reducer.n_components, random_state=cfg.seed)
    if cfg.reducer.name == "qpca_surrogate":
        return QPCASurrogate(
            n_components=cfg.reducer.n_components,
            feature_map=cfg.reducer.feature_map or "zz_feature_map",
            reps=cfg.reducer.reps,
            random_state=cfg.seed,
        )
    raise ValueError(f"Unknown reducer: {cfg.reducer.name}")


def _build_model(cfg: PipelineConfig):
    if cfg.model.name == "rf":
        mcfg = cfg.model.rf or RFModel(random_state=cfg.seed)
        return mcfg.build(), "sklearn"
    if cfg.model.name == "svm":
        mcfg = cfg.model.svm or SVMModel()
        return mcfg.build(), "sklearn"
    if cfg.model.name == "vqc":
        vcfg0 = cfg.model.vqc or VQCConfig(seed=cfg.seed)
        vcfg = VQCConfig(**vcfg0.model_dump(), seed=cfg.seed)  # type: ignore[attr-defined]
        return VariationalQuantumClassifier(vcfg), "vqc"
    raise ValueError(f"Unknown model: {cfg.model.name}")


def run_local(cfg: PipelineConfig) -> RunResult:
    """Run the end-to-end pipeline in local mode.

    This is intentionally written as straightforward, testable code.
    For production, you can:
    - swap `train_test_split` for time-based splits
    - add experiment tracking (MLflow, W&B)
    - write reducers/models as Spark ML transformers
    """
    out_dir = ensure_dir(cfg.output.dir)
    t0 = time.time()

    ds = load_local(cfg.input.path, cfg.input.target_col, cfg.input.id_cols)
    t_load = time.time()

    pre = Preprocessor(normalize=cfg.preprocessing.normalize, max_features=cfg.preprocessing.max_features)
    X = pre.fit_transform(ds.X)
    y = ds.y.values.astype(int)
    t_prep = time.time()

    reducer = _build_reducer(cfg)
    Xr = reducer.fit_transform(X)
    t_red = time.time()

    X_train, X_test, y_train, y_test = train_test_split(Xr, y, test_size=0.25, random_state=cfg.seed, stratify=y)
    model, kind = _build_model(cfg)

    if kind == "sklearn":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = getattr(model, "predict_proba", None)
        proba = y_proba(X_test)[:, 1] if y_proba is not None else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    t_fit = time.time()

    mets = classification_metrics(y_test, y_pred, proba)
    timings = {
        "load_s": t_load - t0,
        "preprocess_s": t_prep - t_load,
        "reduce_s": t_red - t_prep,
        "fit_predict_s": t_fit - t_red,
        "total_s": t_fit - t0,
    }

    save_json({"metrics": mets, "timings": timings, "config": cfg.model_dump()}, f"{out_dir}/run_report.json")
    return RunResult(metrics=mets, timings=timings)
