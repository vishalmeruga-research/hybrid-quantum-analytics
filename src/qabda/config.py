from __future__ import annotations

from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    path: str
    target_col: str
    id_cols: List[str] = Field(default_factory=list)


class PreprocessingConfig(BaseModel):
    normalize: Optional[Literal["standard", "minmax"]] = "standard"
    max_features: Optional[int] = None


class ReducerConfig(BaseModel):
    name: Literal["classical_pca", "qpca_surrogate"]
    n_components: int = 8
    # qpca_surrogate parameters:
    feature_map: Optional[Literal["zz_feature_map", "z_feature_map"]] = "zz_feature_map"
    reps: int = 2


class RFConfig(BaseModel):
    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 7


class SVMConfig(BaseModel):
    c: float = 1.0
    kernel: Literal["rbf", "linear"] = "rbf"
    gamma: Optional[str] = "scale"


class VQCInnerConfig(BaseModel):
    backend: Literal["pennylane_default_qubit", "qiskit_aer"] = "pennylane_default_qubit"
    n_qubits: int = 8
    layers: int = 2
    lr: float = 0.2
    epochs: int = 25
    batch_size: int = 32


class ModelConfig(BaseModel):
    name: Literal["rf", "svm", "vqc"]
    rf: Optional[RFConfig] = None
    svm: Optional[SVMConfig] = None
    vqc: Optional[VQCInnerConfig] = None


class SparkConfig(BaseModel):
    app_name: str = "qabda"
    master: str = "local[*]"
    config: Dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    dir: str = "artifacts"


class PipelineConfig(BaseModel):
    engine: Literal["local", "spark"] = "local"
    seed: int = 7

    spark: Optional[SparkConfig] = None
    input: InputConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    reducer: ReducerConfig
    model: ModelConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
