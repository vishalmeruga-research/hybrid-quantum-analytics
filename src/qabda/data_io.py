from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional, List

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """A simple container for features and labels.

    For real ultra-high-dimensional workloads you will likely keep `X` on Spark or in
    an Arrow-backed store. This abstraction intentionally stays minimal.
    """

    X: pd.DataFrame
    y: pd.Series
    id_cols: Optional[pd.DataFrame] = None


def load_local(path: str, target_col: str, id_cols: Optional[List[str]] = None) -> Dataset:
    """Load a dataset from CSV or Parquet into pandas.

    Parameters
    ----------
    path:
        File path (CSV or Parquet).
    target_col:
        Name of the label column.
    id_cols:
        Optional list of ID columns to carry along to outputs.

    Returns
    -------
    Dataset
    """
    id_cols = id_cols or []
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        raise ValueError("Only .csv and .parquet are supported for local mode.")

    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not in columns: {list(df.columns)[:20]}...")

    ids = df[id_cols].copy() if id_cols else None
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col] + id_cols, errors="ignore")
    return Dataset(X=X, y=y, id_cols=ids)


def load_spark(path: str, target_col: str, id_cols: Optional[List[str]] = None):
    """Load a dataset using Spark DataFrame.

    This function is optional and only works when `pyspark` is installed.
    We keep it separate to avoid forcing Spark dependencies on all users.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pyspark is required for spark mode. Install with: pip install -e '.[spark]'") from e

    id_cols = id_cols or []
    spark = SparkSession.builder.getOrCreate()
    if path.lower().endswith(".csv"):
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    else:
        df = spark.read.parquet(path)

    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not in Spark columns.")

    return df, id_cols
