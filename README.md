# QABDA — Quantum-Assisted Big Data Analytics

This repository is a **production-oriented reference implementation** of the *Quantum-Assisted Big Data Analytics (QABDA)* framework described in the paper **“Quantum-Assisted Big Data Analytics for Ultra-High-Dimensional Datasets”**.  
It provides a modular pipeline that mirrors the paper’s five layers:

1. **Data ingestion & preprocessing**
2. **Classical big-data processing (Spark-compatible)**
3. **Quantum-assisted analytics modules** (feature mapping, “qPCA”-style reduction, VQA/QAOA-style optimization)
4. **Hybrid integration middleware**
5. **Application layer** (classification / clustering outputs)

> Important reality check: the paper presents a *framework concept* and experiment-style results.  
> On today’s NISQ hardware, “true” qPCA via density-matrix exponentiation + QPE is not practical for large `d`.
> This repo therefore implements **NISQ-friendly and production-friendly surrogates** that preserve the *architecture*:
>
> - *Quantum feature mapping* (Qiskit / PennyLane)
> - Variational classifiers (VQC) and QAOA-like loops
> - A `qPCA` **interface** with a **simulated / approximate backend**, plus a classical PCA baseline
>
> This gives you runnable code, unit tests, configuration, a CLI, and clean integration boundaries — i.e., something you can actually ship and iterate.

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install
Minimal (no quantum, no spark):
```bash
pip install -e .
```

With Spark:
```bash
pip install -e ".[spark]"
```

With Qiskit:
```bash
pip install -e ".[qiskit]"
```

With PennyLane:
```bash
pip install -e ".[pennylane]"
```

Dev setup:
```bash
pip install -e ".[dev,spark,qiskit,pennylane]"
pre-commit install
```

## Run an example (local, scikit-learn)

```bash
qabda run --config examples/config_local.yaml
```

This will:
- load a dataset (CSV / Parquet)
- normalize + optional feature pruning
- apply dimensionality reduction (classical PCA or qPCA surrogate)
- run either a classical classifier (SVM/RF) or a variational quantum classifier (VQC)
- write metrics + artifacts to `./artifacts/`

## Run on Spark (optional)

If you have a Spark environment:
```bash
qabda run --config examples/config_spark.yaml
```

## Project layout

```
qabda_repo/
  src/qabda/
    cli.py                 # CLI entrypoint
    config.py              # Pydantic config models
    logging_utils.py       # structured logging helper
    pipeline.py            # orchestration (hybrid integration layer)
    data_io.py             # ingestion (pandas/spark)
    preprocessing.py       # normalization, feature pruning
    reducers/
      base.py              # interface for dimensionality reduction
      classical_pca.py     # baseline PCA
      qpca_surrogate.py    # NISQ-friendly qPCA-like reducer
    quantum/
      feature_maps.py      # amplitude/angle encoding helpers + feature maps
      vqc.py               # variational quantum classifier
      qaoa_opt.py          # QAOA-like optimizer loop (demo)
    models/
      classical.py         # RF/SVM wrappers
    metrics.py             # metrics + reports
    artifacts.py           # model + report saving
  tests/
  examples/
  .github/workflows/
```

## Configuration

The pipeline is configured via YAML. See `examples/config_local.yaml`.

Key knobs:
- `engine`: `local` or `spark`
- `reducer`: `classical_pca` or `qpca_surrogate`
- `model`: `rf`, `svm`, or `vqc`
- `quantum_backend`: `qiskit_aer` or `pennylane_default_qubit`

## Where you should challenge the paper (and your own assumptions)

- **Amplitude encoding** is not “free”. Preparing an arbitrary amplitude state can be expensive.
- Claimed **polylog** scaling for qPCA relies on assumptions (efficient state prep, access model/QRAM, well-conditioned matrices).
- Many advantages are **problem-dependent**; the repo isolates quantum modules so you can benchmark honestly.

If you want this repo to be *truly* aligned to a real production target (e.g., Spark on EMR + IBM Runtime),
say the word and I’ll refactor the interfaces to match that deployment model.

## Citation

If you use this code in academic work, cite the paper (and this repo).

## License

MIT — do what you want, just don’t blame the repo when a NISQ chip ruins your accuracy. 🙂
