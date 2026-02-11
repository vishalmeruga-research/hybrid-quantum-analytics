import json
from pathlib import Path

import yaml

from qabda.config import PipelineConfig
from qabda.pipeline import run_local


def test_local_pipeline_smoke(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "examples" / "config_local.yaml"
    cfg_dict = yaml.safe_load(cfg_path.read_text())
    cfg_dict["output"]["dir"] = str(tmp_path / "artifacts")
    cfg = PipelineConfig(**cfg_dict)

    res = run_local(cfg)
    assert 0.0 <= res.metrics["accuracy"] <= 1.0

    report = Path(cfg.output.dir) / "run_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert "metrics" in data and "timings" in data
