from __future__ import annotations

import pathlib
import typer
import yaml

from qabda.config import PipelineConfig
from qabda.logging_utils import setup_logging
from qabda.pipeline import run_local

app = typer.Typer(add_completion=False)


@app.command()
def run(config: str = typer.Option(..., help="Path to YAML config"), log_level: str = "INFO"):
    """Run the QABDA pipeline."""
    setup_logging(log_level)
    p = pathlib.Path(config)
    if not p.exists():
        raise typer.BadParameter(f"Config not found: {config}")

    cfg_dict = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = PipelineConfig(**cfg_dict)

    if cfg.engine == "local":
        res = run_local(cfg)
        typer.echo(f"Done. Metrics: {res.metrics}")
        typer.echo(f"Timings: {res.timings}")
    else:
        raise typer.BadParameter("Spark mode scaffold exists, but run_spark() is not wired in this demo.")


if __name__ == "__main__":
    app()
