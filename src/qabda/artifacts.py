from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(str(pathlib.Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
