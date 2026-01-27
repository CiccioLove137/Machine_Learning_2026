from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

RUN_RE = re.compile(r"^run_(\d{3})$")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_next_run_dir(runs_root: Path = Path("reports/runs")) -> Path:
    runs_root = ensure_dir(runs_root)

    ids = []
    for d in runs_root.iterdir():
        if d.is_dir():
            m = RUN_RE.match(d.name)
            if m:
                ids.append(int(m.group(1)))

    next_id = (max(ids) + 1) if ids else 1
    run_dir = runs_root / f"run_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_run_dir(run_name: str, runs_root: Path = Path("reports/runs")) -> Path:
    run_dir = runs_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run non trovata: {run_dir}")
    return run_dir


def get_latest_run_dir(runs_root: Path = Path("reports/runs")) -> Path:
    runs_root = ensure_dir(runs_root)

    latest: Optional[Path] = None
    latest_id = -1

    for d in runs_root.iterdir():
        if d.is_dir():
            m = RUN_RE.match(d.name)
            if m:
                rid = int(m.group(1))
                if rid > latest_id:
                    latest_id = rid
                    latest = d

    if latest is None:
        raise FileNotFoundError("Nessuna run trovata in reports/runs. Esegui prima train-rnn.")
    return latest


def copy_config_to_run(config_path: str, run_dir: Path) -> Path:
    src = Path(config_path)
    if not src.exists():
        raise FileNotFoundError(f"Config non trovata: {src}")
    dst = run_dir / "config.yaml"
    shutil.copy2(src, dst)
    return dst
