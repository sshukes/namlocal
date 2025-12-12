from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid


RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class RunRecord:
    run_id: str
    name: str
    status: str
    accelerator: str
    start_time: str
    end_time: Optional[str] = None
    best_metric: Optional[float] = None
    has_exported_nam: bool = False
    path: str = ""

    @classmethod
    def from_paths(cls, run_dir: Path) -> "RunRecord":
        metadata_path = run_dir / "metrics.json"
        metrics = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fp:
                metrics = json.load(fp)
        best_metric = metrics.get("best") if isinstance(metrics, dict) else None
        status = "finished" if (run_dir / "model.nam").exists() else "running"
        if metrics.get("status"):
            status = metrics.get("status")
        start_time = metrics.get("start_time", "")
        end_time = metrics.get("end_time") if metrics else None
        has_exported_nam = (run_dir / "model.nam").exists()
        return cls(
            run_id=run_dir.name,
            name=(run_dir / "run.json").read_text(encoding="utf-8") if (run_dir / "run.json").exists() else run_dir.name,
            status=status,
            accelerator=metrics.get("accelerator", ""),
            start_time=start_time,
            end_time=end_time,
            best_metric=best_metric,
            has_exported_nam=has_exported_nam,
            path=str(run_dir),
        )


def generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def create_run_directory(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def save_run_name(run_dir: Path, name: str) -> None:
    (run_dir / "run.json").write_text(name, encoding="utf-8")


def save_metrics(run_dir: Path, metrics: Dict[str, object]) -> Path:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics_path


def load_runs(root: Path = RUNS_ROOT) -> List[RunRecord]:
    if not root.exists():
        return []
    runs: List[RunRecord] = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir():
            runs.append(RunRecord.from_paths(run_dir))
    return runs


def export_model(run_dir: Path, destination: Path) -> Path:
    model_path = run_dir / "model.nam"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = model_path.read_bytes()
    destination.write_bytes(data)
    return destination


def remove_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for child in run_dir.rglob("*"):
        if child.is_file():
            child.unlink()
    for child in sorted(run_dir.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
    run_dir.rmdir()

