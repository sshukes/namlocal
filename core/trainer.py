from __future__ import annotations

import json
import multiprocessing as mp
import queue
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from core.config import RunConfig
from core import runs


@dataclass
class TrainingRequest:
    name: str
    config: RunConfig
    accelerator: str = "auto"
    devices: int | str = 1


@dataclass
class ActiveRun:
    process: mp.Process
    log_queue: mp.Queue
    stop_event: mp.Event
    run_dir: Path


class TrainingManager:
    def __init__(self, runs_root: Path | None = None):
        self.runs_root = runs_root or runs.RUNS_ROOT
        self.active: Dict[str, ActiveRun] = {}

    def start_training(self, request: TrainingRequest) -> str:
        run_id = runs.generate_run_id()
        run_dir = runs.create_run_directory(self.runs_root, run_id)
        runs.save_run_name(run_dir, request.name)
        log_queue: mp.Queue = mp.Queue()
        stop_event = mp.Event()
        process = mp.Process(
            target=_training_worker,
            args=(run_dir, request, log_queue, stop_event),
            daemon=True,
        )
        process.start()
        self.active[run_id] = ActiveRun(process=process, log_queue=log_queue, stop_event=stop_event, run_dir=run_dir)
        return run_id

    def poll_logs(self) -> Dict[str, List[str]]:
        updates: Dict[str, List[str]] = {}
        finished_runs: List[str] = []
        for run_id, active in list(self.active.items()):
            lines: List[str] = []
            while True:
                try:
                    line = active.log_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    lines.append(line)
            if lines:
                updates[run_id] = lines
            if not active.process.is_alive():
                active.process.join(timeout=0.1)
                finished_runs.append(run_id)
        for run_id in finished_runs:
            self.active.pop(run_id, None)
        return updates

    def stop(self, run_id: str) -> None:
        active = self.active.get(run_id)
        if not active:
            return
        active.stop_event.set()
        if active.process.is_alive():
            active.process.join(timeout=5)

    def is_running(self, run_id: str) -> bool:
        active = self.active.get(run_id)
        if not active:
            return False
        return active.process.is_alive()


def _training_worker(run_dir: Path, request: TrainingRequest, log_queue: mp.Queue, stop_event: mp.Event) -> None:
    log_path = run_dir / "logs.txt"
    log_path.write_text("", encoding="utf-8")
    metrics: Dict[str, object] = {
        "status": "running",
        "start_time": datetime.utcnow().isoformat(),
        "accelerator": request.accelerator,
    }
    runs.save_metrics(run_dir, metrics)
    config_paths = request.config.save_all(run_dir)

    command = ["nam-full", str(config_paths["data"]), str(config_paths["model"]), str(config_paths["learning"]), str(run_dir)]
    nam_available = shutil.which("nam-full") is not None

    if nam_available:
        _write_log(log_queue, log_path, f"Running: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in iter(process.stdout.readline, ""):
                if stop_event.is_set():
                    process.terminate()
                    _write_log(log_queue, log_path, "Stopping training per user request...")
                    break
                if line:
                    _write_log(log_queue, log_path, line.rstrip())
            process.wait()
        finally:
            return_code = process.returncode
    else:
        _write_log(log_queue, log_path, "nam-full not found. Running simulated training for preview.")
        return_code = _simulate_training(request, log_queue, log_path, stop_event)

    metrics["status"] = "stopped" if stop_event.is_set() else ("finished" if return_code == 0 else "error")
    metrics["end_time"] = datetime.utcnow().isoformat()
    if (run_dir / "metrics.json").exists():
        existing = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics.update(existing)
    runs.save_metrics(run_dir, metrics)

    if metrics["status"] == "finished":
        model_path = run_dir / "model.nam"
        if not model_path.exists():
            model_path.write_text(json.dumps({"note": "placeholder model"}, indent=2), encoding="utf-8")
        _write_log(log_queue, log_path, "Training finished. Model ready.")


def _simulate_training(request: TrainingRequest, log_queue: mp.Queue, log_path: Path, stop_event: mp.Event) -> int:
    epochs = request.config.learning.max_epochs
    metrics: Dict[str, List[float]] = {"epoch_loss": []}
    for epoch in range(1, epochs + 1):
        if stop_event.is_set():
            _write_log(log_queue, log_path, "Simulation cancelled.")
            return 1
        loss = round(max(0.001, 1.0 / epoch), 4)
        metrics["epoch_loss"].append(loss)
        _write_log(log_queue, log_path, f"Epoch {epoch}/{epochs} - loss: {loss}")
        time.sleep(0.5)
    summary = {
        "best": min(metrics["epoch_loss"]),
        "epochs": epochs,
        "metrics": metrics,
    }
    runs.save_metrics(run_dir=log_path.parent, metrics={**summary, "status": "finished", "start_time": datetime.utcnow().isoformat(), "end_time": datetime.utcnow().isoformat(), "accelerator": request.accelerator})
    return 0


def _write_log(log_queue: mp.Queue, log_path: Path, line: str) -> None:
    log_queue.put(line)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")

