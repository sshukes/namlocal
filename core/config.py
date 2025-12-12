from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class DataConfig:
    x_path: str
    y_path: str
    delay: int = 0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "common": {
                "x_path": self.x_path,
                "y_path": self.y_path,
                "delay": self.delay,
            }
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_payload(), fp, indent=2)
        return path


@dataclass
class ModelConfig:
    architecture: str = "WaveNet"
    preset: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {"architecture": self.architecture}
        if self.preset:
            payload["preset"] = self.preset
        payload.update(self.extra)
        return payload

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_payload(), fp, indent=2)
        return path


@dataclass
class LearningConfig:
    accelerator: str = "auto"
    devices: int | str = 1
    max_epochs: int = 5
    batch_size: int = 16
    learning_rate: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "trainer": {
                "accelerator": self.accelerator,
                "devices": self.devices,
                "max_epochs": self.max_epochs,
            },
            "train_dataloader": {"batch_size": self.batch_size},
        }
        if self.learning_rate is not None:
            payload["learning_rate"] = self.learning_rate
        payload.update(self.extra)
        return payload

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_payload(), fp, indent=2)
        return path


@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    learning: LearningConfig

    def save_all(self, run_dir: Path) -> Dict[str, Path]:
        paths = {
            "data": run_dir / "data.json",
            "model": run_dir / "model.json",
            "learning": run_dir / "learning.json",
        }
        self.data.save(paths["data"])
        self.model.save(paths["model"])
        self.learning.save(paths["learning"])
        return paths

    def as_command_args(self, run_dir: Path) -> list[str]:
        paths = self.save_all(run_dir)
        return [str(paths["data"]), str(paths["model"]), str(paths["learning"]), str(run_dir)]

