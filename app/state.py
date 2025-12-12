from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.config import DataConfig, LearningConfig, ModelConfig, RunConfig


@dataclass
class CaptureState:
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    delay: int = 0

    def to_data_config(self) -> DataConfig:
        return DataConfig(
            x_path=str(self.input_path) if self.input_path else "",
            y_path=str(self.output_path) if self.output_path else "",
            delay=self.delay,
        )


@dataclass
class TrainingState:
    name: str = ""
    architecture: str = "WaveNet"
    preset: Optional[str] = None
    epochs: int = 5
    batch_size: int = 16
    learning_rate: Optional[float] = None
    accelerator: str = "auto"
    devices: int = 1
    output_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "runs")

    def to_run_config(self, capture: CaptureState) -> RunConfig:
        model = ModelConfig(architecture=self.architecture, preset=self.preset)
        learning = LearningConfig(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        data = capture.to_data_config()
        return RunConfig(data=data, model=model, learning=learning)


@dataclass
class AppState:
    capture: CaptureState = field(default_factory=CaptureState)
    training: TrainingState = field(default_factory=TrainingState)

