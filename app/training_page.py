from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtWidgets

from app.state import AppState
from core.gpu import resolve_accelerator
from core.trainer import TrainingManager, TrainingRequest


class TrainingPage(QtWidgets.QWidget):
    training_started = QtCore.Signal(str)

    def __init__(self, state: AppState, manager: TrainingManager, on_message: Callable[[str], None]):
        super().__init__()
        self.state = state
        self.manager = manager
        self.on_message = on_message
        self.current_run_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QFormLayout()

        self.name_edit = QtWidgets.QLineEdit(self.state.training.name)
        self.arch_combo = QtWidgets.QComboBox()
        self.arch_combo.addItems(["WaveNet", "LSTM"])

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["", "standard", "lite", "feather", "nano"])

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.state.training.epochs)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 2048)
        self.batch_spin.setValue(self.state.training.batch_size)

        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setValue(self.state.training.learning_rate or 0.001)

        self.accel_combo = QtWidgets.QComboBox()
        self.accel_combo.addItems(["auto", "cpu", "gpu"])

        self.device_spin = QtWidgets.QSpinBox()
        self.device_spin.setRange(0, 8)
        self.device_spin.setValue(self.state.training.devices)

        self.output_root_edit = QtWidgets.QLineEdit(str(self.state.training.output_root))
        output_btn = QtWidgets.QPushButton("Choose output folder")

        self.start_btn = QtWidgets.QPushButton("Start training")
        self.stop_btn = QtWidgets.QPushButton("Stop active run")
        self.stop_btn.setEnabled(False)

        layout.addRow("Run name", self.name_edit)
        layout.addRow("Architecture", self.arch_combo)
        layout.addRow("Preset", self.preset_combo)
        layout.addRow("Epochs", self.epochs_spin)
        layout.addRow("Batch size", self.batch_spin)
        layout.addRow("Learning rate", self.lr_spin)
        layout.addRow("Accelerator", self.accel_combo)
        layout.addRow("GPU index", self.device_spin)
        layout.addRow("Output root", self._with_button(self.output_root_edit, output_btn))
        layout.addRow(self.start_btn)
        layout.addRow(self.stop_btn)

        self.setLayout(layout)

        output_btn.clicked.connect(self._pick_output_root)
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)

    def _with_button(self, widget: QtWidgets.QWidget, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout()
        h.addWidget(widget)
        h.addWidget(button)
        container.setLayout(h)
        return container

    def _pick_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose runs folder", str(self.state.training.output_root))
        if path:
            self.state.training.output_root = Path(path)
            self.output_root_edit.setText(path)

    def _start_training(self) -> None:
        self.state.training.name = self.name_edit.text() or "Untitled run"
        self.state.training.architecture = self.arch_combo.currentText()
        preset = self.preset_combo.currentText() or None
        self.state.training.preset = preset
        self.state.training.epochs = self.epochs_spin.value()
        self.state.training.batch_size = self.batch_spin.value()
        self.state.training.learning_rate = float(self.lr_spin.value())
        self.state.training.accelerator = self.accel_combo.currentText()
        self.state.training.devices = self.device_spin.value()
        self.state.training.output_root = Path(self.output_root_edit.text())

        resolved_accel, message = resolve_accelerator(self.state.training.accelerator)
        self.on_message(message)

        run_config = self.state.training.to_run_config(self.state.capture)
        run_config.learning.accelerator = resolved_accel
        request = TrainingRequest(
            name=self.state.training.name,
            config=run_config,
            accelerator=resolved_accel,
            devices=self.state.training.devices,
        )
        run_id = self.manager.start_training(request)
        self.current_run_id = run_id
        self.training_started.emit(run_id)
        self.on_message(f"Started run {run_id}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Training", f"Started training run {run_id}")

    def _stop_training(self) -> None:
        if self.current_run_id:
            self.manager.stop(self.current_run_id)
            self.on_message(f"Requested stop for {self.current_run_id}")
            self.stop_btn.setEnabled(False)

