from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from PySide6 import QtWidgets

from app.state import AppState


class CapturePage(QtWidgets.QWidget):
    def __init__(self, state: AppState, on_updated: Optional[Callable[[], None]] = None):
        super().__init__()
        self.state = state
        self.on_updated = on_updated
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()

        self.input_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        self.delay_spin = QtWidgets.QSpinBox()
        self.delay_spin.setRange(-48000, 48000)
        self.delay_spin.setValue(self.state.capture.delay)

        input_btn = QtWidgets.QPushButton("Select input (DI) WAV")
        output_btn = QtWidgets.QPushButton("Select output (reamp) WAV")
        auto_align_btn = QtWidgets.QPushButton("Auto-align")

        info_group = QtWidgets.QGroupBox("File info")
        info_layout = QtWidgets.QFormLayout()
        self.input_info = QtWidgets.QLabel("-")
        self.output_info = QtWidgets.QLabel("-")
        info_layout.addRow("Input:", self.input_info)
        info_layout.addRow("Output:", self.output_info)
        info_group.setLayout(info_layout)

        form = QtWidgets.QFormLayout()
        form.addRow("Input WAV", self._with_button(self.input_edit, input_btn))
        form.addRow("Output WAV", self._with_button(self.output_edit, output_btn))
        form.addRow("Delay (samples)", self.delay_spin)
        layout.addLayout(form)
        layout.addWidget(auto_align_btn)
        layout.addWidget(info_group)
        layout.addStretch()
        self.setLayout(layout)

        input_btn.clicked.connect(self._pick_input)
        output_btn.clicked.connect(self._pick_output)
        self.delay_spin.valueChanged.connect(self._on_delay_changed)
        auto_align_btn.clicked.connect(self._auto_align)

    def _with_button(self, widget: QtWidgets.QWidget, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout()
        h.addWidget(widget)
        h.addWidget(button)
        container.setLayout(h)
        return container

    def _pick_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input WAV", filter="WAV Files (*.wav)")
        if path:
            self.input_edit.setText(path)
            self.state.capture.input_path = Path(path)
            self._update_info()
            if self.on_updated:
                self.on_updated()

    def _pick_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select output WAV", filter="WAV Files (*.wav)")
        if path:
            self.output_edit.setText(path)
            self.state.capture.output_path = Path(path)
            self._update_info()
            if self.on_updated:
                self.on_updated()

    def _on_delay_changed(self, value: int) -> None:
        self.state.capture.delay = value
        if self.on_updated:
            self.on_updated()

    def _update_info(self) -> None:
        self.input_info.setText(self._read_info(self.state.capture.input_path))
        self.output_info.setText(self._read_info(self.state.capture.output_path))

    def _read_info(self, path: Optional[Path]) -> str:
        if not path:
            return "No file"
        try:
            data, sr = sf.read(path)
            shape = data.shape if hasattr(data, "shape") else (len(data),)
            return f"{sr} Hz, length {len(data)} samples, shape {shape}"
        except Exception as exc:  # pragma: no cover - UI feedback
            return f"Error: {exc}"

    def _auto_align(self) -> None:
        if not self.state.capture.input_path or not self.state.capture.output_path:
            QtWidgets.QMessageBox.warning(self, "Missing files", "Select both input and output WAV files before auto-aligning.")
            return
        try:
            x, _ = sf.read(self.state.capture.input_path)
            y, _ = sf.read(self.state.capture.output_path)
        except Exception as exc:  # pragma: no cover - UI feedback
            QtWidgets.QMessageBox.critical(self, "Read error", str(exc))
            return
        delay = self._estimate_delay(x, y)
        self.delay_spin.setValue(delay)
        self.state.capture.delay = delay
        QtWidgets.QMessageBox.information(self, "Delay estimated", f"Estimated delay: {delay} samples")
        if self.on_updated:
            self.on_updated()

    def _estimate_delay(self, x: np.ndarray, y: np.ndarray) -> int:
        x_mono = x if x.ndim == 1 else x.mean(axis=1)
        y_mono = y if y.ndim == 1 else y.mean(axis=1)
        min_len = min(len(x_mono), len(y_mono), 48000)
        if min_len <= 0:
            return 0
        x_trim = x_mono[:min_len]
        y_trim = y_mono[:min_len]
        corr = np.correlate(y_trim, x_trim, "full")
        delay = int(np.argmax(corr) - (len(x_trim) - 1))
        return delay

