from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6 import QtWidgets

from core import runs


class RunDetailsDialog(QtWidgets.QDialog):
    def __init__(self, run_dir: Path, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.run_dir = run_dir
        self.setWindowTitle(f"Run {run_dir.name}")
        self.resize(700, 500)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()

        self.config_view = QtWidgets.QTextEdit()
        self.config_view.setReadOnly(True)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.metrics_label = QtWidgets.QLabel("")
        export_btn = QtWidgets.QPushButton("Export .nam")

        layout.addWidget(QtWidgets.QLabel("Configs (data/model/learning):"))
        layout.addWidget(self.config_view)
        layout.addWidget(QtWidgets.QLabel("Logs:"))
        layout.addWidget(self.log_view)
        layout.addWidget(QtWidgets.QLabel("Metrics:"))
        layout.addWidget(self.metrics_label)
        layout.addWidget(export_btn)

        self.setLayout(layout)
        export_btn.clicked.connect(self._export)

    def _load(self) -> None:
        configs = []
        for name in ["data.json", "model.json", "learning.json"]:
            cfg_path = self.run_dir / name
            if cfg_path.exists():
                configs.append(f"== {name} ==\n{cfg_path.read_text(encoding='utf-8')}\n")
        self.config_view.setText("\n".join(configs))

        log_path = self.run_dir / "logs.txt"
        if log_path.exists():
            self.log_view.setText(log_path.read_text(encoding="utf-8"))
        metrics_path = self.run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary = []
            if "best" in metrics:
                summary.append(f"Best: {metrics['best']}")
            if "epochs" in metrics:
                summary.append(f"Epochs: {metrics['epochs']}")
            self.metrics_label.setText(" | ".join(summary))

    def _export(self) -> None:
        model_path = self.run_dir / "model.nam"
        if not model_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "model.nam not found yet")
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export model", filter="NAM JSON (*.nam)")
        if dest:
            runs.export_model(self.run_dir, Path(dest))
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved to {dest}")

