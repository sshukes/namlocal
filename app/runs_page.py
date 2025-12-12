from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6 import QtWidgets

from core import runs
from app.run_details import RunDetailsDialog


class RunsPage(QtWidgets.QWidget):
    def __init__(self, runs_root: Path):
        super().__init__()
        self.runs_root = runs_root
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Run ID",
            "Name",
            "Status",
            "Start",
            "End",
            "Accelerator",
            "Best",
            "Path",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)

        btn_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        view_btn = QtWidgets.QPushButton("View")
        export_btn = QtWidgets.QPushButton("Export .nam")
        open_btn = QtWidgets.QPushButton("Open folder")
        delete_btn = QtWidgets.QPushButton("Delete")
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(view_btn)
        btn_row.addWidget(export_btn)
        btn_row.addWidget(open_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.table)
        self.setLayout(layout)

        refresh_btn.clicked.connect(self.refresh)
        view_btn.clicked.connect(self._view)
        export_btn.clicked.connect(self._export)
        open_btn.clicked.connect(self._open_folder)
        delete_btn.clicked.connect(self._delete)

    def refresh(self) -> None:
        run_rows = runs.load_runs(self.runs_root)
        self.table.setRowCount(len(run_rows))
        for row, record in enumerate(run_rows):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(record.run_id))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(record.name))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(record.status))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(record.start_time))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(record.end_time or ""))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(record.accelerator))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(record.best_metric or "")))
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(record.path))

    def _get_selected_run(self) -> Path | None:
        items = self.table.selectedItems()
        if not items:
            return None
        run_id = items[0].text()
        return self.runs_root / run_id

    def _view(self) -> None:
        run_dir = self._get_selected_run()
        if not run_dir:
            return
        dlg = RunDetailsDialog(run_dir, self)
        dlg.exec()

    def _export(self) -> None:
        run_dir = self._get_selected_run()
        if not run_dir:
            return
        model_path = run_dir / "model.nam"
        if not model_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "model.nam not found yet")
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export model", filter="NAM JSON (*.nam)")
        if not dest:
            return
        runs.export_model(run_dir, Path(dest))
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to {dest}")

    def _open_folder(self) -> None:
        run_dir = self._get_selected_run()
        if not run_dir:
            return
        if run_dir.exists():
            subprocess.Popen(["explorer", str(run_dir)])

    def _delete(self) -> None:
        run_dir = self._get_selected_run()
        if not run_dir:
            return
        confirm = QtWidgets.QMessageBox.question(self, "Delete", f"Delete {run_dir.name}? This cannot be undone.")
        if confirm == QtWidgets.QMessageBox.Yes:
            runs.remove_run(run_dir)
            self.refresh()

