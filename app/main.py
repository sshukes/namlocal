from __future__ import annotations

import sys
import multiprocessing as mp

from PySide6 import QtCore, QtWidgets

from app.capture_page import CapturePage
from app.runs_page import RunsPage
from app.state import AppState
from app.training_page import TrainingPage
from core.trainer import TrainingManager
from core.gpu import detect_gpu


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NAM Trainer Local")
        self.state = AppState()
        self.manager = TrainingManager(self.state.training.output_root)
        self._build_ui()
        self._setup_timers()
        self._show_gpu_status()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        nav = QtWidgets.QListWidget()
        nav.addItems(["Capture / Files", "Training Config", "Runs"])
        nav.setFixedWidth(160)

        self.stack = QtWidgets.QStackedWidget()
        self.capture_page = CapturePage(self.state, on_updated=self._on_state_update)
        self.training_page = TrainingPage(self.state, self.manager, self._show_message)
        self.runs_page = RunsPage(self.state.training.output_root)
        self.training_page.training_started.connect(lambda _: self.runs_page.refresh())

        self.stack.addWidget(self.capture_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.runs_page)

        layout.addWidget(nav)
        layout.addWidget(self.stack)
        central.setLayout(layout)
        self.setCentralWidget(central)

        nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        nav.setCurrentRow(0)

        self.status = self.statusBar()

    def _on_state_update(self) -> None:
        self._show_message("Capture info updated")

    def _setup_timers(self) -> None:
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._poll_logs)
        self.timer.start(500)

    def _poll_logs(self) -> None:
        updates = self.manager.poll_logs()
        for run_id, lines in updates.items():
            joined = "\n".join(lines)
            self._show_message(f"{run_id}: {joined}")
        if not self.manager.active:
            self.training_page.start_btn.setEnabled(True)
            self.training_page.stop_btn.setEnabled(False)
            self.runs_page.refresh()

    def _show_message(self, message: str) -> None:
        self.status.showMessage(message, 5000)

    def _show_gpu_status(self) -> None:
        has_gpu, msg = detect_gpu()
        alert = "GPU available" if has_gpu else "GPU not available"
        self._show_message(f"Startup GPU check: {alert} ({msg})")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()

