# NAM Local Trainer (PySide6)

Desktop-only NAM training utility for Windows 10. The app mirrors the core flows from the previous React NAM app without a local HTTP server. Training is launched directly from the GUI using the `neural-amp-modeler` CLI (`nam-full`).

## Features
- Load DI/input and reamp/output WAV files and display basic audio info.
- Manual or auto-estimated delay (latency) control in samples.
- Training configuration: architecture (WaveNet/LSTM), preset (standard/lite/feather/nano), epochs, batch size, learning rate, accelerator (auto/CPU/GPU), GPU device index, run naming, output folder root.
- Background training via a separate `multiprocessing` process to keep the UI responsive; live log polling.
- Runs list with status, accelerator, best metric, timestamps, path, export/delete/open actions.
- Run details: stored configs (data/model/learning JSON), logs, `metrics.json`, and `model.nam` export.
- GPU detection on startup with CPU fallback messaging when GPU is unavailable.

## Project layout
```
app/       # PySide6 UI components and entry point
core/      # Training orchestration, configs, GPU detection, run helpers
runs/      # Per-run artifacts (ignored by git)
scripts/   # Helper scripts (conda environment setup)
tests/     # Minimal automated tests for non-GUI logic
```

## Windows 10 setup (Miniconda recommended)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Open an **Anaconda Prompt** and create a clean environment (Python 3.11):
   ```powershell
   conda create -y -n namlocal python=3.11
   conda activate namlocal
   ```

### CPU-only install
```powershell
pip install torch
pip install neural-amp-modeler PySide6 numpy soundfile pydantic
```

### GPU install (NVIDIA, CUDA 12.2 example)
1. Ensure a matching NVIDIA driver/CUDA runtime is installed.
2. Install a CUDA-enabled PyTorch wheel, then `neural-amp-modeler`:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu129
   pip install neural-amp-modeler PySide6 numpy soundfile pydantic
   ```
3. Verify GPU availability:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

If you see `ModuleNotFoundError: No module named 'torch'`, install PyTorch first (CPU or GPU wheel) in the active environment and rerun the check:
```powershell
pip install torch                # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu129   # GPU example
python -c "import torch; print(torch.cuda.is_available())"
```

You can also run the scripted helper:
```powershell
./scripts/env_setup.ps1 -UseGpu   # add -UseGpu for GPU, omit for CPU
```

### Using the environment from Visual Studio Code
- Open the repo folder in VS Code, then use **Ctrl+Shift+P → Python: Select Interpreter** and pick the `namlocal` conda env.
- In VS Code's integrated terminal, ensure the env is active (prompt shows `(namlocal)`), or run `conda activate namlocal` before any commands.
- Run dev commands from the repo root in that terminal:
  ```powershell
  python -m app.main            # launch the GUI
  pytest                        # run tests (optional)
  ```
- When you're ready to package, run the PyInstaller command from the same activated terminal (see below).

## Running the app (development)
```powershell
conda activate namlocal
python -m app.main
```

## Building a Windows executable (PyInstaller)
1. Install PyInstaller in the same environment:
   ```powershell
   pip install pyinstaller
   ```
2. Build a single-folder executable:
   ```powershell
   pyinstaller --name namlocal --noconfirm --windowed app/main.py
   ```
   Artifacts appear in `dist/namlocal/`. Copy the `runs/` folder alongside the executable to keep run outputs together.

## Training CLI reminder
The full training command invoked by the UI is:
```
nam-full <data.json> <model.json> <learning.json> <output_dir>
```
- `data.json` uses the `common` block with `x_path`, `y_path`, and `delay` (samples).
- `learning.json` contains `trainer.accelerator`, `trainer.devices`, `trainer.max_epochs`, and `train_dataloader.batch_size` (defaults to 16 here) plus optional `learning_rate`.
- `model.json` includes the selected `architecture` and optional preset.

## Tests
Minimal non-GUI tests cover config serialization, run folder helpers, and GPU detection.
```powershell
python -m pip install pytest
pytest
```

## Notes
- If `nam-full` is not found, the app simulates training (with placeholder metrics/model) so you can validate the UI.
- The runs directory (`runs/`) is ignored by git. Each run stores `data.json`, `model.json`, `learning.json`, `logs.txt`, `metrics.json`, optional `checkpoints/`, and `model.nam` when exported.
- Use the runs page to open the run folder in Explorer, export `.nam`, or delete runs. Active runs can be cancelled from the status bar by stopping the app or closing the window.

