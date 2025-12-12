from __future__ import annotations

import importlib.util
from typing import Tuple


def detect_gpu() -> Tuple[bool, str]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False, "torch not installed"
    torch = importlib.import_module("torch")
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"torch error: {exc}"
    return available, "cuda available" if available else "cuda not available"


def resolve_accelerator(requested: str) -> Tuple[str, str]:
    requested_lower = requested.lower()
    has_gpu, message = detect_gpu()
    if requested_lower == "gpu":
        if has_gpu:
            return "gpu", "Using GPU"
        return "cpu", "GPU requested but not available; falling back to CPU"
    if requested_lower == "cpu":
        return "cpu", "Using CPU"
    if has_gpu:
        return "gpu", "Auto-selected GPU"
    return "cpu", "Auto-selected CPU"

