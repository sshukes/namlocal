from core.gpu import detect_gpu, resolve_accelerator


def test_detect_gpu_returns_tuple():
    available, msg = detect_gpu()
    assert isinstance(available, bool)
    assert isinstance(msg, str)


def test_resolve_accelerator_auto():
    resolved, note = resolve_accelerator("auto")
    assert resolved in {"cpu", "gpu"}
    assert isinstance(note, str)

