from pathlib import Path

from core import runs


def test_create_and_remove_run(tmp_path: Path) -> None:
    run_id = "test-run"
    run_dir = runs.create_run_directory(tmp_path, run_id)
    assert run_dir.exists()
    runs.save_run_name(run_dir, "My Run")
    runs.save_metrics(run_dir, {"status": "finished", "best": 0.1})

    loaded = runs.load_runs(tmp_path)
    assert loaded[0].run_id == run_id
    assert loaded[0].has_exported_nam is False

    (run_dir / "model.nam").write_text("model", encoding="utf-8")
    destination = tmp_path / "copy.nam"
    runs.export_model(run_dir, destination)
    assert destination.exists()

    runs.remove_run(run_dir)
    assert not run_dir.exists()

