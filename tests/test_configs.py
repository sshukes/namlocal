from pathlib import Path

from core.config import DataConfig, LearningConfig, ModelConfig, RunConfig


def test_config_serialization(tmp_path: Path) -> None:
    data = DataConfig(x_path="input.wav", y_path="output.wav", delay=12)
    model = ModelConfig(architecture="WaveNet", preset="lite")
    learning = LearningConfig(accelerator="cpu", devices=1, max_epochs=2, batch_size=4, learning_rate=0.01)
    run = RunConfig(data=data, model=model, learning=learning)

    paths = run.save_all(tmp_path)
    assert (tmp_path / "data.json").exists()
    assert paths["data"].read_text().count("x_path") == 1
    args = run.as_command_args(tmp_path)
    assert len(args) == 4
    assert args[-1] == str(tmp_path)

