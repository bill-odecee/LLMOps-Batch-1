import pytest
from utils.config_loader import load_config


def test_load_config_with_explicit_path(tmp_path):
    config_yaml = """
foo:
  bar: 123
baz:
  - 1
  - 2
  - 3
"""
    config_file = tmp_path / "config_loader.yaml"
    config_file.write_text(config_yaml)

    result = load_config(str(config_file))

    assert result == {
        "foo": {"bar": 123},
        "baz": [1, 2, 3],
    }


def test_load_config_with_default_path(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    utils_dir = project_root / "utils"
    config_dir = project_root / "config"
    utils_dir.mkdir(parents=True)
    config_dir.mkdir()

    default_file = config_dir / "config_loader.yaml"
    default_file.write_text("default:\n  value: 42\n")

    import utils.config_loader as config_loader
    monkeypatch.setattr(config_loader, "__file__", str(utils_dir / "config_loader.py"))

    result = config_loader.load_config()

    assert result == {"default": {"value": 42}}


def test_load_config_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing_config.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(str(missing_path))
