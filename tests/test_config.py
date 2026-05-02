import textwrap

import pytest

from epintlm.config import Config, load_config, parse_overrides


def test_default_config_loads(tmp_path):
    cfg_path = tmp_path / "minimal.yaml"
    cfg_path.write_text("run:\n  name: minimal\n")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.run.name == "minimal"
    assert cfg.training.batch_size == 256  # Default still applies


def test_cli_overrides(tmp_path):
    cfg_path = tmp_path / "minimal.yaml"
    cfg_path.write_text("run:\n  name: base\n")
    cfg = load_config(cfg_path, overrides=["training.batch_size=64", "run.seed=42"])
    assert cfg.training.batch_size == 64
    assert cfg.run.seed == 42
    # Untouched fields keep defaults
    assert cfg.training.num_epochs == 40


def test_override_typed_via_yaml():
    out = parse_overrides(["model.attention.use_residual=false", "model.gru.hidden_size=16"])
    assert out["model"]["attention"]["use_residual"] is False
    assert out["model"]["gru"]["hidden_size"] == 16


def test_override_invalid_raises():
    with pytest.raises(ValueError):
        parse_overrides(["bare_key_no_value"])


def test_repo_default_yaml_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.data.cell_line == "HeLa-S3"
    assert cfg.model.attention.embed_dim == 64
    assert cfg.training.num_epochs == 40
