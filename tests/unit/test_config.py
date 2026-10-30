"""
Unit tests for configuration loader (TOML parsing and env var fallback).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent.config import load_config


@pytest.mark.unit
class TestConfig:
    def test_default_config_fallback(self, monkeypatch):
        monkeypatch.delenv("GCP_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GEMINI_LOCATION", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        with patch("subprocess.run", side_effect=RuntimeError("gcloud missing")):
            cfg = load_config(config_path=Path("/nonexistent/config.toml"))
            assert cfg.project == "my-gcp-project"
            assert cfg.location == "global"
            assert cfg.model == "gemini-3.8-flash"
            assert cfg.temperature == 0.2
            assert cfg.auto_approve is False

    def test_load_from_toml_file(self, tmp_path: Path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            """
[defaults]
project = "custom-project-99"
location = "us-central1"
model = "gemini-3.1-pro"
temperature = 0.5
auto_approve = true
""",
            encoding="utf-8",
        )

        cfg = load_config(config_path=cfg_file)
        assert cfg.project == "custom-project-99"
        assert cfg.location == "us-central1"
        assert cfg.model == "gemini-3.1-pro"
        assert cfg.temperature == 0.5
        assert cfg.auto_approve is True

    def test_env_var_overrides_toml(self, tmp_path: Path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[defaults]\nmodel = 'gemini-1.5-flash'\n", encoding="utf-8")

        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash-override")
        cfg = load_config(config_path=cfg_file)
        assert cfg.model == "gemini-3.8-flash-override"
