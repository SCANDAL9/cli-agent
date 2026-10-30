"""
Configuration management for cli-agent.
Loads settings from ~/.config/cli-agent/config.toml with environment variable fallbacks.
"""

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "cli-agent"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_HISTORY_FILE = DEFAULT_CONFIG_DIR / "history"


def _detect_gcp_project() -> str:
    """Attempts to detect the active Google Cloud project from env vars or gcloud CLI."""
    for env_var in ("GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        val = os.getenv(env_var)
        if val:
            return val
    try:
        proc = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        val = proc.stdout.strip()
        if val and val != "(unset)":
            return val
    except Exception:
        pass
    return "my-gcp-project"


@dataclass
class AgentConfig:
    project: str = "my-gcp-project"
    location: str = "global"
    model: str = "gemini-3.8-flash"
    temperature: float = 0.2
    auto_approve: bool = False


def load_config(config_path: Path | None = None) -> AgentConfig:
    """
    Loads agent configuration from a TOML file if present,
    falling back to environment variables and default values.
    """
    path = config_path or DEFAULT_CONFIG_FILE
    file_data: dict[str, Any] = {}

    if path.is_file():
        try:
            with open(path, "rb") as f:
                file_data = tomllib.load(f)
        except Exception:
            pass

    raw_defaults = file_data.get("defaults")
    defaults_sec: dict[str, Any] = raw_defaults if isinstance(raw_defaults, dict) else {}

    detected_project = _detect_gcp_project()
    project = str(os.getenv("GCP_PROJECT") or defaults_sec.get("project") or detected_project)
    location = str(os.getenv("GEMINI_LOCATION") or defaults_sec.get("location") or "global")
    model = str(os.getenv("GEMINI_MODEL") or defaults_sec.get("model") or "gemini-3.8-flash")
    raw_temp = defaults_sec.get("temperature", 0.2)
    temperature = float(raw_temp) if isinstance(raw_temp, (int, float)) else 0.2

    raw_auto = defaults_sec.get("auto_approve", False)
    auto_approve = bool(raw_auto) if isinstance(raw_auto, bool) else False

    return AgentConfig(
        project=project,
        location=location,
        model=model,
        temperature=temperature,
        auto_approve=auto_approve,
    )
