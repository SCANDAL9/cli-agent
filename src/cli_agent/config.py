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
    timeout: float = 300.0


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

    raw_timeout = os.getenv("CLI_AGENT_TIMEOUT") or defaults_sec.get("timeout", 300.0)
    try:
        timeout = float(raw_timeout)
    except (ValueError, TypeError):
        timeout = 300.0

    return AgentConfig(
        project=project,
        location=location,
        model=model,
        temperature=temperature,
        auto_approve=auto_approve,
        timeout=timeout,
    )


def save_model_to_config(model_name: str, config_path: Path | None = None) -> bool:
    """Persists the selected model name to the local TOML configuration file."""
    clean_name = str(model_name).strip()
    if not clean_name or "Mock" in clean_name or "<" in clean_name:
        clean_name = "gemini-3.8-flash"

    cfg_path = config_path or DEFAULT_CONFIG_FILE
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing_content = ""
        if cfg_path.is_file():
            existing_content = cfg_path.read_text(encoding="utf-8")

        lines = existing_content.splitlines()
        defaults_found = False
        model_updated = False
        new_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "[defaults]":
                defaults_found = True
                new_lines.append(line)
            elif defaults_found and (stripped.startswith("model") and "=" in stripped):
                new_lines.append(f'model = "{clean_name}"')
                model_updated = True
            elif stripped.startswith("[") and stripped != "[defaults]":
                if defaults_found and not model_updated:
                    new_lines.append(f'model = "{clean_name}"')
                    model_updated = True
                defaults_found = False
                new_lines.append(line)
            else:
                new_lines.append(line)

        if not defaults_found and not model_updated:
            if new_lines and new_lines[-1] != "":
                new_lines.append("")
            new_lines.append("[defaults]")
            new_lines.append(f'model = "{clean_name}"')
        elif defaults_found and not model_updated:
            new_lines.append(f'model = "{clean_name}"')

        cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False
