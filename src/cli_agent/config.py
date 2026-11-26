"""
Configuration management for cli-agent.
Loads settings from ~/.config/cli-agent/config.toml with environment variable fallbacks.
"""

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
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


VALID_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")


def normalize_effort(value: str | None, default: str = "medium") -> str:
    """Normalizes a reasoning effort string to one of 'low', 'medium', or 'high'."""
    cleaned = str(value or "").strip().lower()
    if cleaned in VALID_EFFORT_LEVELS:
        return cleaned
    return default


@dataclass
class AgentConfig:
    project: str = "my-gcp-project"
    location: str = "global"
    model: str = "gemini-3.8-flash"
    temperature: float = 0.2
    auto_approve: bool = False
    timeout: float = 300.0
    effort: str = "medium"


@dataclass(frozen=True)
class StorageConfig:
    """Storage backend settings; authentication tokens are never loaded from TOML."""

    backend: str = "auto"
    turso_database_url: str | None = None
    turso_auth_token: str | None = field(default=None, repr=False)
    turso_mode: str = "replica"
    sync_interval: float = 30.0
    replica_path: Path | None = None


def load_storage_config(config_path: Path | None = None) -> StorageConfig:
    """Loads storage settings, reading Turso credentials only from the environment."""
    path = config_path or DEFAULT_CONFIG_FILE
    file_data: dict[str, Any] = {}
    if path.is_file():
        try:
            with open(path, "rb") as f:
                file_data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            file_data = {}

    raw_storage = file_data.get("storage")
    storage: dict[str, Any] = raw_storage if isinstance(raw_storage, dict) else {}
    backend = str(os.getenv("CLI_AGENT_STORAGE_BACKEND") or storage.get("backend") or "auto")
    backend = backend.strip().lower()
    if backend not in {"auto", "sqlite", "turso"}:
        raise ValueError("CLI_AGENT_STORAGE_BACKEND must be 'auto', 'sqlite', or 'turso'.")

    turso_mode = str(os.getenv("TURSO_MODE") or storage.get("turso_mode") or "replica")
    turso_mode = turso_mode.strip().lower()
    if turso_mode not in {"replica", "remote"}:
        raise ValueError("TURSO_MODE must be 'replica' or 'remote'.")

    raw_interval = os.getenv("TURSO_SYNC_INTERVAL") or storage.get("sync_interval", 30.0)
    try:
        sync_interval = float(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("TURSO_SYNC_INTERVAL must be a positive number of seconds.") from exc
    if sync_interval <= 0:
        raise ValueError("TURSO_SYNC_INTERVAL must be a positive number of seconds.")

    raw_replica = os.getenv("TURSO_REPLICA_PATH") or storage.get("replica_path")
    replica_path = Path(raw_replica).expanduser() if raw_replica else None

    return StorageConfig(
        backend=backend,
        turso_database_url=os.getenv("TURSO_DATABASE_URL") or storage.get("database_url"),
        turso_auth_token=os.getenv("TURSO_AUTH_TOKEN"),
        turso_mode=turso_mode,
        sync_interval=sync_interval,
        replica_path=replica_path,
    )


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

    raw_effort = os.getenv("CLI_AGENT_EFFORT") or defaults_sec.get("effort", "medium")
    effort = normalize_effort(str(raw_effort))

    return AgentConfig(
        project=project,
        location=location,
        model=model,
        temperature=temperature,
        auto_approve=auto_approve,
        timeout=timeout,
        effort=effort,
    )


def _save_default_setting(key: str, value: str, config_path: Path | None = None) -> bool:
    """Persists a string key/value under [defaults] in the local TOML config file."""
    cfg_path = config_path or DEFAULT_CONFIG_FILE
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing_content = ""
        if cfg_path.is_file():
            existing_content = cfg_path.read_text(encoding="utf-8")

        lines = existing_content.splitlines()
        defaults_found = False
        key_updated = False
        new_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "[defaults]":
                defaults_found = True
                new_lines.append(line)
            elif defaults_found and stripped.startswith(key) and "=" in stripped:
                lhs = stripped.split("=", 1)[0].strip()
                if lhs == key:
                    new_lines.append(f'{key} = "{value}"')
                    key_updated = True
                else:
                    new_lines.append(line)
            elif stripped.startswith("[") and stripped != "[defaults]":
                if defaults_found and not key_updated:
                    new_lines.append(f'{key} = "{value}"')
                    key_updated = True
                defaults_found = False
                new_lines.append(line)
            else:
                new_lines.append(line)

        if not defaults_found and not key_updated:
            if new_lines and new_lines[-1] != "":
                new_lines.append("")
            new_lines.append("[defaults]")
            new_lines.append(f'{key} = "{value}"')
        elif defaults_found and not key_updated:
            new_lines.append(f'{key} = "{value}"')

        cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def save_model_to_config(model_name: str, config_path: Path | None = None) -> bool:
    """Persists the selected model name to the local TOML configuration file."""
    clean_name = str(model_name).strip()
    if not clean_name or "Mock" in clean_name or "<" in clean_name:
        clean_name = "gemini-3.8-flash"
    return _save_default_setting("model", clean_name, config_path=config_path)


def save_effort_to_config(effort: str, config_path: Path | None = None) -> bool:
    """Persists the selected reasoning effort level ('low', 'medium', 'high') to config.toml."""
    clean_effort = normalize_effort(effort)
    return _save_default_setting("effort", clean_effort, config_path=config_path)
