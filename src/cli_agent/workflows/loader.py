"""Discovery and strict parsing for workspace, user, and bundled workflows."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .models import WorkflowDefinition, WorkflowValidationError, parse_workflow
from .recipes import built_in_workflows

_SUPPORTED_SUFFIXES = {".toml", ".yaml", ".yml"}
_MAX_WORKFLOW_BYTES = 1_000_000


class WorkflowLoader:
    """Loads declarative workflows with workspace-over-user-over-builtin precedence."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        user_config_dir: Path | str | None = None,
        *,
        include_builtins: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.user_config_dir = Path(
            user_config_dir or Path.home() / ".config" / "cli-agent" / "workflows"
        ).expanduser()
        self.include_builtins = include_builtins

    def list(self) -> list[WorkflowDefinition]:
        definitions: dict[str, WorkflowDefinition] = {}
        sources: dict[str, str] = {}
        if self.include_builtins:
            for raw in built_in_workflows():
                definition = parse_workflow(raw, source=f"builtin://{raw.get('id', '')}")
                definitions[definition.id] = definition
                sources[definition.id] = "builtin"

        for directory, source_name in (
            (self.user_config_dir, "user"),
            (self.workspace_root / ".cli-agent" / "workflows", "workspace"),
        ):
            for path in sorted(directory.glob("*")) if directory.is_dir() else ():
                if path.suffix.lower() not in _SUPPORTED_SUFFIXES or not path.is_file():
                    continue
                definition = self.load_file(path)
                if definition.id in definitions and sources[definition.id] == source_name:
                    raise WorkflowValidationError(
                        f"Duplicate workflow id '{definition.id}' in {source_name} workflows."
                    )
                definitions[definition.id] = definition
                sources[definition.id] = source_name
        return sorted(definitions.values(), key=lambda item: item.id)

    def get(self, workflow_id: str) -> WorkflowDefinition:
        for definition in self.list():
            if definition.id == workflow_id:
                return definition
        raise KeyError(f"Workflow '{workflow_id}' was not found.")

    def load_file(self, path: Path | str) -> WorkflowDefinition:
        source = Path(path).expanduser().resolve()
        try:
            if source.stat().st_size > _MAX_WORKFLOW_BYTES:
                raise WorkflowValidationError("Workflow files cannot exceed 1 MB.")
            raw = source.read_bytes()
            if source.suffix.lower() == ".toml":
                data: Any = tomllib.loads(raw.decode("utf-8"))
            else:
                try:
                    import yaml  # type: ignore[import-untyped]
                except ImportError as exc:
                    raise WorkflowValidationError(
                        "YAML workflows require PyYAML; install the cli-agent workflow dependencies."
                    ) from exc
                data = yaml.safe_load(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise WorkflowValidationError(
                f"Unable to read workflow '{source.name}': {exc}"
            ) from exc
        except Exception as exc:
            if isinstance(exc, WorkflowValidationError):
                raise
            raise WorkflowValidationError(
                f"Unable to parse workflow '{source.name}': {exc}"
            ) from exc
        return parse_workflow(data, source=str(source))
