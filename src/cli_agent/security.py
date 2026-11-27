"""
Security policy and Human-in-the-Loop (HITL) permission gate for cli-agent.
Classifies tool calls by risk level and intercepts potentially destructive operations.
"""

import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any


class RiskLevel(StrEnum):
    SAFE = "safe"
    MEDIUM = "medium"
    HIGH = "high"


# Dangerous patterns in shell commands that warrant HIGH risk warning
HIGH_RISK_PATTERNS = [
    r"\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*",  # rm -r / rm -rf / rm -fr
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+.*-[a-zA-Z]*f[a-zA-Z]*",  # git clean -f / -fd / -df
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:&\s*\};:",  # Fork bomb
    r"\bdrop\s+(table|database|schema)\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r">\s*/dev/sd[a-z]",
]

COMPILED_HIGH_RISK = [re.compile(p, re.IGNORECASE) for p in HIGH_RISK_PATTERNS]
PermissionCallback = Callable[[str, dict[str, Any], RiskLevel], bool | Awaitable[bool]]


class SecurityPolicy:
    """
    Evaluates tool invocation safety and enforces human-in-the-loop confirmation.
    """

    def __init__(
        self,
        auto_approve: bool = False,
        confirm_callback: PermissionCallback | None = None,
    ):
        self.auto_approve = auto_approve
        self.confirm_callback = confirm_callback
        self._session_allowlist: set[str] = set()
        self._confirm_override: ContextVar[PermissionCallback | None] = ContextVar(
            f"security_confirm_override_{id(self)}", default=None
        )

    def evaluate_risk(self, tool_name: str, arguments: dict[str, Any]) -> RiskLevel:
        """Classifies tool invocation into SAFE, MEDIUM, or HIGH risk."""
        if tool_name == "apply_worker_diff":
            return RiskLevel.HIGH

        if tool_name == "run_wasm_plugin":
            capability_fields = (
                "read_dirs",
                "write_dirs",
                "environment_variables",
            )
            if arguments.get("network_access") or any(
                arguments.get(field) for field in capability_fields
            ):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        if tool_name == "apply_code_action":
            selection = any(
                arguments.get(field) is not None
                for field in ("server_name", "action_index", "action_title", "selection_id")
            )
            return RiskLevel.MEDIUM if selection else RiskLevel.SAFE

        if tool_name in {
            "read_file",
            "list_directory",
            "search_code",
            "lsp_workspace_diagnostics",
            "ast_grep_replace",
            "hybrid_search_code",
            "semantic_search_code",
            "get_symbol_call_tree",
            "get_type_hierarchy",
            "get_symbol_outline",
            "find_symbol_definition",
            "find_symbol_references",
            "analyze_symbol_impact",
            "search_memory",
            "save_memory",
            "check_background_process",
        }:
            return RiskLevel.SAFE

        if tool_name in {
            "write_file",
            "edit_file",
            "apply_code_action",
            "stop_background_process",
        }:
            return RiskLevel.MEDIUM

        if tool_name in {"run_command", "start_background_process"}:
            if arguments.get("network_access") or arguments.get("allow_reduced_isolation"):
                return RiskLevel.HIGH
            cmd = arguments.get("command", "")
            for pattern in COMPILED_HIGH_RISK:
                if pattern.search(cmd):
                    return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Any unknown or external tool defaults to MEDIUM risk
        return RiskLevel.MEDIUM

    def check_permission(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """
        Determines whether tool execution is permitted.
        Returns (is_allowed, reason).
        """
        risk = self.evaluate_risk(tool_name, arguments)

        # 1. Safe read-only tools always proceed without prompting
        if risk == RiskLevel.SAFE:
            return True, "Safe read-only tool auto-permitted."

        # 2. Check if this exact tool signature or command was pre-approved for this session
        cache_key = self._get_cache_key(tool_name, arguments)
        if cache_key in self._session_allowlist:
            return True, f"Action '{cache_key}' pre-approved for active session."

        # 3. Auto-approve mode permits MEDIUM risk; HIGH risk still requires explicit confirmation
        if self.auto_approve and risk != RiskLevel.HIGH:
            return True, "Auto-approve flag enabled for medium risk operations."

        # 4. If an interactive confirmation callback is registered, query the human operator
        callback = self._confirm_override.get() or self.confirm_callback
        if callback is not None:
            is_confirmed = callback(tool_name, arguments, risk)
            if inspect.isawaitable(is_confirmed):
                close = getattr(is_confirmed, "close", None)
                if callable(close):
                    close()
                return False, "Asynchronous approval requires check_permission_async."
            if is_confirmed:
                return True, "User confirmed tool execution."
            return False, "User denied permission for this action."

        # 5. Non-interactive environment with no callback and no auto-approve
        return False, f"Mutating action '{tool_name}' blocked: requires interactive approval."

    async def check_permission_async(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str]:
        """Async variant that safely awaits context-local approval brokers."""
        risk = self.evaluate_risk(tool_name, arguments)
        if risk == RiskLevel.SAFE:
            return True, "Safe read-only tool auto-permitted."
        cache_key = self._get_cache_key(tool_name, arguments)
        if cache_key in self._session_allowlist:
            return True, f"Action '{cache_key}' pre-approved for active session."
        if self.auto_approve and risk != RiskLevel.HIGH:
            return True, "Auto-approve flag enabled for medium risk operations."
        callback = self._confirm_override.get() or self.confirm_callback
        if callback is None:
            return False, f"Mutating action '{tool_name}' blocked: requires interactive approval."
        try:
            confirmed = callback(tool_name, arguments, risk)
            if inspect.isawaitable(confirmed):
                confirmed = await confirmed
        except Exception as exc:
            return False, f"Permission approval failed: {exc}"
        if confirmed:
            return True, "User confirmed tool execution."
        return False, "User denied permission for this action."

    @contextmanager
    def confirmation_scope(self, callback: PermissionCallback) -> Iterator[None]:
        """Override confirmation only in this async execution context."""
        token = self._confirm_override.set(callback)
        try:
            yield
        finally:
            self._confirm_override.reset(token)

    def grant_session_permission(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Grants recurring permission for this tool or command for the remainder of the session."""
        cache_key = self._get_cache_key(tool_name, arguments)
        self._session_allowlist.add(cache_key)

    def _get_cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name == "apply_worker_diff":
            diff = arguments.get("diff")
            if isinstance(diff, str):
                diff_bytes = diff.encode("utf-8")
            elif isinstance(diff, bytes):
                diff_bytes = diff
            else:
                diff_bytes = repr(diff).encode("utf-8")
            diff_hash = hashlib.sha256(diff_bytes).hexdigest()
            return (
                "apply_worker_diff:"
                f"{arguments.get('task_id', '')}:"
                f"{arguments.get('worker_id', '')}:"
                f"{arguments.get('lease_id', '')}:"
                f"{arguments.get('generation', '')}:"
                f"{arguments.get('diff_sha256', '')}:"
                f"{diff_hash}"
            )

        if tool_name == "run_wasm_plugin":

            def normalized_values(name: str) -> list[str]:
                value = arguments.get(name, [])
                if isinstance(value, (list, tuple, set)):
                    return sorted(str(item) for item in value)
                return [repr(value)] if value else []

            capabilities = {
                "read_dirs": normalized_values("read_dirs"),
                "write_dirs": normalized_values("write_dirs"),
                "environment_variables": normalized_values("environment_variables"),
                "network_access": bool(arguments.get("network_access")),
            }
            fingerprint = hashlib.sha256(
                json.dumps(capabilities, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            module_path = str(arguments.get("module_path", "")).strip().replace("\\", "/")
            return f"run_wasm_plugin:{module_path}:{fingerprint}"

        capability_suffix = ""
        if tool_name in {"run_command", "start_background_process"}:
            network_access = bool(arguments.get("network_access"))
            reduced_isolation = bool(arguments.get("allow_reduced_isolation"))
            if network_access or reduced_isolation:
                capability_suffix = (
                    f":network={int(network_access)}:reduced-isolation={int(reduced_isolation)}"
                )
        if tool_name == "run_command":
            cmd = arguments.get("command", "").strip()
            # Cache the base command name (e.g. pytest, git status)
            return f"run_command:{cmd}{capability_suffix}"
        if tool_name == "start_background_process" and capability_suffix:
            cmd = arguments.get("command", "").strip()
            return f"start_background_process:{cmd}{capability_suffix}"
        path = arguments.get("path", "").strip()
        return f"{tool_name}:{path}{capability_suffix}"
