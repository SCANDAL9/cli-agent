"""
Security policy and Human-in-the-Loop (HITL) permission gate for cli-agent.
Classifies tool calls by risk level and intercepts potentially destructive operations.
"""

import re
from collections.abc import Callable
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


class SecurityPolicy:
    """
    Evaluates tool invocation safety and enforces human-in-the-loop confirmation.
    """

    def __init__(
        self,
        auto_approve: bool = False,
        confirm_callback: Callable[[str, dict[str, Any], RiskLevel], bool] | None = None,
    ):
        self.auto_approve = auto_approve
        self.confirm_callback = confirm_callback
        self._session_allowlist: set[str] = set()

    def evaluate_risk(self, tool_name: str, arguments: dict[str, Any]) -> RiskLevel:
        """Classifies tool invocation into SAFE, MEDIUM, or HIGH risk."""
        if tool_name in {
            "read_file",
            "list_directory",
            "search_code",
            "hybrid_search_code",
            "get_symbol_outline",
            "find_symbol_definition",
            "find_symbol_references",
            "analyze_symbol_impact",
            "search_memory",
            "save_memory",
        }:
            return RiskLevel.SAFE

        if tool_name in {"write_file", "edit_file"}:
            return RiskLevel.MEDIUM

        if tool_name == "run_command":
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
        if self.confirm_callback is not None:
            is_confirmed = self.confirm_callback(tool_name, arguments, risk)
            if is_confirmed:
                return True, "User confirmed tool execution."
            return False, "User denied permission for this action."

        # 5. Non-interactive environment with no callback and no auto-approve
        return False, f"Mutating action '{tool_name}' blocked: requires interactive approval."

    def grant_session_permission(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Grants recurring permission for this tool or command for the remainder of the session."""
        cache_key = self._get_cache_key(tool_name, arguments)
        self._session_allowlist.add(cache_key)

    def _get_cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name == "run_command":
            cmd = arguments.get("command", "").strip()
            # Cache the base command name (e.g. pytest, git status)
            return f"run_command:{cmd}"
        path = arguments.get("path", "").strip()
        return f"{tool_name}:{path}"
