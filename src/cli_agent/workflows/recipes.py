"""Built-in workflow recipes shipped with cli-agent."""

from __future__ import annotations

from typing import Any


def built_in_workflows() -> tuple[dict[str, Any], ...]:
    """Return fresh recipe objects so callers cannot mutate global defaults."""
    return (
        {
            "id": "strict-tdd",
            "name": "Strict TDD",
            "description": "Establish a focused red test, implement, then verify green.",
            "version": "1.0.0",
            "inputs": {
                "feature_description": {"type": "string", "description": "Feature or bugfix"},
                "target_module": {"type": "path", "description": "Target module file path"},
                "test_name": {"type": "string", "description": "Expected failing test name"},
            },
            "steps": [
                {
                    "id": "write_red_test",
                    "name": "Author a failing regression test",
                    "action": "agent_prompt",
                    "prompt": "Add a focused pytest for {{ feature_description }} in {{ target_module }}. Do not implement the fix yet.",
                    "verify_gate": {
                        "type": "command_exit_code",
                        "command": "pytest tests/ -k {{ test_name }}",
                        "expected_exit_code": 1,
                        "expected_output": "FAILED {{ test_name }}",
                    },
                },
                {
                    "id": "implement",
                    "name": "Implement the minimum fix",
                    "action": "agent_prompt",
                    "prompt": "Implement the minimum change for {{ feature_description }} in {{ target_module }}.",
                },
                {
                    "id": "verify_ast",
                    "name": "Validate the target module AST",
                    "action": "verify_gate",
                    "verify_gate": {
                        "type": "command_exit_code",
                        "command": "python -m ast {{ target_module }}",
                        "expected_exit_code": 0,
                    },
                },
                {
                    "id": "verify_green",
                    "name": "Verify the focused test",
                    "action": "verify_gate",
                    "verify_gate": {
                        "type": "command_exit_code",
                        "command": "pytest tests/ -k {{ test_name }}",
                        "expected_exit_code": 0,
                    },
                },
            ],
        },
        {
            "id": "code-health",
            "name": "Code Health",
            "description": "Run formatting, lint, type, and repository diagnostics.",
            "version": "1.0.0",
            "steps": [
                {
                    "id": "scan_secrets",
                    "name": "Scan source for hard-coded credentials",
                    "action": "tool",
                    "tool_name": "search_code",
                    "params": {
                        "path": ".",
                        "pattern": r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
                    },
                },
                {
                    "id": "quality_checks",
                    "name": "Format, lint, and type-check the workspace",
                    "action": "shell",
                    "command": "ruff format . && ruff check --fix src/ tests/ && mypy src/cli_agent",
                },
                {
                    "id": "lsp_diagnostics",
                    "name": "Collect workspace LSP diagnostics",
                    "action": "tool",
                    "tool_name": "lsp_workspace_diagnostics",
                    "params": {"path": ".", "max_files": 250},
                },
            ],
        },
        {
            "id": "security-audit",
            "name": "Security Audit",
            "description": "Inspect the workspace for common command and secret-handling risks.",
            "version": "1.0.0",
            "steps": [
                {
                    "id": "scan-dangerous-patterns",
                    "name": "Find shell and credential entry points",
                    "action": "tool",
                    "tool_name": "search_code",
                    "params": {"pattern": r"shell=True|subprocess\.|password|secret|token"},
                },
                {
                    "id": "review-findings",
                    "name": "Review security findings",
                    "action": "agent_prompt",
                    "prompt": "Review the previous scan results for concrete security weaknesses. Report file/line evidence and do not modify files.",
                },
            ],
        },
        {
            "id": "dependency-audit",
            "name": "Dependency Audit",
            "description": "Inspect installed dependency versions and available updates.",
            "version": "1.0.0",
            "steps": [
                {
                    "id": "inspect-dependencies",
                    "name": "Inspect dependency graph",
                    "action": "shell",
                    "command": "uv tree --outdated",
                },
                {
                    "id": "find-unused-dependencies",
                    "name": "Find unused and misplaced dependencies",
                    "action": "shell",
                    "command": "uvx deptry .",
                },
                {
                    "id": "audit-known-vulnerabilities",
                    "name": "Audit installed packages for known advisories",
                    "action": "shell",
                    "command": "uv run --with pip-audit pip-audit --local",
                },
            ],
        },
    )
