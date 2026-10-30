"""
Unit tests for SecurityPolicy and permission gates.
"""

import pytest

from cli_agent.security import RiskLevel, SecurityPolicy


@pytest.mark.unit
class TestSecurityPolicy:
    def test_risk_classification(self):
        policy = SecurityPolicy()

        # Safe tools
        assert policy.evaluate_risk("read_file", {"path": "a.txt"}) == RiskLevel.SAFE
        assert policy.evaluate_risk("list_directory", {"path": "."}) == RiskLevel.SAFE
        assert policy.evaluate_risk("search_code", {"pattern": "def"}) == RiskLevel.SAFE
        assert policy.evaluate_risk("search_memory", {"query": "auth"}) == RiskLevel.SAFE
        assert policy.evaluate_risk("save_memory", {"title": "T", "content": "C"}) == RiskLevel.SAFE

        # Medium risk tools
        assert (
            policy.evaluate_risk("write_file", {"path": "out.txt", "content": "..."})
            == RiskLevel.MEDIUM
        )
        assert (
            policy.evaluate_risk("edit_file", {"path": "a.py", "target": "1", "replacement": "2"})
            == RiskLevel.MEDIUM
        )
        assert policy.evaluate_risk("run_command", {"command": "pytest -v"}) == RiskLevel.MEDIUM

        # High risk destructive commands
        assert policy.evaluate_risk("run_command", {"command": "rm -rf /"}) == RiskLevel.HIGH
        assert (
            policy.evaluate_risk("run_command", {"command": "git reset --hard HEAD~1"})
            == RiskLevel.HIGH
        )
        assert policy.evaluate_risk("run_command", {"command": "git clean -fd"}) == RiskLevel.HIGH
        assert (
            policy.evaluate_risk("run_command", {"command": "drop database production"})
            == RiskLevel.HIGH
        )

    def test_safe_tools_always_permitted(self):
        # Even with no callback and no auto_approve
        policy = SecurityPolicy(auto_approve=False, confirm_callback=None)
        allowed, reason = policy.check_permission("read_file", {"path": "test.txt"})
        assert allowed is True
        assert "Safe" in reason

    def test_auto_approve_permits_medium_but_not_high(self):
        policy = SecurityPolicy(auto_approve=True, confirm_callback=None)

        # Medium permitted
        allowed_med, _ = policy.check_permission("write_file", {"path": "test.txt", "content": "x"})
        assert allowed_med is True

        # High risk blocked without explicit callback
        allowed_high, reason = policy.check_permission(
            "run_command", {"command": "rm -rf /tmp/data"}
        )
        assert allowed_high is False
        assert "blocked" in reason

    def test_interactive_confirm_callback(self):
        # 1. User confirms
        policy_yes = SecurityPolicy(
            auto_approve=False, confirm_callback=lambda name, args, risk: True
        )
        allowed_yes, _ = policy_yes.check_permission(
            "write_file", {"path": "a.txt", "content": "..."}
        )
        assert allowed_yes is True

        # 2. User denies
        policy_no = SecurityPolicy(
            auto_approve=False, confirm_callback=lambda name, args, risk: False
        )
        allowed_no, reason_no = policy_no.check_permission(
            "write_file", {"path": "a.txt", "content": "..."}
        )
        assert allowed_no is False
        assert "denied" in reason_no

    def test_grant_session_permission(self):
        policy = SecurityPolicy(auto_approve=False, confirm_callback=None)
        args = {"command": "pytest tests/"}

        # Initially blocked
        allowed_before, _ = policy.check_permission("run_command", args)
        assert allowed_before is False

        # Grant permission for the session
        policy.grant_session_permission("run_command", args)

        allowed_after, reason = policy.check_permission("run_command", args)
        assert allowed_after is True
        assert "pre-approved" in reason
