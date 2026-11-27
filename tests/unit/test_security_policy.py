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
        assert (
            policy.evaluate_risk(
                "apply_code_action",
                {"path": "src/app.py", "action_title": "Add import"},
            )
            == RiskLevel.MEDIUM
        )

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

    def test_apply_code_action_reuses_existing_permission_modes(self):
        arguments = {
            "path": "src/app.py",
            "line": 4,
            "character": 2,
            "server_name": "pyright",
            "action_index": 0,
            "action_title": "Add import",
            "selection_id": "f353c4e202ad12345fa0",
        }
        denied, _ = SecurityPolicy().check_permission("apply_code_action", arguments)
        assert denied is False

        automatic, _ = SecurityPolicy(auto_approve=True).check_permission(
            "apply_code_action", arguments
        )
        assert automatic is True

        session_policy = SecurityPolicy()
        session_policy.grant_session_permission("apply_code_action", arguments)
        assert session_policy.check_permission("apply_code_action", arguments)[0] is True
        assert (
            session_policy.check_permission(
                "apply_code_action", {**arguments, "path": "src/other.py"}
            )[0]
            is False
        )

        listing = {"path": "src/app.py", "line": 4, "character": 2}
        assert SecurityPolicy().evaluate_risk("apply_code_action", listing) == RiskLevel.SAFE
        assert SecurityPolicy().check_permission("apply_code_action", listing)[0] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_permission_awaits_confirmation_callback():
    observed: list[tuple[str, RiskLevel]] = []

    async def approve(tool_name, _arguments, risk):
        observed.append((tool_name, risk))
        return True

    policy = SecurityPolicy(auto_approve=False, confirm_callback=approve)
    allowed, reason = await policy.check_permission_async(
        "write_file", {"path": "result.py", "content": "value = 1\n"}
    )

    assert allowed is True
    assert reason == "User confirmed tool execution."
    assert observed == [("write_file", RiskLevel.MEDIUM)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_permission_reports_denial_callback_errors_and_missing_approval():
    async def deny(_tool_name, _arguments, _risk):
        return False

    denied, denial_reason = await SecurityPolicy(confirm_callback=deny).check_permission_async(
        "write_file", {"path": "result.py", "content": "value = 1\n"}
    )
    assert denied is False
    assert denial_reason == "User denied permission for this action."

    def fail(_tool_name, _arguments, _risk):
        raise RuntimeError("approval dialog closed")

    failed, failure_reason = await SecurityPolicy(confirm_callback=fail).check_permission_async(
        "write_file", {"path": "result.py", "content": "value = 1\n"}
    )
    assert failed is False
    assert failure_reason == "Permission approval failed: approval dialog closed"

    blocked, blocked_reason = await SecurityPolicy().check_permission_async(
        "write_file", {"path": "result.py", "content": "value = 1\n"}
    )
    assert blocked is False
    assert blocked_reason == (
        "Mutating action 'write_file' blocked: requires interactive approval."
    )


def test_command_sandbox_grants_require_high_risk_approval_without_widening_allowlist():
    from cli_agent.security import RiskLevel, SecurityPolicy

    policy = SecurityPolicy(auto_approve=True)
    ordinary = {"command": "echo hello"}
    network_grant = {"command": "echo hello", "network_access": True}
    fallback_grant = {"command": "echo hello", "allow_reduced_isolation": True}
    background_grant = {
        "command": "sleep 1",
        "network_access": True,
        "allow_reduced_isolation": True,
    }

    assert policy.evaluate_risk("run_command", ordinary) == RiskLevel.MEDIUM
    assert policy.check_permission("run_command", ordinary)[0] is True
    assert policy.evaluate_risk("run_command", network_grant) == RiskLevel.HIGH
    assert policy.evaluate_risk("run_command", fallback_grant) == RiskLevel.HIGH
    assert policy.evaluate_risk("start_background_process", background_grant) == RiskLevel.HIGH
    assert policy.check_permission("run_command", network_grant)[0] is False
    assert policy.check_permission("run_command", fallback_grant)[0] is False

    policy.grant_session_permission("start_background_process", {"command": "sleep 1"})
    assert policy.check_permission("start_background_process", {"command": "sleep 1"})[0] is True
    assert policy.check_permission("start_background_process", background_grant)[0] is False
    policy.grant_session_permission("start_background_process", background_grant)
    assert policy.check_permission("start_background_process", background_grant)[0] is True
    assert (
        policy.check_permission(
            "start_background_process",
            {**background_grant, "command": "different process"},
        )[0]
        is False
    )

    policy.grant_session_permission("run_command", ordinary)
    assert policy.check_permission("run_command", ordinary)[0] is True
    assert policy.check_permission("run_command", network_grant)[0] is False
    assert policy.check_permission("run_command", fallback_grant)[0] is False
