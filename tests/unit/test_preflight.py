"""
Unit tests for the Enterprise Pre-Flight, IAM & Guided Onboarding Engine (Wave 3).
Verifies gcloud session checks, project resolution, service API verification,
IAM permission validation, billing status guards, and auto-fix capabilities.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli_agent.preflight import (
    BillingValidator,
    GCloudSessionManager,
    IAMValidator,
    PreflightEngine,
    PreflightStatus,
    ProjectSelector,
    ServiceUsageManager,
)


@pytest.mark.unit
class TestGCloudSessionManager:
    def test_get_active_account_authenticated(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "dev@example.com\n"

        with patch("subprocess.run", return_value=mock_proc):
            account = GCloudSessionManager.get_active_account()
            assert account == "dev@example.com"

    def test_get_active_account_unauthenticated(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "\n"

        with patch("subprocess.run", return_value=mock_proc):
            account = GCloudSessionManager.get_active_account()
            assert account is None

    def test_trigger_login_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            success = GCloudSessionManager.trigger_login(update_adc=True)
            assert success is True
            assert "--update-adc" in mock_run.call_args[0][0]


@pytest.mark.unit
class TestProjectSelector:
    def test_list_accessible_projects_success(self) -> None:
        sample_projects = [
            {"projectId": "proj-alpha", "name": "Alpha", "projectNumber": "111"},
            {"projectId": "proj-beta", "name": "Beta", "projectNumber": "222"},
        ]
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps(sample_projects)

        with patch("subprocess.run", return_value=mock_proc):
            projects = ProjectSelector.list_accessible_projects()
            assert len(projects) == 2
            assert projects[0]["projectId"] == "proj-alpha"

    def test_list_accessible_projects_failure(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""

        with patch("subprocess.run", return_value=mock_proc):
            projects = ProjectSelector.list_accessible_projects()
            assert projects == []

    def test_save_project_to_config_new_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        assert not cfg_file.exists()

        success = ProjectSelector.save_project_to_config("new-project-123", cfg_file)
        assert success is True
        content = cfg_file.read_text(encoding="utf-8")
        assert "[defaults]" in content
        assert 'project = "new-project-123"' in content

    def test_save_project_to_config_existing_update(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            """
[defaults]
project = "old-project"
location = "global"
""",
            encoding="utf-8",
        )

        success = ProjectSelector.save_project_to_config("updated-project", cfg_file)
        assert success is True
        content = cfg_file.read_text(encoding="utf-8")
        assert 'project = "updated-project"' in content
        assert 'location = "global"' in content


@pytest.mark.unit
class TestServiceUsageAndBillingValidators:
    def test_is_api_enabled_true(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "aiplatform.googleapis.com\n"

        with patch("subprocess.run", return_value=mock_proc):
            assert ServiceUsageManager.is_api_enabled("test-proj") is True

    def test_is_api_enabled_false(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""

        with patch("subprocess.run", return_value=mock_proc):
            assert ServiceUsageManager.is_api_enabled("test-proj") is False

    def test_enable_api_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            assert ServiceUsageManager.enable_api("test-proj") is True

    def test_billing_check_enabled(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({"billingEnabled": True})

        with patch("subprocess.run", return_value=mock_proc):
            ok, msg = BillingValidator.check_billing("test-proj")
            assert ok is True
            assert "actively linked" in msg

    def test_billing_check_disabled(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({"billingEnabled": False})

        with patch("subprocess.run", return_value=mock_proc):
            ok, msg = BillingValidator.check_billing("test-proj")
            assert ok is False
            assert "Billing is disabled" in msg


@pytest.mark.unit
class TestIAMValidator:
    def test_iam_check_with_valid_role(self) -> None:
        policy = {
            "bindings": [
                {
                    "role": "roles/aiplatform.user",
                    "members": ["user:developer@corp.com"],
                }
            ]
        }
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps(policy)

        with patch("subprocess.run", return_value=mock_proc):
            ok, msg = IAMValidator.check_permissions("test-proj", "developer@corp.com")
            assert ok is True
            assert "verified Vertex AI permissions" in msg

    def test_iam_check_empty_project(self) -> None:
        ok, msg = IAMValidator.check_permissions("")
        assert ok is False
        assert "not specified" in msg


@pytest.mark.unit
class TestPreflightEngine:
    def test_missing_gcloud_cli_fails_immediately(self) -> None:
        engine = PreflightEngine()
        with patch("shutil.which", return_value=None):
            report = engine.run_checks()
            assert report.passed is False
            assert len(report.items) == 1
            assert report.items[0].status == PreflightStatus.FAIL

    def test_full_successful_preflight_pipeline(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        engine = PreflightEngine(config_path=cfg_file)

        with (
            patch("shutil.which", return_value="/usr/bin/gcloud"),
            patch.object(engine.session_mgr, "get_active_account", return_value="dev@corp.com"),
            patch.object(engine.service_mgr, "is_api_enabled", return_value=True),
            patch.object(
                engine.billing_validator, "check_billing", return_value=(True, "Billing active")
            ),
            patch.object(engine.iam_validator, "check_permissions", return_value=(True, "IAM OK")),
        ):
            report = engine.run_checks(project_id="corp-ai-prod")
            assert report.passed is True
            assert report.account == "dev@corp.com"
            assert report.project == "corp-ai-prod"
            assert all(item.status == PreflightStatus.PASS for item in report.items)

    def test_preflight_auto_fixes_missing_project(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        engine = PreflightEngine(config_path=cfg_file)

        sample_projects = [
            {"projectId": "auto-discovered-proj", "name": "Auto", "projectNumber": "123"}
        ]

        with (
            patch("shutil.which", return_value="/usr/bin/gcloud"),
            patch.object(engine.session_mgr, "get_active_account", return_value="dev@corp.com"),
            patch.object(
                engine.project_selector, "list_accessible_projects", return_value=sample_projects
            ),
            patch.object(engine.service_mgr, "is_api_enabled", return_value=True),
            patch.object(
                engine.billing_validator, "check_billing", return_value=(True, "Billing OK")
            ),
            patch.object(engine.iam_validator, "check_permissions", return_value=(True, "IAM OK")),
            patch("os.getenv", return_value=None),
            patch("subprocess.run", return_value=MagicMock(stdout="(unset)\n")),
        ):
            report = engine.run_checks(project_id=None, auto_fix=True)
            assert report.passed is True
            assert report.project == "auto-discovered-proj"
            assert cfg_file.exists()
            assert 'project = "auto-discovered-proj"' in cfg_file.read_text(encoding="utf-8")

    def test_unauthenticated_session_fails(self) -> None:
        engine = PreflightEngine()
        with (
            patch("shutil.which", return_value="/usr/bin/gcloud"),
            patch.object(engine.session_mgr, "get_active_account", return_value=None),
        ):
            report = engine.run_checks(auto_fix=False)
            assert report.passed is False
            assert any(
                item.name == "Google Cloud Session" and item.status == PreflightStatus.FAIL
                for item in report.items
            )

    def test_api_disabled_auto_fix_and_warnings(self) -> None:
        engine = PreflightEngine()
        with (
            patch("shutil.which", return_value="/usr/bin/gcloud"),
            patch.object(engine.session_mgr, "get_active_account", return_value="user@corp.com"),
            patch.object(engine.service_mgr, "is_api_enabled", side_effect=[False, True]),
            patch.object(engine.service_mgr, "enable_api", return_value=True),
            patch.object(
                engine.billing_validator, "check_billing", return_value=(False, "No billing linked")
            ),
            patch.object(
                engine.iam_validator, "check_permissions", return_value=(False, "Missing role")
            ),
        ):
            report = engine.run_checks(project_id="test-proj", auto_fix=True)
            assert report.passed is True
            assert report.has_warnings is True
            statuses = {item.name: item.status for item in report.items}
            assert statuses["Vertex AI Service API"] == PreflightStatus.PASS
            assert statuses["Project Billing"] == PreflightStatus.WARN
            assert statuses["IAM Permissions"] == PreflightStatus.WARN

    def test_billing_and_iam_permission_denied_handling(self) -> None:
        mock_perm_denied = MagicMock()
        mock_perm_denied.returncode = 1
        mock_perm_denied.stderr = "ERROR: PERMISSION_DENIED on resource"

        with patch("subprocess.run", return_value=mock_perm_denied):
            b_ok, b_msg = BillingValidator.check_billing("restricted-proj")
            assert b_ok is True
            assert "skipped" in b_msg

            i_ok, i_msg = IAMValidator.check_permissions("restricted-proj", "dev@corp.com")
            assert i_ok is True
            assert "developer access" in i_msg
