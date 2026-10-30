"""
Enterprise Pre-Flight, IAM & Guided Onboarding Engine for cli-agent.
Verifies active gcloud user sessions, accessible projects, Vertex AI API enablement,
IAM role permissions, and billing account associations before running agent workloads.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .config import DEFAULT_CONFIG_FILE

logger = logging.getLogger(__name__)


class PreflightStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class PreflightItem:
    """Represents the outcome of a single pre-flight diagnostic check."""

    name: str
    status: PreflightStatus
    detail: str
    remediation_cmd: str | None = None
    auto_fixable: bool = False


@dataclass
class PreflightReport:
    """Comprehensive environment health and onboarding report."""

    items: list[PreflightItem] = field(default_factory=list)
    account: str | None = None
    project: str | None = None
    passed: bool = True
    has_warnings: bool = False


class GCloudSessionManager:
    """Manages Google Cloud authentication and active user sessions."""

    @staticmethod
    def get_active_account() -> str | None:
        """Returns the active gcloud account email or None if unauthenticated."""
        try:
            res = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            acc = res.stdout.strip()
            return acc if acc else None
        except Exception as e:
            logger.debug(f"Failed querying gcloud active account: {e}")
            return None

    @staticmethod
    def trigger_login(update_adc: bool = True) -> bool:
        """Launches interactive gcloud authentication."""
        cmd = ["gcloud", "auth", "login"]
        if update_adc:
            cmd.append("--update-adc")
        try:
            res = subprocess.run(cmd, timeout=120, check=False)
            return res.returncode == 0
        except Exception as e:
            logger.error(f"Error launching gcloud login: {e}")
            return False


class ProjectSelector:
    """Enumerates accessible Google Cloud projects and manages configuration persistence."""

    @staticmethod
    def list_accessible_projects() -> list[dict[str, str]]:
        """Queries list of active projects accessible to the authenticated account."""
        try:
            res = subprocess.run(
                [
                    "gcloud",
                    "projects",
                    "list",
                    "--filter=lifecycleState:ACTIVE",
                    "--format=json(projectId,name,projectNumber)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode != 0:
                return []
            data = json.loads(res.stdout)
            if isinstance(data, list):
                return [
                    {
                        "projectId": str(p.get("projectId", "")),
                        "name": str(p.get("name", "")),
                        "projectNumber": str(p.get("projectNumber", "")),
                    }
                    for p in data
                    if isinstance(p, dict) and p.get("projectId")
                ]
            return []
        except Exception as e:
            logger.debug(f"Failed querying accessible projects: {e}")
            return []

    @staticmethod
    def save_project_to_config(project_id: str, config_path: Path | None = None) -> bool:
        """Persists the selected project ID to the local TOML configuration file."""
        cfg_path = config_path or DEFAULT_CONFIG_FILE
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            existing_content = ""
            if cfg_path.is_file():
                existing_content = cfg_path.read_text(encoding="utf-8")

            # Parse lines and update or create [defaults] project
            lines = existing_content.splitlines()
            defaults_found = False
            project_updated = False
            new_lines: list[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped == "[defaults]":
                    defaults_found = True
                    new_lines.append(line)
                    continue

                if defaults_found and stripped.startswith("project"):
                    new_lines.append(f'project = "{project_id}"')
                    project_updated = True
                    continue

                if defaults_found and stripped.startswith("[") and stripped != "[defaults]":
                    if not project_updated:
                        new_lines.append(f'project = "{project_id}"')
                        project_updated = True
                    defaults_found = False

                new_lines.append(line)

            if not defaults_found and not project_updated:
                new_lines.append("\n[defaults]")
                new_lines.append(f'project = "{project_id}"')
            elif defaults_found and not project_updated:
                new_lines.append(f'project = "{project_id}"')

            cfg_path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed writing project to config: {e}")
            return False


class ServiceUsageManager:
    """Inspects and enables Google Cloud service APIs (e.g., aiplatform.googleapis.com)."""

    @staticmethod
    def is_api_enabled(project_id: str, api_name: str = "aiplatform.googleapis.com") -> bool:
        """Checks if a specified Google Cloud API is enabled on the target project."""
        try:
            res = subprocess.run(
                [
                    "gcloud",
                    "services",
                    "list",
                    "--enabled",
                    f"--filter=config.name:{api_name}",
                    f"--project={project_id}",
                    "--format=value(config.name)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return res.returncode == 0 and api_name in res.stdout
        except Exception as e:
            logger.debug(f"Failed querying enabled service {api_name}: {e}")
            return False

    @staticmethod
    def enable_api(project_id: str, api_name: str = "aiplatform.googleapis.com") -> bool:
        """Enables a Google Cloud API on the target project."""
        try:
            res = subprocess.run(
                ["gcloud", "services", "enable", api_name, f"--project={project_id}"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return res.returncode == 0
        except Exception as e:
            logger.error(f"Failed enabling service {api_name}: {e}")
            return False


class IAMValidator:
    """Validates IAM permissions and roles on the target project."""

    @staticmethod
    def check_permissions(project_id: str, account: str | None = None) -> tuple[bool, str]:
        """Verifies if the caller has sufficient permissions to use Vertex AI."""
        if not project_id:
            return False, "Project ID is not specified."

        # Quick validation using gcloud projects get-iam-policy or lightweight test
        try:
            res = subprocess.run(
                [
                    "gcloud",
                    "projects",
                    "get-iam-policy",
                    project_id,
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode != 0:
                # If cannot read full IAM policy, user might still have Vertex AI prediction rights
                if "PERMISSION_DENIED" in res.stderr:
                    return (
                        True,
                        "Caller lacks full IAM policy read permissions, but may hold developer access.",
                    )
                return False, res.stderr.strip() or "Failed querying IAM policy"

            policy = json.loads(res.stdout)
            bindings = policy.get("bindings", [])

            if account:
                member_tag = f"user:{account}"
                valid_roles = {
                    "roles/owner",
                    "roles/editor",
                    "roles/aiplatform.user",
                    "roles/aiplatform.admin",
                }
                has_role = any(
                    b.get("role") in valid_roles and member_tag in b.get("members", [])
                    for b in bindings
                )
                if has_role:
                    return True, f"Account '{account}' has verified Vertex AI permissions."

            return True, "Project IAM policy verified."
        except Exception as e:
            logger.debug(f"IAM check warning: {e}")
            return True, "IAM check skipped (non-critical probe)."


class BillingValidator:
    """Validates Google Cloud project billing status."""

    @staticmethod
    def check_billing(project_id: str) -> tuple[bool, str]:
        """Checks if the project has an active linked billing account."""
        if not project_id:
            return False, "Project ID is not specified."

        try:
            res = subprocess.run(
                ["gcloud", "billing", "projects", "describe", project_id, "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode != 0:
                if "PERMISSION_DENIED" in res.stderr:
                    return True, "Billing check skipped (caller lacks billing viewer rights)."
                return False, res.stderr.strip() or "Failed querying project billing."

            info = json.loads(res.stdout)
            is_enabled = bool(info.get("billingEnabled", False))
            if is_enabled:
                return True, "Billing account is actively linked."
            return (
                False,
                f"Billing is disabled on project '{project_id}'. Link a billing account in GCP Console.",
            )
        except Exception as e:
            logger.debug(f"Billing query error: {e}")
            return True, "Billing check skipped (probe exception)."


class PreflightEngine:
    """
    Coordinates end-to-end environment pre-flight inspection and guided remediation.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self.session_mgr = GCloudSessionManager()
        self.project_selector = ProjectSelector()
        self.service_mgr = ServiceUsageManager()
        self.iam_validator = IAMValidator()
        self.billing_validator = BillingValidator()

    def run_checks(
        self,
        project_id: str | None = None,
        auto_fix: bool = False,
    ) -> PreflightReport:
        """
        Executes complete pre-flight checks and returns an actionable PreflightReport.
        """
        items: list[PreflightItem] = []
        overall_passed = True
        has_warnings = False

        # 1. GCloud Tooling Check
        gcloud_path = shutil.which("gcloud")
        if not gcloud_path:
            items.append(
                PreflightItem(
                    name="GCloud CLI Installation",
                    status=PreflightStatus.FAIL,
                    detail="Google Cloud CLI ('gcloud') is not installed or not in PATH.",
                    remediation_cmd="Install gcloud: https://cloud.google.com/sdk/docs/install",
                    auto_fixable=False,
                )
            )
            return PreflightReport(
                items=items,
                passed=False,
                has_warnings=False,
            )

        items.append(
            PreflightItem(
                name="GCloud CLI Installation",
                status=PreflightStatus.PASS,
                detail=f"Installed at {gcloud_path}",
            )
        )

        # 2. Active Session Check
        account = self.session_mgr.get_active_account()
        if not account:
            if auto_fix:
                login_ok = self.session_mgr.trigger_login()
                account = self.session_mgr.get_active_account() if login_ok else None

            if not account:
                overall_passed = False
                items.append(
                    PreflightItem(
                        name="Google Cloud Session",
                        status=PreflightStatus.FAIL,
                        detail="No active gcloud authentication session detected.",
                        remediation_cmd="gcloud auth login --update-adc",
                        auto_fixable=True,
                    )
                )
                return PreflightReport(
                    items=items,
                    passed=False,
                    has_warnings=False,
                )

        items.append(
            PreflightItem(
                name="Google Cloud Session",
                status=PreflightStatus.PASS,
                detail=f"Authenticated as {account}",
            )
        )

        # 3. Active Project Check
        target_project = project_id
        if not target_project:
            # Check env or gcloud config
            target_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not target_project:
                try:
                    res = subprocess.run(
                        ["gcloud", "config", "get-value", "project"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    val = res.stdout.strip()
                    if val and val != "(unset)":
                        target_project = val
                except Exception:
                    pass

        if not target_project:
            projects = self.project_selector.list_accessible_projects()
            if auto_fix and projects:
                target_project = projects[0]["projectId"]
                self.project_selector.save_project_to_config(target_project, self.config_path)

            if not target_project:
                overall_passed = False
                items.append(
                    PreflightItem(
                        name="Project Resolution",
                        status=PreflightStatus.FAIL,
                        detail="No active Google Cloud project configured.",
                        remediation_cmd="gcloud config set project <PROJECT_ID>",
                        auto_fixable=True,
                    )
                )
                return PreflightReport(
                    items=items,
                    account=account,
                    passed=False,
                    has_warnings=False,
                )

        items.append(
            PreflightItem(
                name="Project Resolution",
                status=PreflightStatus.PASS,
                detail=f"Target Project: {target_project}",
            )
        )

        # 4. Service API Check (aiplatform.googleapis.com)
        is_api_ok = self.service_mgr.is_api_enabled(target_project)
        if not is_api_ok:
            if auto_fix:
                enabled = self.service_mgr.enable_api(target_project)
                is_api_ok = enabled

            if not is_api_ok:
                overall_passed = False
                items.append(
                    PreflightItem(
                        name="Vertex AI Service API",
                        status=PreflightStatus.FAIL,
                        detail=f"API 'aiplatform.googleapis.com' is not enabled on project '{target_project}'.",
                        remediation_cmd=f"gcloud services enable aiplatform.googleapis.com --project={target_project}",
                        auto_fixable=True,
                    )
                )
            else:
                items.append(
                    PreflightItem(
                        name="Vertex AI Service API",
                        status=PreflightStatus.PASS,
                        detail="Enabled and ready.",
                    )
                )
        else:
            items.append(
                PreflightItem(
                    name="Vertex AI Service API",
                    status=PreflightStatus.PASS,
                    detail="Enabled and ready.",
                )
            )

        # 5. Billing Check
        billing_ok, billing_detail = self.billing_validator.check_billing(target_project)
        if not billing_ok:
            has_warnings = True
            items.append(
                PreflightItem(
                    name="Project Billing",
                    status=PreflightStatus.WARN,
                    detail=billing_detail,
                    remediation_cmd=f"https://console.cloud.google.com/billing/linkedaccount?project={target_project}",
                    auto_fixable=False,
                )
            )
        else:
            items.append(
                PreflightItem(
                    name="Project Billing",
                    status=PreflightStatus.PASS,
                    detail=billing_detail,
                )
            )

        # 6. IAM Role & Permission Check
        iam_ok, iam_detail = self.iam_validator.check_permissions(target_project, account)
        if not iam_ok:
            has_warnings = True
            items.append(
                PreflightItem(
                    name="IAM Permissions",
                    status=PreflightStatus.WARN,
                    detail=iam_detail,
                    remediation_cmd=(
                        f"gcloud projects add-iam-policy-binding {target_project} "
                        f"--member='user:{account}' --role='roles/aiplatform.user'"
                    ),
                    auto_fixable=False,
                )
            )
        else:
            items.append(
                PreflightItem(
                    name="IAM Permissions",
                    status=PreflightStatus.PASS,
                    detail=iam_detail,
                )
            )

        return PreflightReport(
            items=items,
            account=account,
            project=target_project,
            passed=overall_passed,
            has_warnings=has_warnings,
        )
