"""
Autonomous Self-Healing TDD Repair Loop for cli-agent.
Adheres to strict Test-Driven Development (TDD) principles:
Phase 1: RED - Confirm reproducing failure.
Phase 2: PATCH - Formulate and apply minimal surgical edit.
Phase 3: GREEN - Verify reproducing test passes.
Phase 4: NO_REGRESSION - Run full regression test suite; rollback on regression.
"""

import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .worktree import _is_git_repo

logger = logging.getLogger(__name__)


class TDDPhase(StrEnum):
    RED = "RED"
    PATCH = "PATCH"
    GREEN = "GREEN"
    NO_REGRESSION = "NO_REGRESSION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class TestFailureDetails:
    """Represents structured failure information from test runner output."""

    __test__ = False

    test_name: str
    error_type: str = "AssertionError"
    message: str = ""
    file_path: str | None = None
    line_number: int | None = None
    traceback_lines: list[str] = field(default_factory=list)
    raw_output: str = ""


class TestFailureParser:
    """Parses pytest and standard test runner outputs into structured failure representations."""

    __test__ = False

    SUMMARY_PATTERN = re.compile(
        r"^FAILED\s+([^\s:]+(?:::[\w\-\[\]\.\<\>]+)+)(?:\s+-\s+(.*))?$",
        re.MULTILINE,
    )
    LOC_PATTERN = re.compile(
        r"([a-zA-Z0-9_\-\./\\]+\.py):(\d+):\s*([a-zA-Z0-9_]+Error|[a-zA-Z0-9_]+Exception|AssertionError|Exception)",
    )

    @classmethod
    def parse_pytest_output(cls, output: str) -> list[TestFailureDetails]:
        failures: list[TestFailureDetails] = []
        summary_matches = list(cls.SUMMARY_PATTERN.finditer(output))

        for match in summary_matches:
            test_id = match.group(1).strip()
            raw_msg = match.group(2).strip() if match.group(2) else ""

            error_type = "AssertionError"
            msg = raw_msg
            if ": " in raw_msg:
                parts = raw_msg.split(": ", 1)
                if parts[0].endswith("Error") or parts[0].endswith("Exception"):
                    error_type = parts[0]
                    msg = parts[1]

            file_path: str | None = None
            line_no: int | None = None
            if "::" in test_id:
                candidate_path = test_id.split("::", 1)[0]
                if candidate_path.endswith(".py"):
                    file_path = candidate_path

            # Attempt to locate line number in failure traceback
            for loc_match in cls.LOC_PATTERN.finditer(output):
                if loc_match.group(1) == file_path or (
                    file_path and file_path.endswith(loc_match.group(1))
                ):
                    try:
                        line_no = int(loc_match.group(2))
                        if loc_match.group(3):
                            error_type = loc_match.group(3)
                        break
                    except ValueError:
                        pass

            failures.append(
                TestFailureDetails(
                    test_name=test_id,
                    error_type=error_type,
                    message=msg,
                    file_path=file_path,
                    line_number=line_no,
                    raw_output=output[:4000],
                )
            )

        if not failures and ("FAILED" in output or "FAIL:" in output or "ERROR:" in output):
            loc_matches = list(cls.LOC_PATTERN.finditer(output))
            if loc_matches:
                m = loc_matches[0]
                failures.append(
                    TestFailureDetails(
                        test_name=m.group(1),
                        error_type=m.group(3),
                        message="Test failed during execution",
                        file_path=m.group(1),
                        line_number=int(m.group(2)),
                        raw_output=output[:4000],
                    )
                )
            else:
                failures.append(
                    TestFailureDetails(
                        test_name="unknown_test",
                        error_type="ExecutionError",
                        message="Test execution failed with non-zero exit code",
                        raw_output=output[:4000],
                    )
                )

        return failures


@dataclass
class TDDLoopConfig:
    """Configuration settings for the Autonomous TDD Engine."""

    max_iterations: int = 5
    reproducing_command: str = ""
    regression_command: str = "uv run pytest -q"
    timeout_seconds: int = 60
    auto_rollback: bool = True
    working_directory: Path | None = None


@dataclass
class TDDStepResult:
    """Record of a single phase execution step inside an iteration."""

    iteration: int
    phase: TDDPhase
    success: bool
    output: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TDDResult:
    """Final outcome of an autonomous TDD repair cycle."""

    success: bool
    iterations: int
    phases_traversed: list[TDDPhase]
    final_status: str
    initial_failures: list[TestFailureDetails] = field(default_factory=list)
    diff: str | None = None
    history: list[TDDStepResult] = field(default_factory=list)


PatchCallable = Callable[[TestFailureDetails, int], Awaitable[bool]]


class AutonomousTDDEngine:
    """
    Autonomous TDD Repair Loop Orchestrator.
    Guarantees that no patch is declared successful without:
    1. Validated failure reproduction (RED phase).
    2. Passing target test (GREEN phase).
    3. Zero regressions in the wider test suite (NO_REGRESSION phase).
    """

    def __init__(
        self,
        config: TDDLoopConfig | None = None,
        patch_fn: PatchCallable | None = None,
    ) -> None:
        self.config = config or TDDLoopConfig()
        self.patch_fn = patch_fn

    async def run_command_async(
        self,
        command: str,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """Runs a shell command asynchronously and returns (exit_code, stdout, stderr)."""
        work_dir = cwd or self.config.working_directory or Path.cwd()
        t = timeout or self.config.timeout_seconds

        proc = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=t,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def capture_git_diff(self, cwd: Path | None = None) -> str:
        """Captures unified git diff of current changes."""
        work_dir = cwd or self.config.working_directory or Path.cwd()
        if not _is_git_repo(work_dir):
            return ""

        try:
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout
        except Exception as e:
            logger.warning(f"Failed capturing git diff: {e}")
            return ""

    def rollback_git_changes(self, cwd: Path | None = None) -> bool:
        """Rolls back unstaged and staged changes using git restore/checkout."""
        work_dir = cwd or self.config.working_directory or Path.cwd()
        if not _is_git_repo(work_dir):
            return False

        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=work_dir,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=work_dir,
                capture_output=True,
                check=False,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed rolling back git changes: {e}")
            return False

    async def run_cycle(
        self,
        reproducing_command: str,
        regression_command: str | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> TDDResult:
        """
        Executes a complete autonomous TDD loop cycle.
        """
        phases_traversed: list[TDDPhase] = []
        history: list[TDDStepResult] = []
        reg_cmd = regression_command or self.config.regression_command

        def notify(evt_type: str, msg: str) -> None:
            if on_event:
                on_event(evt_type, msg)

        # -------------------------------------------------------------
        # Phase 1: RED - Confirm Reproduction
        # -------------------------------------------------------------
        phases_traversed.append(TDDPhase.RED)
        notify("phase_red", f"Executing reproducing command: {reproducing_command}")

        exit_code, stdout, stderr = await self.run_command_async(reproducing_command)
        combined_output = f"{stdout}\n{stderr}"

        if exit_code == 0:
            msg = (
                "RED Phase Failed: Test passed unexpectedly. "
                "TDD requires a confirmed failing test to reproduce the issue before modifying code."
            )
            notify("red_failed", msg)
            history.append(
                TDDStepResult(
                    iteration=0,
                    phase=TDDPhase.RED,
                    success=False,
                    output=combined_output,
                    details={"error": "Test passed in RED phase"},
                )
            )
            return TDDResult(
                success=False,
                iterations=0,
                phases_traversed=phases_traversed,
                final_status=msg,
                history=history,
            )

        failures = TestFailureParser.parse_pytest_output(combined_output)
        fail_summary = ", ".join(f"{f.test_name} ({f.error_type})" for f in failures)
        notify(
            "red_confirmed",
            f"✔ Failure reproduced: {fail_summary or 'Execution failure detected'}",
        )
        history.append(
            TDDStepResult(
                iteration=0,
                phase=TDDPhase.RED,
                success=True,
                output=combined_output,
                details={"failures": len(failures)},
            )
        )

        initial_failure = (
            failures[0]
            if failures
            else TestFailureDetails(
                test_name=reproducing_command,
                message=combined_output[:500],
            )
        )

        # -------------------------------------------------------------
        # Iteration Loop: PATCH -> GREEN -> NO_REGRESSION
        # -------------------------------------------------------------
        total_iterations = 0
        for i in range(1, self.config.max_iterations + 1):
            total_iterations = i
            notify(
                "iteration_start", f"Starting TDD repair iteration {i}/{self.config.max_iterations}"
            )

            # --- Phase 2: PATCH ---
            phases_traversed.append(TDDPhase.PATCH)
            notify("phase_patch", f"Iteration {i}: Synthesizing surgical fix...")

            patch_success = False
            if self.patch_fn:
                try:
                    patch_success = await self.patch_fn(initial_failure, i)
                except Exception as e:
                    logger.error(f"Error executing patch function: {e}")
                    patch_success = False
            else:
                patch_success = True

            history.append(
                TDDStepResult(
                    iteration=i,
                    phase=TDDPhase.PATCH,
                    success=patch_success,
                    output=f"Patch applied: {patch_success}",
                )
            )

            if not patch_success:
                notify("patch_failed", f"Iteration {i}: Failed applying patch. Retrying...")
                continue

            # --- Phase 3: GREEN ---
            phases_traversed.append(TDDPhase.GREEN)
            notify(
                "phase_green",
                f"Iteration {i}: Verifying reproducing test: {reproducing_command}",
            )
            g_exit, g_stdout, g_stderr = await self.run_command_async(reproducing_command)
            g_output = f"{g_stdout}\n{g_stderr}"

            if g_exit != 0:
                notify(
                    "green_failed",
                    f"Iteration {i}: Reproducing test still failing. Analyzing failure...",
                )
                history.append(
                    TDDStepResult(
                        iteration=i,
                        phase=TDDPhase.GREEN,
                        success=False,
                        output=g_output,
                    )
                )
                continue

            notify("green_passed", f"Iteration {i}: Target test is now GREEN! ✔")
            history.append(
                TDDStepResult(
                    iteration=i,
                    phase=TDDPhase.GREEN,
                    success=True,
                    output=g_output,
                )
            )

            # --- Phase 4: NO_REGRESSION ---
            phases_traversed.append(TDDPhase.NO_REGRESSION)
            notify("phase_regression", f"Iteration {i}: Running full regression suite: {reg_cmd}")

            r_exit, r_stdout, r_stderr = await self.run_command_async(reg_cmd)
            r_output = f"{r_stdout}\n{r_stderr}"

            if r_exit != 0:
                notify(
                    "regression_detected",
                    f"Iteration {i}: ⚠ Regression detected in regression test suite! Rolling back patch...",
                )
                if self.config.auto_rollback:
                    self.rollback_git_changes()

                history.append(
                    TDDStepResult(
                        iteration=i,
                        phase=TDDPhase.NO_REGRESSION,
                        success=False,
                        output=r_output,
                        details={"rolled_back": self.config.auto_rollback},
                    )
                )
                continue

            # Complete success! All tests green with no regressions!
            phases_traversed.append(TDDPhase.COMPLETE)
            diff_text = self.capture_git_diff()
            status_msg = (
                f"TDD repair cycle succeeded in iteration {i}! "
                "Target test passed and 100% of regression suite verified."
            )
            notify("tdd_complete", f"✔ {status_msg}")

            history.append(
                TDDStepResult(
                    iteration=i,
                    phase=TDDPhase.COMPLETE,
                    success=True,
                    output="All tests green with zero regressions.",
                    details={"diff_length": len(diff_text)},
                )
            )

            return TDDResult(
                success=True,
                iterations=i,
                phases_traversed=phases_traversed,
                final_status=status_msg,
                initial_failures=failures,
                diff=diff_text,
                history=history,
            )

        # Max iterations reached without complete success
        phases_traversed.append(TDDPhase.FAILED)
        fail_msg = (
            f"TDD repair failed after {self.config.max_iterations} iterations. "
            "Unable to satisfy both reproducing test and zero-regression criteria."
        )
        notify("tdd_failed", f"✖ {fail_msg}")
        if self.config.auto_rollback:
            self.rollback_git_changes()

        return TDDResult(
            success=False,
            iterations=total_iterations,
            phases_traversed=phases_traversed,
            final_status=fail_msg,
            initial_failures=failures,
            history=history,
        )
