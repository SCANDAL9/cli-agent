"""
Unit tests for the Autonomous Self-Healing TDD Repair Loop (W2.6).
Verifies strict TDD lifecycle: RED reproduction, PATCH synthesis, GREEN verification,
NO_REGRESSION validation, rollback mechanics, and iteration limits.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cli_agent.tdd import (
    AutonomousTDDEngine,
    TDDLoopConfig,
    TDDPhase,
    TestFailureDetails,
    TestFailureParser,
)


@pytest.mark.unit
class TestTestFailureParser:
    def test_parse_single_pytest_assertion_failure(self) -> None:
        raw_output = """
=================================== FAILURES ===================================
____________________________ test_sample_addition _____________________________

    def test_sample_addition():
>       assert add(2, 2) == 5
E       assert 4 == 5

tests/unit/test_calc.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/unit/test_calc.py::test_sample_addition - assert 4 == 5
============================== 1 failed in 0.05s ===============================
"""
        failures = TestFailureParser.parse_pytest_output(raw_output)
        assert len(failures) == 1
        f = failures[0]
        assert f.test_name == "tests/unit/test_calc.py::test_sample_addition"
        assert f.error_type == "AssertionError"
        assert f.message == "assert 4 == 5"
        assert f.file_path == "tests/unit/test_calc.py"
        assert f.line_number == 12

    def test_parse_multiple_failures_with_custom_exceptions(self) -> None:
        raw_output = """
=================================== FAILURES ===================================
_______________________________ test_bad_config _______________________________
    def test_bad_config():
>       raise ValueError("Invalid configuration provided")
E       ValueError: Invalid configuration provided

tests/unit/test_config.py:45: ValueError
_______________________________ test_type_error _______________________________
    def test_type_error():
>       raise TypeError("Expected int got str")
E       TypeError: Expected int got str

tests/unit/test_calc.py:88: TypeError
=========================== short test summary info ============================
FAILED tests/unit/test_config.py::test_bad_config - ValueError: Invalid configuration provided
FAILED tests/unit/test_calc.py::test_type_error - TypeError: Expected int got str
============================== 2 failed in 0.12s ===============================
"""
        failures = TestFailureParser.parse_pytest_output(raw_output)
        assert len(failures) == 2
        assert failures[0].error_type == "ValueError"
        assert failures[0].message == "Invalid configuration provided"
        assert failures[0].line_number == 45

        assert failures[1].error_type == "TypeError"
        assert failures[1].message == "Expected int got str"
        assert failures[1].line_number == 88

    def test_parse_generic_execution_failure_fallback(self) -> None:
        raw_output = """
Traceback (most recent call last):
  File "src/main.py", line 10, in <module>
    raise RuntimeError("Fatal bootstrap failure")
RuntimeError: Fatal bootstrap failure
ERROR: Process exited with non-zero code.
"""
        failures = TestFailureParser.parse_pytest_output(raw_output)
        assert len(failures) == 1
        assert failures[0].test_name == "unknown_test"
        assert failures[0].error_type == "ExecutionError"

    def test_parse_empty_or_clean_output(self) -> None:
        raw_output = (
            "============================== 10 passed in 0.42s ==============================="
        )
        failures = TestFailureParser.parse_pytest_output(raw_output)
        assert len(failures) == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestAutonomousTDDEngine:
    async def test_red_phase_rejects_passing_test(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=3))

        events: list[tuple[str, str]] = []

        def on_event(evt: str, msg: str) -> None:
            events.append((evt, msg))

        # Mock run_command_async to return exit code 0 (test passed in RED phase)
        engine.run_command_async = AsyncMock(return_value=(0, "All tests passed", ""))  # type: ignore[method-assign]

        result = await engine.run_cycle(
            reproducing_command="pytest tests/unit/test_foo.py",
            on_event=on_event,
        )

        assert result.success is False
        assert result.iterations == 0
        assert TDDPhase.RED in result.phases_traversed
        assert "RED Phase Failed" in result.final_status
        assert any(e[0] == "red_failed" for e in events)

    async def test_full_successful_tdd_cycle(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=3))

        # Mock patch callable
        patch_mock = AsyncMock(return_value=True)
        engine.patch_fn = patch_mock

        # Setup mock command outputs:
        # Call 1 (RED phase reproducing test): fails (code 1)
        # Call 2 (GREEN phase reproducing test): passes (code 0)
        # Call 3 (NO_REGRESSION phase full suite): passes (code 0)
        reproducing_fail = """
=========================== short test summary info ============================
FAILED tests/unit/test_foo.py::test_reproduce - AssertionError: assert 1 == 2
============================== 1 failed in 0.05s ===============================
"""
        command_responses = [
            (1, reproducing_fail, ""),
            (0, "1 passed in 0.02s", ""),
            (0, "45 passed in 1.10s", ""),
        ]
        engine.run_command_async = AsyncMock(side_effect=command_responses)  # type: ignore[method-assign]
        engine.capture_git_diff = MagicMock(
            return_value="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-1\n+2"
        )  # type: ignore[method-assign]

        events: list[tuple[str, str]] = []
        result = await engine.run_cycle(
            reproducing_command="pytest tests/unit/test_foo.py::test_reproduce",
            regression_command="pytest",
            on_event=lambda evt, msg: events.append((evt, msg)),
        )

        assert result.success is True
        assert result.iterations == 1
        assert TDDPhase.RED in result.phases_traversed
        assert TDDPhase.PATCH in result.phases_traversed
        assert TDDPhase.GREEN in result.phases_traversed
        assert TDDPhase.NO_REGRESSION in result.phases_traversed
        assert TDDPhase.COMPLETE in result.phases_traversed
        assert result.diff is not None
        assert "foo.py" in result.diff
        assert patch_mock.await_count == 1
        assert any(e[0] == "tdd_complete" for e in events)

    async def test_patch_fails_on_first_attempt_and_succeeds_on_second(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=3))

        patch_mock = AsyncMock(return_value=True)
        engine.patch_fn = patch_mock

        # Call sequence:
        # 1. RED -> Fail
        # 2. Iter 1 GREEN -> Fail (still failing)
        # 3. Iter 2 GREEN -> Pass
        # 4. Iter 2 NO_REGRESSION -> Pass
        reproducing_fail = "FAILED tests/unit/test_foo.py::test_bar - AssertionError: fail"
        command_responses = [
            (1, reproducing_fail, ""),  # RED
            (1, reproducing_fail, ""),  # Iter 1 Green (failed)
            (0, "1 passed in 0.01s", ""),  # Iter 2 Green (passed)
            (0, "80 passed in 2.00s", ""),  # Iter 2 No-Regression (passed)
        ]
        engine.run_command_async = AsyncMock(side_effect=command_responses)  # type: ignore[method-assign]
        engine.capture_git_diff = MagicMock(return_value="patch diff")  # type: ignore[method-assign]

        result = await engine.run_cycle(
            reproducing_command="pytest tests/unit/test_foo.py::test_bar",
        )

        assert result.success is True
        assert result.iterations == 2
        assert patch_mock.await_count == 2

    async def test_regression_detected_triggers_rollback_and_continues(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=2, auto_rollback=True))

        rollback_mock = MagicMock(return_value=True)
        engine.rollback_git_changes = rollback_mock  # type: ignore[method-assign]
        engine.patch_fn = AsyncMock(return_value=True)

        reproducing_fail = "FAILED tests/unit/test_foo.py::test_bar - AssertionError"
        reg_fail = "FAILED tests/unit/test_existing.py::test_regression - AssertionError"

        # Sequence:
        # 1. RED -> Fail
        # 2. Iter 1 GREEN -> Pass
        # 3. Iter 1 NO_REGRESSION -> Fail (triggers rollback!)
        # 4. Iter 2 GREEN -> Pass
        # 5. Iter 2 NO_REGRESSION -> Pass
        command_responses = [
            (1, reproducing_fail, ""),
            (0, "1 passed in 0.01s", ""),
            (1, reg_fail, ""),
            (0, "1 passed in 0.01s", ""),
            (0, "80 passed in 2.00s", ""),
        ]
        engine.run_command_async = AsyncMock(side_effect=command_responses)  # type: ignore[method-assign]
        engine.capture_git_diff = MagicMock(return_value="clean diff")  # type: ignore[method-assign]

        events: list[tuple[str, str]] = []
        result = await engine.run_cycle(
            reproducing_command="pytest tests/unit/test_foo.py::test_bar",
            on_event=lambda evt, msg: events.append((evt, msg)),
        )

        assert result.success is True
        assert result.iterations == 2
        assert rollback_mock.call_count == 1
        assert any(e[0] == "regression_detected" for e in events)

    async def test_max_iterations_exhausted(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=2, auto_rollback=True))
        rollback_mock = MagicMock(return_value=True)
        engine.rollback_git_changes = rollback_mock  # type: ignore[method-assign]
        engine.patch_fn = AsyncMock(return_value=True)

        reproducing_fail = "FAILED tests/unit/test_foo.py::test_bar - AssertionError"
        # Always fail green verification
        command_responses = [
            (1, reproducing_fail, ""),  # RED
            (1, reproducing_fail, ""),  # Iter 1 Green
            (1, reproducing_fail, ""),  # Iter 2 Green
        ]
        engine.run_command_async = AsyncMock(side_effect=command_responses)  # type: ignore[method-assign]

        result = await engine.run_cycle(
            reproducing_command="pytest tests/unit/test_foo.py::test_bar",
        )

        assert result.success is False
        assert result.iterations == 2
        assert TDDPhase.FAILED in result.phases_traversed
        assert "Unable to satisfy" in result.final_status
        assert rollback_mock.call_count == 1

    async def test_git_operations_in_non_git_repo(self, tmp_path: Path) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(working_directory=tmp_path))
        diff = engine.capture_git_diff()
        assert diff == ""
        rolled_back = engine.rollback_git_changes()
        assert rolled_back is False

    async def test_patch_exception_handling(self) -> None:
        engine = AutonomousTDDEngine(config=TDDLoopConfig(max_iterations=1))

        async def failing_patch(fail: TestFailureDetails, it: int) -> bool:
            raise RuntimeError("Patch synthesis crash")

        engine.patch_fn = failing_patch
        engine.run_command_async = AsyncMock(return_value=(1, "FAILED test - AssertionError", ""))  # type: ignore[method-assign]
        engine.rollback_git_changes = MagicMock()  # type: ignore[method-assign]

        result = await engine.run_cycle(reproducing_command="pytest test")
        assert result.success is False
