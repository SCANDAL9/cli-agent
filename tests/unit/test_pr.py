"""Unit tests for Autonomous GitHub & GitLab PR/MR Engine (Wave 4 Milestone 2)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cli_agent.pr import (
    CIFailureReport,
    CIStatus,
    CIWatcher,
    GitHubProvider,
    GitLabProvider,
    GitPlatform,
    PRMetadata,
    PRSynthesizer,
    SymbolDelta,
    detect_git_host_provider,
    parse_origin_url,
    sanitize_ci_log,
)


def test_sanitize_ci_log_strips_ansi_and_preserves_tail() -> None:
    ansi_log = "\x1b[31mFAIL: test_circuit_breaker.py\x1b[0m\nAssertionError: assert 1 == 2"
    cleaned = sanitize_ci_log(ansi_log)
    assert "\x1b" not in cleaned
    assert "FAIL: test_circuit_breaker.py" in cleaned
    assert "AssertionError: assert 1 == 2" in cleaned

    # Tail capping
    long_log = "a" * 20000 + "TAIL_DATA"
    capped = sanitize_ci_log(long_log, max_chars=100)
    assert len(capped) == 100
    assert capped.endswith("TAIL_DATA")


def test_parse_origin_url_github_and_gitlab() -> None:
    # GitHub SSH
    p, slug = parse_origin_url("git@github.com:acme-corp/cli-agent.git")
    assert p == GitPlatform.GITHUB
    assert slug == "acme-corp/cli-agent"

    # GitHub HTTPS
    p, slug = parse_origin_url("https://github.com/google/gemini-agent.git")
    assert p == GitPlatform.GITHUB
    assert slug == "google/gemini-agent"

    # GitLab SSH nested subgroup
    p, slug = parse_origin_url("git@gitlab.com:enterprise/infra/k8s-agent.git")
    assert p == GitPlatform.GITLAB
    assert slug == "enterprise/infra/k8s-agent"

    # GitLab custom host
    p, slug = parse_origin_url("https://gitlab.internal.corp/core/engine.git")
    assert p == GitPlatform.GITLAB
    assert slug == "core/engine"


def test_detect_git_host_provider(tmp_path: Path) -> None:
    def fake_git_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        return 0, "git@github.com:my-org/my-tool.git\n", ""

    provider = detect_git_host_provider(tmp_path, git_runner=fake_git_runner)
    assert isinstance(provider, GitHubProvider)
    assert provider.repo_slug == "my-org/my-tool"

    def fake_gitlab_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        return 0, "git@gitlab.com:group/sub/my-tool.git\n", ""

    provider_gl = detect_git_host_provider(tmp_path, git_runner=fake_gitlab_runner)
    assert isinstance(provider_gl, GitLabProvider)
    assert provider_gl.project_path == "group/sub/my-tool"
    assert provider_gl.encoded_project_path == "group%2Fsub%2Fmy-tool"


def test_github_provider_cli_create_and_find(tmp_path: Path) -> None:
    def fake_cmd_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        if "list" in args:
            return 0, "[]", ""
        if "create" in args:
            return 0, "https://github.com/acme/agent/pull/55\n", ""
        return 0, "", ""

    provider = GitHubProvider("acme/agent", tmp_path, command_runner=fake_cmd_runner)
    pr = provider.create_or_update_pr(
        title="feat(pr): implement engine",
        body="## Summary",
        source_branch="feat/w4-pr",
        target_branch="main",
        draft=True,
    )
    assert pr.platform == GitPlatform.GITHUB
    assert pr.number == 55
    assert pr.url == "https://github.com/acme/agent/pull/55"
    assert pr.draft is True


def test_github_provider_idempotent_update(tmp_path: Path) -> None:
    existing_json = json.dumps(
        [
            {
                "number": 42,
                "title": "Old title",
                "body": "Old body",
                "url": "https://github.com/acme/agent/pull/42",
                "headRefOid": "abc123sha",
                "isDraft": False,
            }
        ]
    )

    executed_commands: list[list[str]] = []

    def fake_cmd_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        executed_commands.append(list(args))
        if "list" in args:
            return 0, existing_json, ""
        if "edit" in args:
            return 0, "", ""
        return 0, "", ""

    provider = GitHubProvider("acme/agent", tmp_path, command_runner=fake_cmd_runner)
    pr = provider.create_or_update_pr(
        title="feat(core): updated title",
        body="updated body",
        source_branch="feat/w4-pr",
        target_branch="main",
    )
    assert pr.number == 42
    assert pr.title == "feat(core): updated title"
    # Ensure edit was called rather than create
    assert any("edit" in cmd for cmd in executed_commands)
    assert not any("create" in cmd for cmd in executed_commands)


def test_github_provider_rest_fallback(tmp_path: Path) -> None:
    def fake_cmd_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        return 127, "", "gh: command not found"

    def fake_http_client(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> tuple[int, str]:
        if method == "GET":
            return 200, "[]"
        if method == "POST":
            return 201, json.dumps(
                {
                    "number": 88,
                    "html_url": "https://github.com/acme/agent/pull/88",
                    "head": {"sha": "sha-888"},
                    "draft": False,
                }
            )
        return 404, "{}"

    provider = GitHubProvider(
        "acme/agent",
        tmp_path,
        command_runner=fake_cmd_runner,
        http_client=fake_http_client,
    )
    pr = provider.create_or_update_pr(
        title="feat(rest): fallback pr",
        body="via REST v3",
        source_branch="feat/rest",
        target_branch="main",
    )
    assert pr.number == 88
    assert pr.url == "https://github.com/acme/agent/pull/88"
    assert pr.head_sha == "sha-888"


def test_github_ci_status_and_failed_logs(tmp_path: Path) -> None:
    check_runs_data = json.dumps(
        {
            "check_runs": [
                {
                    "id": 1001,
                    "name": "pytest-suite",
                    "status": "completed",
                    "conclusion": "failure",
                    "html_url": "https://github.com/acme/agent/runs/1001",
                    "output": {
                        "title": "Pytest failed",
                        "text": "\x1b[31mFAILED test_circuit.py\x1b[0m",
                    },
                },
                {
                    "id": 1002,
                    "name": "linter",
                    "status": "completed",
                    "conclusion": "success",
                },
            ]
        }
    )

    def fake_http_client(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> tuple[int, str]:
        return 200, check_runs_data

    provider = GitHubProvider("acme/agent", tmp_path, http_client=fake_http_client)
    status = provider.get_ci_status(42, "head-sha-1")
    assert status == CIStatus.FAILED

    logs = provider.fetch_failed_ci_logs(42, "head-sha-1")
    assert len(logs) == 1
    assert logs[0].job_name == "pytest-suite"
    assert logs[0].status == CIStatus.FAILED
    assert "FAILED test_circuit.py" in logs[0].sanitized_log_tail
    assert "\x1b" not in logs[0].sanitized_log_tail


def test_gitlab_provider_cli_and_rest_fallback(tmp_path: Path) -> None:
    # CLI Missing -> Fallback to REST v4
    def fake_cmd_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        return 127, "", "glab: command not found"

    def fake_http_client(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> tuple[int, str]:
        if method == "GET" and "merge_requests" in url:
            return 200, "[]"
        if method == "POST" and "merge_requests" in url:
            assert payload is not None
            assert payload["title"].startswith("Draft: ")
            return 201, json.dumps(
                {
                    "iid": 108,
                    "web_url": "https://gitlab.com/group/repo/-/merge_requests/108",
                    "sha": "gitlab-sha-108",
                }
            )
        return 404, "{}"

    provider = GitLabProvider(
        "group/repo",
        tmp_path,
        command_runner=fake_cmd_runner,
        http_client=fake_http_client,
    )
    mr = provider.create_or_update_pr(
        title="feat(gitlab): new MR",
        body="description",
        source_branch="feat/gl",
        target_branch="main",
        draft=True,
    )
    assert mr.platform == GitPlatform.GITLAB
    assert mr.number == 108
    assert mr.title == "Draft: feat(gitlab): new MR"
    assert mr.url == "https://gitlab.com/group/repo/-/merge_requests/108"


def test_pr_synthesizer_symbol_deltas_and_iso_checklist(tmp_path: Path) -> None:
    diff_output = "M\tsrc/cli_agent/circuit_breaker.py\nA\tsrc/cli_agent/pr.py\n"

    def fake_git_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
        return 0, diff_output, ""

    synthesizer = PRSynthesizer(tmp_path, git_runner=fake_git_runner)
    deltas = synthesizer.compute_symbol_deltas("main", "HEAD")
    assert len(deltas) == 2
    assert deltas[0].change_type == "MODIFIED"
    assert deltas[1].change_type == "ADDED"

    custom_delta = [
        SymbolDelta(
            file_path="src/cli_agent/pr.py",
            change_type="ADDED",
            added_symbols=("GitHostProvider", "PRSynthesizer", "CIWatcher"),
        )
    ]
    commit_msg = synthesizer.synthesize_commit_message(
        "implement autonomous PR engine", custom_delta
    )
    assert commit_msg == "feat(cli_agent): autonomous PR engine"

    title, body = synthesizer.synthesize_pr_description(
        task_objective="Implement autonomous PR engine",
        root_cause_notes="Bridge verified worktree to remote PR",
        symbol_deltas=custom_delta,
        test_summary={"passed": 20, "failed": 0, "duration_ms": 120, "iterations": 1},
    )
    assert title == "feat(cli_agent): autonomous PR engine"
    assert "## 1. Executive Summary & Root Cause Analysis" in body
    assert "## 2. AST Symbol Delta Matrix (Tree-Sitter Verified)" in body
    assert "`GitHostProvider`" in body
    assert "## 3. Verification & Test Execution Matrix" in body
    assert "`20 passed, 0 failed`" in body
    assert "## 4. ISO/IEC 25010 Software Quality Compliance Checklist" in body
    assert "- [x] **Functional Suitability**" in body
    assert "- [x] **Reliability**" in body


def test_ci_watcher_direct_success() -> None:
    provider = MagicMock(spec=GitHubProvider)
    provider.get_ci_status.return_value = CIStatus.SUCCESS

    healer = MagicMock()
    watcher = CIWatcher(provider, healer, poll_interval_seconds=0.01, sleep_fn=lambda _: None)

    metadata = PRMetadata(
        platform=GitPlatform.GITHUB,
        number=1,
        title="feat",
        body="body",
        url="url",
        source_branch="b1",
        target_branch="main",
        head_sha="sha-initial",
    )

    result = watcher.watch_and_heal(metadata)
    assert result.final_status == CIStatus.SUCCESS
    assert result.heal_attempts_used == 0
    healer.assert_not_called()


def test_ci_watcher_heal_recovery_loop() -> None:
    provider = MagicMock(spec=GitHubProvider)
    # Status transitions: FAILED on sha-1 -> SUCCESS on sha-2
    provider.get_ci_status.side_effect = [
        CIStatus.FAILED,
        CIStatus.SUCCESS,
    ]
    failure_report = CIFailureReport(
        pipeline_id="p1",
        job_id="j1",
        job_name="pytest",
        head_sha="sha-1",
        status=CIStatus.FAILED,
        error_summary="pytest failure",
        sanitized_log_tail="assert 1 == 2",
    )
    provider.fetch_failed_ci_logs.return_value = [failure_report]

    healer = MagicMock(return_value="sha-2-fixed")
    watcher = CIWatcher(provider, healer, poll_interval_seconds=0.01, sleep_fn=lambda _: None)

    metadata = PRMetadata(
        platform=GitPlatform.GITHUB,
        number=10,
        title="feat",
        body="body",
        url="url",
        source_branch="b1",
        target_branch="main",
        head_sha="sha-1",
    )

    result = watcher.watch_and_heal(metadata)
    assert result.final_status == CIStatus.SUCCESS
    assert result.heal_attempts_used == 1
    assert result.commit_shas_pushed == ["sha-2-fixed"]
    healer.assert_called_once_with([failure_report])


def test_ci_watcher_max_heal_attempts_cap() -> None:
    provider = MagicMock(spec=GitHubProvider)
    provider.get_ci_status.return_value = CIStatus.FAILED
    provider.fetch_failed_ci_logs.return_value = [
        CIFailureReport(
            pipeline_id="p1",
            job_id="j1",
            job_name="pytest",
            head_sha="sha-current",
            status=CIStatus.FAILED,
            error_summary="failure",
            sanitized_log_tail="log",
        )
    ]

    healer = MagicMock(side_effect=["sha-fix-1", "sha-fix-2", "sha-fix-3", "sha-fix-4"])
    watcher = CIWatcher(
        provider,
        healer,
        max_heal_attempts=3,
        poll_interval_seconds=0.01,
        sleep_fn=lambda _: None,
    )

    metadata = PRMetadata(
        platform=GitPlatform.GITHUB,
        number=10,
        title="feat",
        body="body",
        url="url",
        source_branch="b1",
        target_branch="main",
        head_sha="sha-0",
    )

    result = watcher.watch_and_heal(metadata)
    assert result.final_status == CIStatus.FAILED
    assert result.heal_attempts_used == 3
    assert len(result.commit_shas_pushed) == 3


def test_default_subprocess_runner_real(tmp_path: Path) -> None:
    from cli_agent.pr import default_subprocess_runner

    code, out, err = default_subprocess_runner(["python", "-c", "print('hello_pr')"], tmp_path)
    assert code == 0
    assert "hello_pr" in out

    code_missing, _, err_missing = default_subprocess_runner(
        ["non_existent_binary_xyz123"], tmp_path
    )
    assert code_missing == 127
    assert "Command not found" in err_missing


def test_gitlab_ci_status_and_failed_logs(tmp_path: Path) -> None:
    pipelines_data = json.dumps([{"id": 4040, "status": "failed", "sha": "sha-gl-failed"}])
    jobs_data = json.dumps(
        [
            {
                "id": 9090,
                "name": "build_and_test",
                "status": "failed",
                "web_url": "https://gitlab.com/job/9090",
            }
        ]
    )

    def fake_http(method: str, url: str, payload: Any, headers: Any) -> tuple[int, str]:
        if "pipelines" in url and "jobs" not in url:
            return 200, pipelines_data
        if "jobs" in url and "trace" not in url:
            return 200, jobs_data
        if "trace" in url:
            return 200, "\x1b[31mFAILED in pipeline trace\x1b[0m"
        return 404, "{}"

    provider = GitLabProvider("enterprise/repo", tmp_path, http_client=fake_http)
    status = provider.get_ci_status(1, "sha-gl-failed")
    assert status == CIStatus.FAILED

    logs = provider.fetch_failed_ci_logs(1, "sha-gl-failed")
    assert len(logs) == 1
    assert logs[0].job_name == "build_and_test"
    assert "FAILED in pipeline trace" in logs[0].sanitized_log_tail


def test_gitlab_ci_statuses_mapping(tmp_path: Path) -> None:
    for status_str, expected in [
        ("running", CIStatus.RUNNING),
        ("pending", CIStatus.PENDING),
        ("canceled", CIStatus.CANCELLED),
        ("success", CIStatus.SUCCESS),
    ]:

        def fake_http(m: str, u: str, p: Any, h: Any, s: str = status_str) -> tuple[int, str]:
            return 200, json.dumps([{"id": 1, "status": s}])

        provider = GitLabProvider("repo", tmp_path, http_client=fake_http)
        assert provider.get_ci_status(1, "sha") == expected


def test_ci_watcher_cancelled_and_timeout() -> None:
    provider = MagicMock(spec=GitHubProvider)
    provider.get_ci_status.return_value = CIStatus.CANCELLED

    watcher = CIWatcher(
        provider, lambda _: None, poll_interval_seconds=0.01, sleep_fn=lambda _: None
    )
    meta = PRMetadata(
        platform=GitPlatform.GITHUB,
        number=1,
        title="t",
        body="b",
        url="u",
        source_branch="s",
        target_branch="main",
        head_sha="sha",
    )
    res_cancel = watcher.watch_and_heal(meta)
    assert res_cancel.final_status == CIStatus.CANCELLED

    # Timeout
    provider_timeout = MagicMock(spec=GitHubProvider)
    provider_timeout.get_ci_status.return_value = CIStatus.RUNNING
    watcher_timeout = CIWatcher(
        provider_timeout,
        lambda _: None,
        max_wait_seconds=0.01,
        poll_interval_seconds=0.02,
        sleep_fn=lambda _: None,
    )
    res_timeout = watcher_timeout.watch_and_heal(meta)
    assert res_timeout.final_status == CIStatus.RUNNING


def test_pr_creation_failure_raises(tmp_path: Path) -> None:
    def fake_cmd_missing(args: Any, cwd: Any) -> tuple[int, str, str]:
        return 127, "", ""

    def fake_http_err(m: Any, u: Any, p: Any, h: Any) -> tuple[int, str]:
        if m == "GET":
            return 200, "[]"
        return 500, "Internal Server Error"

    gh = GitHubProvider(
        "org/repo", tmp_path, command_runner=fake_cmd_missing, http_client=fake_http_err
    )
    with pytest.raises(RuntimeError, match="Failed to create GitHub PR"):
        gh.create_or_update_pr("t", "b", "s", "m")

    gl = GitLabProvider(
        "org/repo", tmp_path, command_runner=fake_cmd_missing, http_client=fake_http_err
    )
    with pytest.raises(RuntimeError, match="Failed to create GitLab MR"):
        gl.create_or_update_pr("t", "b", "s", "m")
