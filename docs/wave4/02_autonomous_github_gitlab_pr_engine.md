# Wave 4: Autonomous Delivery & Production Workflows
## 02. Autonomous GitHub & GitLab PR/MR Engine (W4.2)

### 1. Problem Statement & Motivation

Through Waves 1 to 3, `cli-agent` established a complete local engineering lifecycle: Tree-Sitter AST symbol graph indexing (W1.1), Hybrid RRF Code Search (W1.2), Hierarchical Multi-Agent Swarms with isolated Git Worktree sandboxes (`Lead` $\rightarrow$ `Worker` $\rightarrow$ `QA`, W2.1), Self-Healing TDD Repair Loops (W2.2), and Multi-Modal Vision Diagnostics (W3.2). However, once a local feature branch passes all local unit tests and QA verification inside its ephemeral Git worktree, a critical **last-mile delivery gap** remains:

1. **Manual Context Translation Overhead**: Developers must manually inspect multi-file diffs, reconstruct the architectural motivation, list modified AST symbols, and write structured Pull Request (PR) or Merge Request (MR) descriptions.
2. **Platform Fragmentation (GitHub vs. GitLab)**: Enterprise teams operate across heterogeneous Git hosting topologies—GitHub Cloud/Enterprise (`gh` CLI & REST API v3) and self-hosted or SaaS GitLab instances (`glab` CLI & REST API v4)—each with distinct authentication headers, URL remote formats, and pipeline state semantics.
3. **Remote CI/CD Environment Drift**: Code that passes local pytest suites frequently fails in remote CI runners due to matrix OS differences, strict linter gates, container build checks, or integration secrets. Without an autonomous remote pipeline loop, engineers are forced back into manual log scraping and fix-push cycles.

**W4.2 Autonomous GitHub & GitLab PR/MR Engine** bridges the verified local Git worktree directly to a production-ready, CI-validated remote Pull/Merge Request. It combines a unified `GitHostProvider` abstraction, an AST-aware semantic commit and description synthesizer aligned with **ISO/IEC 25010** quality attributes, and an automated **CI Watcher & Auto-Healer** that feeds remote runner failure logs directly back into the Wave 2 Self-Healing TDD engine.

---

### 2. `GitHostProvider` Abstraction Architecture

The hosting layer decouples upstream remote operations from platform-specific CLI tools and REST endpoints through a dual-transport strategy: **Native CLI Primary (`gh` / `glab`)** with automatic **Direct REST API Fallback (`urllib` / `httpx`)** when CLI binaries are absent in headless or containerized environments.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     Verified Local Git Worktree (Wave 2 Sandbox)                    │
│        • Clean working tree • Local pytest green • AST Symbol Graph indexed         │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          Remote Origin URL Auto-Detector                            │
│    Parses `git remote get-url origin` (SSH `git@host:ns/repo.git` or HTTPS URL)     │
└───────────────────┬──────────────────────────────────────────────┬──────────────────┘
                    │ Host matches github.*                        │ Host matches gitlab.*
                    ▼                                              ▼
┌──────────────────────────────────────────┐  ┌───────────────────────────────────────┐
│              GitHubProvider              │  │             GitLabProvider            │
│  • Primary: `gh pr create / view / ...`  │  │  • Primary: `glab mr create / view`   │
│  • Fallback: GitHub REST API v3          │  │  • Fallback: GitLab REST API v4       │
│    `/repos/{owner}/{repo}/pulls`         │  │    `/api/v4/projects/{id}/merge_...`  │
│  • Auth: `GH_TOKEN` / `GITHUB_TOKEN`     │  │  • Auth: `GITLAB_TOKEN` / `PRIVATE-..`│
└───────────────────┬──────────────────────┘  └────────────────────┬──────────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                   CIWatcher & Wave 2 Self-Healing TDD Feedback Loop                 │
│   Polls GitHub Actions Check Runs / GitLab CI Pipelines -> Extracts Failed Logs ->  │
│   Dispatches Wave 2 TDD Repair Agent -> Commits Fix -> Pushes -> Re-verifies Green  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1 Dual-Platform Capability Matrix

| Capability | `GitHubProvider` (GitHub Cloud & Enterprise) | `GitLabProvider` (GitLab SaaS & Self-Hosted) |
| :--- | :--- | :--- |
| **Remote URL Detection** | `github.com` or custom `GH_HOST` via SSH (`git@github.com:org/repo.git`) / HTTPS | `gitlab.com` or custom `GITLAB_HOST` supporting nested subgroup namespaces (`group/sub/repo`) |
| **CLI Transport (Primary)** | `gh pr create`, `gh pr view --json`, `gh run view --log-failed` | `glab mr create`, `glab mr view -F json`, `glab ci trace` |
| **REST Transport (Fallback)** | `POST /repos/{owner}/{repo}/pulls`<br>`GET /repos/{owner}/{repo}/commits/{sha}/check-runs` | `POST /api/v4/projects/{url_encoded_path}/merge_requests`<br>`GET /api/v4/projects/{id}/pipelines/{pipeline_id}/jobs` |
| **Authentication Resolution** | `GH_TOKEN` $\rightarrow$ `GITHUB_TOKEN` $\rightarrow$ `gh auth token` | `GITLAB_TOKEN` $\rightarrow$ `GLAB_TOKEN` $\rightarrow$ `glab auth status` |
| **Draft / WIP Support** | `--draft` CLI flag / `"draft": true` JSON field | `--draft` CLI flag / `"Draft: "` title prefix in API v4 |
| **Idempotency Guarantee** | Queries open PRs for `(head_branch, base_branch)` before creation; updates existing PR if found. | Queries open MRs with `source_branch` & `target_branch`; updates existing MR if found. |

---

### 3. Semantic Commit & AST-Aware PR Description Synthesizer

Rather than generating generic LLM summaries from raw unified diffs—which suffer from token bloat and hallucinated intent on large refactors—the `PRSynthesizer` constructs deterministic, structurally grounded commit messages and PR/MR bodies by combining **Wave 1 Tree-Sitter AST Symbol Diffs** with **Wave 2 QA Verification Telemetry**.

#### 3.1 AST Symbol Diff Extraction Pipeline

1. **Changed File Discovery**: Executes `git diff --name-status <base_branch>...HEAD` to classify files into `ADDED`, `MODIFIED`, `DELETED`, and `RENAMED`.
2. **Pre/Post AST Skeleton Comparison**:
   - Parses the base revision (`git show <base_branch>:<path>`) and head revision (`HEAD:<path>`) through `TreeSitterSymbolExtractor`.
   - Computes set operations over extracted `SymbolNode` signatures (classes, methods, functions, protocols):
     - **Added Symbols ($\Delta^+$)**: New public classes, functions, or methods introduced.
     - **Modified Signatures ($\Delta^\sim$)**: Functions whose parameters, return annotations, or decorators changed.
     - **Removed Symbols ($\Delta^-$)**: Deprecated or deleted definitions and their updated call-site counts via `CodeGraph`.
3. **Conventional Commit Synthesis**:
   - Automatically selects the semantic prefix (`feat`, `fix`, `refactor`, `perf`, `test`, `docs`) and scope from the primary module path and AST symbol delta.
   - Enforces the 72-character subject line limit and appends structured footers (`Refs:`, `Tested-By: cli-agent-qa`).

#### 3.2 Structured PR/MR Description Template & ISO/IEC 25010 Compliance

Every synthesized PR/MR body adheres to a strict four-section engineering specification:

```markdown
## 1. Executive Summary & Root Cause Analysis
- **Problem / Objective**: <Synthesized from task prompt and linked issue context>
- **Root Cause / Architectural Motivation**: <Pinpoints the exact module/symbol bottleneck or defect>

## 2. AST Symbol Delta Matrix (Tree-Sitter Verified)
| File Path | Change Type | Added Symbols (`+`) | Modified Symbols (`~`) | Removed Symbols (`-`) |
| :--- | :--- | :--- | :--- | :--- |
| `src/cli_agent/pr.py` | `ADDED` | `GitHostProvider`, `PRSynthesizer`, `CIWatcher` | — | — |

## 3. Verification & Test Execution Matrix
- **Unit & Integration Suite**: `18 passed, 0 failed` (`240ms`)
- **Self-Healing TDD Iterations**: `1 iteration` (0 regressions detected)
- **Coverage Delta**: `+4.2%` on touched modules

## 4. ISO/IEC 25010 Software Quality Compliance Checklist
- [x] **Functional Suitability**: All acceptance criteria and edge cases verified via automated tests.
- [x] **Reliability & Fault Tolerance**: Graceful fallback between CLI and REST transports; idempotent retries.
- [x] **Maintainability & Modularity**: Strict `Protocol` boundaries; zero circular dependencies in AST graph.
- [x] **Security**: Zero hardcoded credentials, tokens, or private project identifiers in diff.
- [x] **Performance Efficiency**: Sub-second AST diffing and bounded exponential polling intervals.
```

---

### 4. Automated Remote Pipeline Watcher & Auto-Healer

Creating a PR/MR is only half of autonomous delivery; `CIWatcher` closes the loop by monitoring remote GitHub Actions workflows or GitLab CI pipelines until terminal resolution and autonomously repairing remote failures.

```
                  ┌──────────────────────────────────────────┐
                  │   Push Branch & Create/Update PR or MR   │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │   Poll Remote CI Status (`poll_interval`)│◄───────────────┐
                  │   (GitHub Check Runs / GitLab Pipelines) │                │
                  └────────────────────┬─────────────────────┘                │
                                       │                                      │
              ┌────────────────────────┼────────────────────────┐             │
              │ PENDING / RUNNING      │ SUCCESS                │ FAILED      │
              ▼                        ▼                        ▼             │
     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐    │
     │ Backoff Wait    │      │ Mark PR Ready & │      │ Fetch Failed    │    │
     │ (Jittered Wait) │      │ Emit Green HUD  │      │ Job/Step Logs   │    │
     └────────┬────────┘      └─────────────────┘      └────────┬────────┘    │
              │                                                 │             │
              └────────────────────────┐                        ▼             │
                                       │               ┌─────────────────┐    │
                                       │               │ Sanitize ANSI & │    │
                                       │               │ Extract Trace   │    │
                                       │               └────────┬────────┘    │
                                       │                        │             │
                                       │                        ▼             │
                                       │               ┌─────────────────┐    │
                                       │               │ Wave 2 TDD      │    │
                                       │               │ Repair Engine   │    │
                                       │               │ (Max 3 Attempts)│    │
                                       │               └────────┬────────┘    │
                                       │                        │             │
                                       │                        ▼             │
                                       │               ┌─────────────────┐    │
                                       └──────────────►│ Commit & Push   │────┘
                                                       │ Healing Patch   │
                                                       └─────────────────┘
```

#### 4.1 Log Extraction, Sanitization & Wave 2 TDD Integration

1. **State Normalization**: Maps GitHub check statuses (`queued`, `in_progress`, `completed` + conclusions `success`, `failure`, `cancelled`, `timed_out`) and GitLab pipeline statuses (`created`, `pending`, `running`, `success`, `failed`, `canceled`) into a canonical `CIStatus` enum (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `CANCELLED`).
2. **Targeted Failure Log Retrieval**:
   - When `CIStatus.FAILED` is detected, `CIWatcher` queries only the failed job IDs.
   - Strips ANSI escape sequences, timestamps, and runner setup noise, extracting the tailing failure window (capped at `16,384` characters around pytest tracebacks, compiler errors, or linter violations).
3. **Closed-Loop Auto-Healing**:
   - Passes the structured `CIFailureReport` into Wave 2's `TDDExecutionEngine`.
   - The repair agent applies the surgical fix locally, runs the local verification command to confirm zero regressions, creates a semantic fixup commit (`fix(ci): resolve remote pipeline failure in <job_name>`), pushes to the remote branch, and resumes `CIWatcher` polling (bounded by `max_heal_attempts = 3`).

---

### 5. Technical Interface: `src/cli_agent/pr.py`

All classes, data structures, and protocols for the autonomous PR/MR engine are consolidated in `src/cli_agent/pr.py`.

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


class GitPlatform(str, Enum):
    """Supported remote Git hosting platforms."""

    GITHUB = "github"
    GITLAB = "gitlab"


class CIStatus(str, Enum):
    """Canonical normalized state for remote CI/CD pipelines and check suites."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class SymbolDelta:
    """AST symbol-level diff summary for a single modified file."""

    file_path: str
    change_type: str  # "ADDED", "MODIFIED", "DELETED", "RENAMED"
    added_symbols: Tuple[str, ...] = ()
    modified_symbols: Tuple[str, ...] = ()
    removed_symbols: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PRMetadata:
    """Normalized metadata representing a remote Pull Request or Merge Request."""

    platform: GitPlatform
    number: int
    title: str
    body: str
    url: str
    source_branch: str
    target_branch: str
    head_sha: str
    draft: bool = False
    labels: Tuple[str, ...] = ()
    ci_status: CIStatus = CIStatus.PENDING


@dataclass(frozen=True)
class CIFailureReport:
    """Structured diagnostic payload extracted from a failed remote CI job."""

    pipeline_id: str
    job_id: str
    job_name: str
    head_sha: str
    status: CIStatus
    error_summary: str
    sanitized_log_tail: str
    web_url: Optional[str] = None


@dataclass
class CIWatchResult:
    """Final outcome of a CIWatcher monitoring and auto-healing session."""

    pr_metadata: PRMetadata
    final_status: CIStatus
    heal_attempts_used: int = 0
    commit_shas_pushed: List[str] = field(default_factory=list)
    failure_reports: List[CIFailureReport] = field(default_factory=list)


@runtime_checkable
class GitHostProvider(Protocol):
    """
    Unified protocol abstracting GitHub (PR) and GitLab (MR) remote operations
    across CLI (`gh` / `glab`) and REST API v3 / v4 transports.
    """

    @property
    def platform(self) -> GitPlatform:
        """Returns the target platform identifier (GITHUB or GITLAB)."""
        ...

    def find_existing_pr(
        self,
        source_branch: str,
        target_branch: str,
    ) -> Optional[PRMetadata]:
        """Returns existing open PR/MR metadata for (source_branch, target_branch) if one exists."""
        ...

    def create_or_update_pr(
        self,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        labels: Sequence[str] = (),
    ) -> PRMetadata:
        """
        Idempotently creates a new PR/MR or updates the title/body of an existing
        open PR/MR for the given branch pair.
        """
        ...

    def get_ci_status(self, pr_number: int, head_sha: str) -> CIStatus:
        """Queries the aggregate status of all remote check runs / pipelines for `head_sha`."""
        ...

    def fetch_failed_ci_logs(self, pr_number: int, head_sha: str) -> List[CIFailureReport]:
        """Retrieves and sanitizes failure logs for all failed jobs associated with `head_sha`."""
        ...


class GitHubProvider:
    """
    GitHub implementation of `GitHostProvider` supporting `gh` CLI primary execution
    and GitHub REST API v3 fallback.
    """

    def __init__(
        self,
        repo_slug: str,
        worktree_path: Path,
        api_base_url: str = "https://api.github.com",
        token: Optional[str] = None,
        command_runner: Optional[Callable[[Sequence[str], Path], Tuple[int, str, str]]] = None,
        http_client: Optional[
            Callable[[str, str, Optional[Dict[str, Any]], Dict[str, str]], Tuple[int, str]]
        ] = None,
    ) -> None:
        self.repo_slug = repo_slug  # Format: "<owner>/<repo>"
        self.worktree_path = worktree_path
        self.api_base_url = api_base_url.rstrip("/")
        self._token = token
        self._command_runner = command_runner
        self._http_client = http_client

    @property
    def platform(self) -> GitPlatform:
        return GitPlatform.GITHUB

    def find_existing_pr(
        self,
        source_branch: str,
        target_branch: str,
    ) -> Optional[PRMetadata]:
        """Checks open GitHub Pull Requests via `gh pr list` or `GET /repos/{owner}/{repo}/pulls`."""
        ...

    def create_or_update_pr(
        self,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        labels: Sequence[str] = (),
    ) -> PRMetadata:
        """Creates or updates a GitHub Pull Request idempotently."""
        ...

    def get_ci_status(self, pr_number: int, head_sha: str) -> CIStatus:
        """Evaluates GitHub Actions check-runs and commit statuses for `head_sha`."""
        ...

    def fetch_failed_ci_logs(self, pr_number: int, head_sha: str) -> List[CIFailureReport]:
        """Extracts failed step logs via `gh run view --log-failed` or Actions Job Logs REST endpoint."""
        ...


class GitLabProvider:
    """
    GitLab implementation of `GitHostProvider` supporting `glab` CLI primary execution
    and GitLab REST API v4 fallback (including nested namespace URL encoding).
    """

    def __init__(
        self,
        project_path: str,
        worktree_path: Path,
        api_base_url: str = "https://gitlab.com/api/v4",
        token: Optional[str] = None,
        command_runner: Optional[Callable[[Sequence[str], Path], Tuple[int, str, str]]] = None,
        http_client: Optional[
            Callable[[str, str, Optional[Dict[str, Any]], Dict[str, str]], Tuple[int, str]]
        ] = None,
    ) -> None:
        self.project_path = project_path  # Format: "<namespace>/<project>" or nested "<group>/<sub-group>/<project>"
        self.worktree_path = worktree_path
        self.api_base_url = api_base_url.rstrip("/")
        self._token = token
        self._command_runner = command_runner
        self._http_client = http_client

    @property
    def platform(self) -> GitPlatform:
        return GitPlatform.GITLAB

    def find_existing_pr(
        self,
        source_branch: str,
        target_branch: str,
    ) -> Optional[PRMetadata]:
        """Checks open GitLab Merge Requests via `glab mr list` or `GET /projects/{id}/merge_requests`."""
        ...

    def create_or_update_pr(
        self,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        labels: Sequence[str] = (),
    ) -> PRMetadata:
        """Creates or updates a GitLab Merge Request idempotently."""
        ...

    def get_ci_status(self, pr_number: int, head_sha: str) -> CIStatus:
        """Evaluates GitLab CI pipeline status for `head_sha`."""
        ...

    def fetch_failed_ci_logs(self, pr_number: int, head_sha: str) -> List[CIFailureReport]:
        """Fetches failed job trace logs via `glab ci trace` or `GET /projects/{id}/jobs/{job_id}/trace`."""
        ...


class PRSynthesizer:
    """
    Synthesizes semantic Conventional Commit messages and AST-aware, ISO/IEC 25010
    compliant PR/MR descriptions from local Git diffs and Tree-Sitter symbol graphs.
    """

    def __init__(
        self,
        worktree_path: Path,
        git_runner: Optional[Callable[[Sequence[str], Path], Tuple[int, str, str]]] = None,
    ) -> None:
        self.worktree_path = worktree_path
        self._git_runner = git_runner

    def compute_symbol_deltas(
        self,
        base_branch: str,
        head_ref: str = "HEAD",
    ) -> List[SymbolDelta]:
        """
        Compares Tree-Sitter AST symbol outlines between `base_branch` and `head_ref`
        across all modified Python/source files in the worktree.
        """
        ...

    def synthesize_commit_message(
        self,
        task_objective: str,
        symbol_deltas: Sequence[SymbolDelta],
    ) -> str:
        """Generates a deterministic Conventional Commit message (`<type>(<scope>): <summary>`)."""
        ...

    def synthesize_pr_description(
        self,
        task_objective: str,
        root_cause_notes: str,
        symbol_deltas: Sequence[SymbolDelta],
        test_summary: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Returns `(pr_title, pr_markdown_body)` containing the Executive Summary,
        AST Symbol Delta Table, Test Matrix, and ISO/IEC 25010 compliance checklist.
        """
        ...


class CIWatcher:
    """
    Polls remote GitHub Actions or GitLab CI pipelines and orchestrates autonomous
    self-healing repair cycles via the Wave 2 TDD engine when remote checks fail.
    """

    def __init__(
        self,
        provider: GitHostProvider,
        tdd_healer_fn: Callable[[Sequence[CIFailureReport]], Optional[str]],
        poll_interval_seconds: float = 15.0,
        max_Wait_seconds: float = 900.0,
        max_heal_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.tdd_healer_fn = tdd_healer_fn
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_Wait_seconds
        self.max_heal_attempts = max_heal_attempts
        self._sleep = sleep_fn

    def watch_and_heal(
        self,
        pr_metadata: PRMetadata,
    ) -> CIWatchResult:
        """
        Continuously monitors remote CI status for `pr_metadata`. If status transitions
        to `FAILED`, fetches failed job logs, invokes `tdd_healer_fn` to generate, test,
        commit, and push a fix, and re-watches the new `head_sha` up to `max_heal_attempts`.
        """
        ...
```

---

### 6. Exhaustive Unit & Integration Testing Strategy (`tests/unit/test_pr.py`)

All unit and integration tests for W4.2 reside in `tests/unit/test_pr.py`. To guarantee **100% hermetic execution with zero real network calls or external CLI dependencies**, tests inject deterministic fakes into `command_runner`, `http_client`, `git_runner`, and `sleep_fn`.

#### 6.1 Mock & Fixture Architecture

- **`FakeSubprocessRunner`**: Intercepts `git`, `gh`, and `glab` argument vectors (`Sequence[str]`), validates flags, and returns canned `(exit_code, stdout, stderr)` tuples. Can simulate missing CLI binaries (`FileNotFoundError` / exit code `127`) to force REST API fallback paths.
- **`FakeHTTPClient`**: Captures `(method, url, json_payload, headers)` requests and returns synthetic GitHub REST v3 / GitLab REST v4 JSON payloads without opening any socket.
- **`NoOpSleepClock`**: Replaces `time.sleep` in `CIWatcher` so multi-minute polling loops execute in `< 5ms`.

#### 6.2 Core Test Scenarios in `tests/unit/test_pr.py`

1. **Remote URL Parsing & Platform Auto-Detection (`test_remote_origin_detection`)**
   - Verify SSH (`git@github.com:acme-org/core-engine.git`) and HTTPS (`https://github.com/acme-org/core-engine.git`) resolve to `GitHubProvider(repo_slug="acme-org/core-engine")`.
   - Verify GitLab nested subgroups (`git@gitlab.example.internal:platform/ai/agent-runtime.git`) resolve to `GitLabProvider(project_path="platform/ai/agent-runtime")` with URL-encoded REST path `platform%2Fai%2Fagent-runtime`.

2. **`GitHubProvider` CLI & REST Fallback (`test_github_provider_cli_and_rest_fallback`)**
   - **CLI Happy Path**: Mock `gh pr list` returning empty `[]`, followed by `gh pr create` returning a PR URL, and assert `PRMetadata` is populated accurately.
   - **Idempotent Update**: Mock `gh pr list` returning an existing open PR `#42` for the branch; assert `create_or_update_pr` invokes `gh pr edit 42` instead of creating a duplicate PR.
   - **REST v3 Fallback**: Simulate `gh` binary missing (`exit_code = 127`); assert `GitHubProvider` automatically falls back to `POST https://api.github.com/repos/acme-org/core-engine/pulls` with `Authorization: Bearer <token>`.

3. **`GitLabProvider` CLI & REST v4 Fallback (`test_gitlab_provider_cli_and_rest_v4_fallback`)**
   - **CLI Happy Path**: Verify `glab mr create --source-branch ... --target-branch ...` parses MR `!108` metadata.
   - **REST v4 Fallback & Draft Handling**: Simulate `glab` unavailable; assert fallback issues `POST /api/v4/projects/platform%2Fai%2Fagent-runtime/merge_requests` with `PRIVATE-TOKEN` header and prefixes `"Draft: "` when `draft=True`.

4. **AST Symbol Delta & ISO/IEC 25010 PR Synthesis (`test_pr_synthesizer_ast_deltas_and_iso_25010`)**
   - Feed two revisions of a Python module into `PRSynthesizer.compute_symbol_deltas` where a new class `CIWatcher` is added, `create_or_update_pr` signature is modified, and a legacy helper `_old_push` is deleted.
   - Assert `SymbolDelta` accurately categorizes `added_symbols=("CIWatcher",)`, `modified_symbols=("create_or_update_pr",)`, and `removed_symbols=("_old_push",)`.
   - Verify `synthesize_pr_description` outputs all four required sections, the markdown AST delta table, test execution counts, and all five ISO/IEC 25010 compliance checkboxes.

5. **`CIWatcher` Polling & Autonomous TDD Healing Loop (`test_ci_watcher_auto_heal_loop`)**
   - **Direct Success Path**: Simulate `get_ci_status` transitioning `PENDING` $\rightarrow$ `RUNNING` $\rightarrow$ `SUCCESS`; assert `watch_and_heal` returns `CIStatus.SUCCESS` with `heal_attempts_used == 0`.
   - **Failure $\rightarrow$ Heal $\rightarrow$ Green Recovery**:
     - Cycle 1 (`sha_1`): `get_ci_status` returns `CIStatus.FAILED`. `fetch_failed_ci_logs` returns a `CIFailureReport` containing ANSI-colored pytest assertion output.
     - Verify ANSI escape codes are stripped and `tdd_healer_fn` is called with the sanitized `CIFailureReport`, returning new commit `sha_2`.
     - Cycle 2 (`sha_2`): `get_ci_status` returns `CIStatus.SUCCESS`. Assert `CIWatchResult.final_status == CIStatus.SUCCESS`, `heal_attempts_used == 1`, and `commit_shas_pushed == ["sha_2"]`.
   - **Max Heal Attempts Cap**: Simulate persistent CI failure across 3 consecutive repair commits; assert `CIWatcher` halts cleanly after `max_heal_attempts == 3` and returns `CIStatus.FAILED` with all 3 historical `CIFailureReport` entries preserved for human inspection.
