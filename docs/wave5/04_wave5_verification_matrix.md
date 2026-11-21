# Wave 5: Unified Reactive Swarm & Semantic Runtime
## 04. Wave 5 Verification Matrix & ISO/IEC 25010 Quality Benchmark

### 1. Executive Quality & Traceability Overview

Wave 5 ("Unified Reactive Swarm & Semantic Runtime") elevates `cli-agent` into a concurrent, event-driven, semantically aware autonomous engineering platform. Every Wave 5 deliverable (W5.1 through W5.3) is governed by the **ISO/IEC 25010 Software Product Quality Model**, strict static typing (`mypy --strict`), zero-warning linting (`ruff check` & `ruff format --check`), and a hard **`>= 85%` branch coverage gate** enforced via `./scripts/verify.sh`.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Wave 5 Milestone Traceability Matrix                                   │
├──────────┬─────────────────────────────────────────────┬────────────────────────┬────────────────────────┤
│ ID       │ Subsystem & Architectural Scope             │ Implementation Modules │ Verification Suites    │
├──────────┼─────────────────────────────────────────────┼────────────────────────┼────────────────────────┤
│ **W5.1** │ Concurrent DAG Planner & Worktree Swarm     │ `team.py`, `worktree.py│ `test_swarm_dag.py`    │
│          │ (`ContextVar` Isolation & AST 3-Way Merge)  │ `tools.py`             │ `test_cu06_...py`      │
│          │                                             │                        │ `test_cu07_...py`      │
├──────────┼─────────────────────────────────────────────┼────────────────────────┼────────────────────────┤
│ **W5.2** │ Async Pub/Sub EventBus & 60 FPS TUI Bridge  │ `events.py`, `tui.py`  │ `test_events.py`       │
│          │ (Ring-Buffer Coalescing & 5 Core Topics)    │                        │ `test_cu08_...py`      │
├──────────┼─────────────────────────────────────────────┼────────────────────────┼────────────────────────┤
│ **W5.3** │ Headless LSP Semantic Diagnostic Bridge     │ `lsp.py`, `tools.py`   │ `test_lsp.py`          │
│          │ (JSON-RPC 2.0 Stdio & Sub-50ms Injection)   │                        │ `test_cu09_...py`      │
└──────────┴─────────────────────────────────────────────┴────────────────────────┴────────────────────────┘
```

---

### 2. ISO/IEC 25010 Quality Attribute Verification Matrix

| ISO/IEC 25010 Quality Characteristic | Sub-Characteristic | Wave 5 Architectural Mechanism | Quantitative Verification Proof / Assertion |
| :--- | :--- | :--- | :--- |
| **1. Functional Suitability** | Functional Completeness & Correctness | - Topological DAG scheduler (`SwarmDAGPlan`, `SwarmDAGNode`) with `asyncio.TaskGroup`.<br>- Per-worker `WorktreeSession` via `contextvars` + AST 3-way merge.<br>- 5-topic `EventBus` wired to 4-pane `TUIApp`.<br>- JSON-RPC 2.0 `HeadlessLSPClient` auto-injecting diagnostics into `write_file`/`edit_file`. | 100% pass rate across all unit tests (`test_swarm_dag.py`, `test_events.py`, `test_lsp.py`) and end-to-end integration journeys (`CU-06` through `CU-09`). |
| **2. Performance Efficiency** | Time Behaviour & Resource Utilization | - Bounded concurrency via `asyncio.Semaphore(4)`.<br>- $O(1)$ lock-free `deque(maxlen=4096)` event publishing (`< 50 µs`).<br>- 60 FPS (`16.67ms`) frame-paced coalescer eliminating redundant UI repaints.<br>- Warm stdio LSP server returning incremental diagnostics in `< 50ms`. | - 10,000-event burst benchmark completes in `< 50ms` with 0 event loop stalls.<br>- 500 rapid token chunks within `16ms` coalesce into 1 `TUIApp` frame render.<br>- Warm LSP diagnostic injection asserts `lsp_latency_ms < 50.0`. |
| **3. Compatibility** | Co-existence & Interoperability | - Standard LSP 3.17 JSON-RPC 2.0 `Content-Length` wire framing compatible with `pyright-langserver`, `basedpyright`, and `ruff server`.<br>- Standard Git worktree (`git worktree`, `git merge-file`) compatible with Git 2.25+. | Verified against both mock JSON-RPC 2.0 stdio servers and live `ruff` / Python AST fallback pipelines. |
| **4. Usability (Operability)** | User Error Protection & Observability | - Real-time DAG execution tree (`PENDING` $\rightarrow$ `RUNNING` $\rightarrow$ `COMPLETED`) rendered in `ChatPane`.<br>- Per-worktree unified diffs and LSP error counts displayed in `DiffViewerPane` and `StatusHUD`.<br>- Instant `Ctrl+C` (`CANCEL_SWARM`) structured task cancellation. | `VirtualScreenBuffer` assertions verify live rendering of DAG nodes, branch labels, LSP badges, and circuit health in `TUIApp`. |
| **5. Reliability** | Fault Tolerance & Recoverability | - Upstream DAG node failure automatically transitions dependent downstream nodes to `SKIPPED` without deadlocking `asyncio.TaskGroup`.<br>- Failed 3-way merge aborts cleanly, leaving the primary workspace 100% uncorrupted.<br>- Crashed LSP server subprocess automatically restarts or falls back to in-process `ast.parse()`. | Failure injection suites verify zero zombie worktrees, zero deadlocks on partial DAG failures, and seamless fallback when LSP binary crashes or times out. |
| **6. Security** | Integrity, Non-repudiation & Access Control | - Hard tool-level `_enforce_write_scope()` in `tools.py` enforces `allowed_files` ACL per `ContextVar` worker.<br>- Path traversal (`../../`) outside worktree root raises `ToolScopeViolationError`.<br>- Zero hardcoded personal usernames, emails, or GCP project IDs. | `CU-06` and unit tests assert `ToolScopeViolationError` when a worker attempts to edit an unapproved file, confirming 0 bytes mutated on disk. |
| **7. Maintainability** | Modularity, Reusability & Analysability | - Strict separation between `events.py` (telemetry broker), `lsp.py` (semantic protocol client), `team.py` (DAG orchestration), and `worktree.py` (Git/AST sandboxing).<br>- Full PEP 695 / `mypy --strict` type annotations with `@dataclass(slots=True)`. | `uv run mypy src tests` reports `0` issues; `uv run ruff check src tests` reports `0` warnings. |
| **8. Portability** | Adaptability & Installability | - Graceful degradation when external LSP binaries (`pyright-langserver`) or TTY alternate screens are absent.<br>- Temporary worktrees isolated under configurable `/tmp/cli-agent/worktrees`. | Headless CI tests pass in minimal Linux containers without X11, audio hardware, or global npm packages. |

---

### 3. Core End-to-End Integration Use Cases (CU-06 to CU-09)

Building upon Waves 1–4 Use Cases (`CU-01` through `CU-05`), Wave 5 establishes four new end-to-end integration test suites under `tests/integration/`:

#### 3.1 `CU-06`: Concurrent DAG Swarm with Isolated Per-Worker Git Worktrees & Hard `allowed_files` Enforcement
- **Test File**: `tests/integration/test_cu06_concurrent_dag_worktree_swarm.py`
- **Actors**: `LeadArchitect`, `DAGTaskScheduler`, `Worker-A`, `Worker-B`, `QA-Verifier`, `WorktreeManager`.
- **Scenario**:
  1. Initialize a real temporary Git repository (`tmp_path`) containing `module_a.py`, `module_b.py`, and `protected_config.py`.
  2. Construct a 3-node `SwarmDAGPlan`:
     - `node_a` (`WORKER`, `allowed_files=["module_a.py"]`, `dependencies=[]`)
     - `node_b` (`WORKER`, `allowed_files=["module_b.py"]`, `dependencies=[]`)
     - `node_qa` (`QA`, `allowed_files=[]`, `dependencies=["node_a", "node_b"]`)
  3. Execute `node_a` and `node_b` concurrently inside `asyncio.TaskGroup`.
  4. Inside `node_b`, simulate an unauthorized tool call `write_file("protected_config.py", ...)` followed by a valid `write_file("module_b.py", ...)`.
- **Verification Assertions**:
  - `node_a` and `node_b` run in distinct ephemeral Git worktree directories (`/tmp/cli-agent/worktrees/...`) simultaneously without `.git/index.lock` errors or `ContextVar` sandbox cross-talk.
  - The unauthorized edit to `protected_config.py` is blocked by `_enforce_write_scope()`, returning an `ALLOWED_FILES_ACL_VIOLATION` error to the worker while `protected_config.py` remains untouched on disk.
  - `node_qa` waits until both `node_a` and `node_b` complete, verifies the merged integration worktree, and atomic promotion updates both `module_a.py` and `module_b.py` in the main repository while cleaning up all temporary worktrees.

#### 3.2 `CU-07`: Lead Architect AST-Aware Three-Way Merge & Semantic Conflict Detection
- **Test File**: `tests/integration/test_cu07_ast_three_way_merge.py`
- **Actors**: `ASTThreeWayMerger`, `WorktreeSession`, `CodeGraphIndexer`.
- **Scenario A (Clean Disjoint AST Symbol Merge in Shared File)**:
  1. Base commit has `service.py` with `def compute_alpha(): ...` and `def compute_beta(): ...`.
  2. Concurrent `Worker-1` modifies `compute_alpha()` in `wt-1/service.py`.
  3. Concurrent `Worker-2` modifies `compute_beta()` and appends `def compute_gamma(): ...` in `wt-2/service.py`.
  4. `ASTThreeWayMerger.merge_worker_sessions()` reconciles `wt-1` and `wt-2` into `wt-integration`.
- **Scenario B (Overlapping AST Symbol Collision Escalation)**:
  1. Both `Worker-1` and `Worker-2` mutate the body of `def compute_alpha(): ...` in incompatible ways.
- **Verification Assertions**:
  - In Scenario A, `MergeResult.status` is `"clean"` or `"ast_resolved"`, the merged `service.py` contains both `Worker-1`'s `compute_alpha()`, `Worker-2`'s `compute_beta()`, and `compute_gamma()`, and `ast.parse()` succeeds with zero syntax errors.
  - In Scenario B, `MergeResult.status == "conflict"`, `MergeResult.conflicts` identifies `colliding_symbols=["compute_alpha"]`, and the main workspace remains uncorrupted.

#### 3.3 `CU-08`: High-Frequency Async EventBus Telemetry & 60 FPS Bi-Directional TUI Coalescing
- **Test File**: `tests/integration/test_cu08_event_bus_tui_coalescing.py`
- **Actors**: `EventBus`, `FrameCoalescer`, `TUIApp` (`ChatPane`, `DiffViewerPane`, `ProcessLogPane`, `StatusHUD`).
- **Scenario**:
  1. Connect `TUIApp` to `EventBus` via `TUIEventBusBridge`.
  2. Simulate a concurrent multi-agent telemetry storm within a single `16.67ms` frame window:
     - 100 `token.stream` chunks from `worker-1` and `worker-2`.
     - 6 rapid `swarm.node_state` transitions (`PENDING` $\rightarrow$ `READY` $\rightarrow$ `RUNNING` $\rightarrow$ `COMPLETED`).
     - 5 `worktree.diff_updated` updates for `src/cli_agent/events.py`.
     - 20 `subproc.log` lines from `TDD` and `LSP`.
     - 3 `circuit.health` transitions (`CLOSED` $\rightarrow$ `OPEN` $\rightarrow$ `HALF_OPEN`).
  3. Drain and render a single coalesced frame in `TUIApp`, then dispatch a reverse `tui.control` (`CANCEL_SWARM`) signal from `TUIApp`.
- **Verification Assertions**:
  - The 100 token chunks coalesce into 1 combined string per worker without dropping characters.
  - Only the final `swarm.node_state` (`COMPLETED`), final `worktree.diff_updated`, and final `circuit.health` (`HALF_OPEN`) are applied to `ChatPane`, `DiffViewerPane`, and `StatusHUD`.
  - All 20 `subproc.log` lines are preserved in exact chronological order in `ProcessLogPane`.
  - Reverse `TUIControlPayload(action="CANCEL_SWARM")` is received by the runtime subscriber in `< 5ms`.

#### 3.4 `CU-09`: Real-Time Headless LSP Diagnostic Injection & Zero-Test Self-Correction Loop
- **Test File**: `tests/integration/test_cu09_headless_lsp_self_healing.py`
- **Actors**: `VertexAgentClient` (or `FeatureWorker`), `edit_file` / `write_file` (`tools.py`), `HeadlessLSPManager` (`lsp.py`).
- **Scenario**:
  1. Turn 1: Agent calls `write_file("calculator.py", content="def add(a: int, b: int) -> int:\n    return a + 'bad_type'\n")` (or introduces a syntax/type error).
  2. `write_file` automatically synchronizes with `HeadlessLSPManager` (`didOpen` / `didChange`) and injects the sub-50ms `LSPDiagnostic` (`Line 2: Type/Syntax error`) directly into the tool response dictionary returned to the model.
  3. Turn 2: Seeing `lsp_status: "diagnostics_found"` and the exact line/column diagnostic in the tool output—**without invoking `run_command("pytest")`**—the model immediately calls `edit_file("calculator.py", ...)` to fix the defect.
  4. `edit_file` returns `lsp_status: "clean"`, `lsp_errors: 0`.
- **Verification Assertions**:
  - Turn 1 tool response includes `lsp_status == "diagnostics_found"`, `lsp_errors >= 1`, `lsp_latency_ms < 50.0`, and structured line/column diagnostic metadata.
  - Turn 2 tool response reports `lsp_status == "clean"` and `lsp_errors == 0`.
  - Zero `run_command("pytest")` subprocesses were needed to detect and repair the static defect.

---

### 4. Failure Injection & Chaos Resilience Scenarios

To verify system stability under adverse production conditions, the Wave 5 test harness includes deterministic failure injection scenarios:

| Scenario ID | Failure Mode Injected | Expected Resilient System Behavior | Verification Test |
| :--- | :--- | :--- | :--- |
| **FI-01** | **Cyclic Dependency Graph**: Lead Architect emits a `SwarmDAGPlan` containing a 3-node cycle (`A -> B -> C -> A`). | `DAGPlanner.validate_dag()` detects the cycle in `< 2ms` via Kahn's algorithm and raises `DAGCycleError` with the exact cycle path before spawning any worktrees. | `test_dag_cycle_detection_rejects_cyclic_graph` |
| **FI-02** | **Mid-DAG Worker Crash / Timeout**: Node `A` in `A -> B` raises an unhandled exception or exceeds `timeout_seconds`. | Node `A` transitions to `FAILED`, its ephemeral worktree is cleaned up, dependent Node `B` automatically transitions to `SKIPPED` without deadlocking `asyncio.TaskGroup`, and sibling independent Node `C` completes normally. | `test_dag_upstream_failure_skips_dependents_without_deadlock` |
| **FI-03** | **Ring-Buffer Subscriber Saturation**: Producer publishes `10,000` events while consumer is paused on a `maxlen=512` subscription. | `publish_nowait()` never blocks or raises `QueueFull`; oldest `9,488` events are evicted in $O(1)$, `dropped_count == 9488`, and the newest `512` events remain intact. | `test_event_bus_ring_buffer_drop_oldest_backpressure` |
| **FI-04** | **LSP Server Crash or Hang**: Language server stdio subprocess exits unexpectedly (`SIGKILL`) or stalls past the `50ms` diagnostic timeout. | `notify_and_collect_diagnostics()` catches `TimeoutError` / `IncompleteReadError`, returns fallback AST diagnostics within budget, and marks the server for automatic background restart without failing `edit_file`. | `test_lsp_client_timeout_and_crash_graceful_fallback` |
| **FI-05** | **Malicious Path Escape in Worker**: Worker passes `path="../../etc/hosts"` or a file outside `allowed_files` to `edit_file`. | `_enforce_write_scope()` raises `ToolScopeViolationError`, returns structured access-denied JSON to the LLM, and leaves the filesystem completely unmodified. | `test_worker_scope_blocks_path_traversal_and_unapproved_files` |

---

### 5. Unit & Integration Test Suite Architecture

#### 5.1 New Unit Test Suites (`tests/unit/`)
1. **`tests/unit/test_swarm_dag.py`**:
   - Topological sorting, Kahn's cycle detection, parallel tier grouping.
   - `asyncio.TaskGroup` + `asyncio.Semaphore` concurrency cap enforcement (verifying max active workers $\le K$).
   - `ContextVar` sandbox isolation across concurrent coroutines (`_active_sandbox_var`, `_allowed_files_var`).
   - Hard `_enforce_write_scope()` ACL enforcement and path traversal prevention.
   - `ASTThreeWayMerger` disjoint file merge, disjoint symbol merge in shared file, and overlapping symbol conflict detection.
2. **`tests/unit/test_events.py`**:
   - `EventBus` subscribe/unsubscribe, topic filtering, wildcard subscribers.
   - `RingBufferSubscription` bounded capacity and `dropped_count` accounting.
   - `coalesce_frame_events()` 60 FPS batch merging for `token.stream`, `swarm.node_state`, `worktree.diff_updated`, `subproc.log`, and `circuit.health`.
   - Bi-directional `TUIApp` $\leftrightarrow$ `EventBus` bridge updates across all 4 panes.
3. **`tests/unit/test_lsp.py`**:
   - JSON-RPC 2.0 `Content-Length` wire framing parser and serializer (including multi-byte UTF-8 payloads and fragmented stream reads).
   - `initialize` $\rightarrow$ `initialized` $\rightarrow$ `didOpen` $\rightarrow$ `didChange` $\rightarrow$ `publishDiagnostics` protocol state machine.
   - Sub-50ms timeout enforcement and fallback to Python `ast.parse()` syntax diagnostics.
   - Automatic diagnostic injection into `tools.write_file()` and `tools.edit_file()`.

#### 5.2 New Integration Test Suites (`tests/integration/`)
1. `tests/integration/test_cu06_concurrent_dag_worktree_swarm.py`
2. `tests/integration/test_cu07_ast_three_way_merge.py`
3. `tests/integration/test_cu08_event_bus_tui_coalescing.py`
4. `tests/integration/test_cu09_headless_lsp_self_healing.py`

---

### 6. Automated Quality Gate Execution (`./scripts/verify.sh`)

All Wave 5 code and specifications must pass the deterministic verification pipeline:

```bash
./scripts/verify.sh
```

| Gate Stage | Command | Acceptance Threshold |
| :--- | :--- | :--- |
| **1. Code Formatting** | `uv run ruff format --check src tests` | `100%` formatted (`0` diffs) |
| **2. Static Linting** | `uv run ruff check src tests` | `0` warnings / `0` errors |
| **3. Strict Type Checking** | `uv run mypy src tests` | `0` type errors under `strict = true` |
| **4. Unit & Integration Tests + Coverage** | `uv run pytest --cov=src/cli_agent --cov-fail-under=85` | `100%` tests passing, **`>= 85.0%` branch coverage** |
| **5. DevSecOps Privacy Scan** | Regex audit across `docs/wave5/`, `src/`, `tests/` | `0` personal usernames, emails, or real GCP project IDs |
