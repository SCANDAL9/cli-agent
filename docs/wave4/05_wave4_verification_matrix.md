# Wave 4: Autonomous Delivery & Production Workflows
## 05. Wave 4 Verification Matrix & ISO/IEC 25010 Quality Benchmark

### 1. Executive Quality & Traceability Overview

Wave 4 establishes enterprise-grade autonomous delivery, operational resilience, spatial visualization, and multimodal interaction for `cli-agent`. Every milestone (W4.1 through W4.4) adheres strictly to the **ISO/IEC 25010 Software Product Quality Model**, strict static typing (`mypy --strict`), modern linting (`ruff`), and `>=85%` branch coverage enforcement.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Wave 4 Milestone Delivery Matrix                                │
├──────────┬───────────────────────────────────────────┬───────────────────────┬──────────────────┤
│ ID       │ Subsystem & Architectural Scope           │ Implementation Module │ Test Suite       │
├──────────┼───────────────────────────────────────────┼───────────────────────┼──────────────────┤
│ **W4.1** │ Model Quota Circuit Breaker & Fallback    │ `circuit_breaker.py`  │ `test_circuit...`│
│ **W4.2** │ Autonomous GitHub & GitLab PR Engine      │ `pr.py`               │ `test_pr.py`     │
│ **W4.3** │ Fullscreen Split-Pane TUI Workspace       │ `tui.py`              │ `test_tui.py`    │
│ **W4.4** │ Multimodal Live Voice Streaming Engine    │ `voice.py`            │ `test_voice.py`  │
└──────────┴───────────────────────────────────────────┴───────────────────────┴──────────────────┘
```

---

### 2. ISO/IEC 25010 Quality Attribute Verification Matrix

| ISO/IEC 25010 Attribute | Architectural Mechanism & Implementation | Verification Proof / Assertion |
| :--- | :--- | :--- |
| **Functional Suitability** | Complete coverage of multi-tier model failover (Tier 1 $\rightarrow$ Tier 3), dual-transport Git hosting (CLI/REST), 4-zone terminal viewports, and full-duplex WebSocket audio dialogue. | 38 dedicated unit tests across all 4 new test suites passing with 100% success. |
| **Performance Efficiency** | - `< 200ms` failover routing without network delay.<br>- Zero-allocation AST walks using Tree-Sitter.<br>- 60 FPS frame coalescing in TUI.<br>- Sub-20ms VAD frame energy processing. | Controllable monotonic clocks and in-memory frame rendering validate low-latency thresholds deterministically. |
| **Compatibility** | - Dual-transport `GitHostProvider` across GitHub Cloud/Enterprise and GitLab SaaS/Self-Hosted.<br>- Cross-family payload normalizer converting Anthropic Messages API $\leftrightarrow$ Vertex Gemini format.<br>- Headless degradation for environments without TTY or audio hardware. | Tests validate URL encoding, REST v3/v4 fallbacks, and dummy device graceful fallback. |
| **Reliability & Fault Tolerance** | - Stateful Quota Circuit Breaker (`CLOSED` $\rightarrow$ `OPEN` $\rightarrow$ `HALF_OPEN`) with exponential backoff and `Retry-After` jitter.<br>- Autonomous CI Watcher with bounded self-healing TDD loops (`max_heal_attempts = 3`). | Simulated HTTP 429 / 503 and remote CI failure recovery cycles verified in unit tests. |
| **Security & Privacy** | - Zero hardcoded project IDs, tokens, or usernames.<br>- Dynamic GCP project resolution.<br>- ANSI and secret sanitization on remote CI logs before ingestion. | Hardened `.gitignore`, sanitization regex tests, and clean security audit. |
| **Maintainability** | Clean Architecture separation: Domain entities, state machines, protocol abstractions, and presentation layers decoupled with strict protocols and type hints. | Strict `mypy` type checking passes with zero type errors. |

---

### 3. Automated Test Suites & Verification Execution

#### 3.1 Test Suite Breakdown
1. **`tests/unit/test_circuit_breaker.py`** (14 tests):
   - `test_initial_state_closed`
   - `test_immediate_trip_on_429`
   - `test_sliding_window_503_threshold`
   - `test_sliding_window_503_expired`
   - `test_non_tripping_client_errors`
   - `test_retry_after_header_parsing`
   - `test_retry_after_grpc_json_parsing`
   - `test_open_to_half_open_and_single_probe_gate`
   - `test_half_open_probe_success_recovery`
   - `test_half_open_probe_failure_exponential_backoff`
   - `test_fallback_router_cross_region_failover`
   - `test_fallback_router_exhaustion_raises_structured_error`
   - `test_cross_family_payload_normalizer_anthropic_to_gemini`
   - `test_cross_family_payload_normalizer_gemini_to_anthropic`

2. **`tests/unit/test_pr.py`** (12 tests):
   - `test_sanitize_ci_log_strips_ansi_and_preserves_tail`
   - `test_parse_origin_url_github_and_gitlab`
   - `test_detect_git_host_provider`
   - `test_github_provider_cli_create_and_find`
   - `test_github_provider_idempotent_update`
   - `test_github_provider_rest_fallback`
   - `test_github_ci_status_and_failed_logs`
   - `test_gitlab_provider_cli_and_rest_fallback`
   - `test_pr_synthesizer_symbol_deltas_and_iso_checklist`
   - `test_ci_watcher_direct_success`
   - `test_ci_watcher_heal_recovery_loop`
   - `test_ci_watcher_max_heal_attempts_cap`

3. **`tests/unit/test_tui.py`** (6 tests):
   - `test_layout_manager_responsive_breakpoints`
   - `test_chat_pane_streaming_and_swarm_tree`
   - `test_diff_viewer_pane_modes_and_file_cycling`
   - `test_process_log_pane_ring_buffer_and_filtering`
   - `test_status_hud_circuit_breaker_states`
   - `test_keyboard_dispatch_and_repl_toggle`

4. **`tests/unit/test_voice.py`** (6 tests):
   - `test_audio_format_config`
   - `test_vad_engine_silence_vs_speech`
   - `test_audio_device_manager_playback_and_purge`
   - `test_gemini_live_session_setup_and_audio_messages`
   - `test_gemini_live_session_parse_server_message`
   - `test_voice_agent_engine_barge_in_interruption`

---

### 4. Continuous Verification Protocol

Execute the full verification gate via:
```bash
./scripts/verify.sh
```
This runs:
1. `ruff check src tests` (Formatting & Linting)
2. `mypy --strict src` (Strict Static Type Analysis)
3. `pytest --cov=src/cli_agent --cov-fail-under=85` (Test execution & branch coverage gate)
