# Wave 4: Autonomous Delivery & Production Workflows
## 01. Model Quota Circuit Breaker & Automatic Fallback (W4.1)

### 1. Problem Statement & Motivation

Autonomous engineering workflows—such as Wave 2's Hierarchical Multi-Agent Swarms (`Lead` $\rightarrow$ `Worker` $\rightarrow$ `QA`) and Self-Healing TDD loops—generate bursty, high-throughput token workloads. When interacting with Google Cloud Vertex AI and Partner Model Garden endpoints (Anthropic Claude, Google Gemini, xAI Grok, Mistral Codestral), production sessions frequently encounter two critical transient failure modes:

1. **Per-Minute Quota Exhaustion (`HTTP 429 RESOURCE_EXHAUSTED`)**: Triggered when Requests Per Minute (RPM) or Input/Output Tokens Per Minute (TPM / ITPM / OTPM) exceed regional project quotas.
2. **Regional Capacity Saturation (`HTTP 503 SERVICE_UNAVAILABLE`)**: Triggered when a specific GCP region (e.g., `us-east5` or `us-central1`) experiences temporary compute exhaustion or upstream gateway throttling (`OVERLOADED`).

Without an intelligent resilience layer, a single `429` or `503` response mid-turn aborts active tool dispatch chains, orphans uncommitted Git worktree states, and forces manual developer intervention. **W4.1 Model Quota Circuit Breaker & Automatic Fallback** guarantees zero-downtime inference continuity by isolating degraded `(model_id, region)` endpoints and transparently routing requests across a multi-tiered fallback chain in `< 200ms`.

---

### 2. Circuit Breaker State Machine (`CLOSED`, `OPEN`, `HALF_OPEN`)

Each unique `(model_id, region)` endpoint tuple is governed by an isolated, thread-safe and async-safe state machine.

```
                        Failure threshold reached
                    (e.g., HTTP 429 or consecutive 503s)
          +------------------------------------------------------+
          |                                                      |
          |                                                      v
   +-------------+                                        +-------------+
   |             |                                        |             |
   |   CLOSED    |                                        |    OPEN     |
   |  (Healthy)  |                                        |  (Tripped)  |
   |             |                                        |             |
   +-------------+                                        +-------------+
          ^                                                      |
          |                                                      | Cooldown timer expires
          | Probe succeeds                                       | (Retry-After or default 60s)
          | (HTTP 200 OK)                                        v
          |                                               +-------------+
          |                                               |             |
          +-----------------------------------------------+  HALF_OPEN  |
                                                          |  (Probing)  |
                                                          |             |
                                                          +-------------+
                                                                 |
                                                                 | Probe fails (HTTP 429 / 503)
                                                                 +-----> Re-enters OPEN
                                                                         (Exponential Backoff)
```

#### 2.1 State Definitions & Transition Rules

| State | Operational Behavior | Transition Out Condition |
| :--- | :--- | :--- |
| **`CLOSED`** | Normal routing state. Requests are dispatched directly to the `(model_id, region)` endpoint. Sliding-window failure counters track transient errors. | Transitions immediately to **`OPEN`** upon receiving an explicit quota exhaustion (`HTTP 429`) or when consecutive server unavailability errors (`HTTP 503`) reach `failure_threshold` (default: `2` within a `30s` sliding window). |
| **`OPEN`** | Fast-fail bypass state. Zero network calls are sent to this endpoint; the `FallbackRouter` immediately advances to the next healthy candidate in the fallback chain (`< 1ms` overhead). | Remains `OPEN` until `cooldown_until` timestamp elapses. The cooldown duration is dynamically computed from the `Retry-After` HTTP header (if present) or defaults to `base_cooldown_seconds` (`60.0s`). Once elapsed, transitions to **`HALF_OPEN`**. |
| **`HALF_OPEN`** | Controlled recovery probe state. Permits exactly **one** in-flight trial request (`max_probe_requests = 1`) to test upstream quota recovery while concurrent requests continue routing to fallback tiers. | - **On Success (`HTTP 200`)**: Resets failure counters, clears exponential backoff multiplier, and transitions back to **`CLOSED`**. <br>- **On Failure (`HTTP 429` / `503`)**: Immediately re-opens the circuit (**`OPEN`**) and doubles the cooldown window (`cooldown * 2^attempt`, capped at `300s`). |

---

### 3. Detection of HTTP 429 & HTTP 503 Errors

Vertex AI native Gemini endpoints (`:streamGenerateContent`), Anthropic Partner endpoints (`:streamRawPredict`), and OpenAI-compatible MaaS endpoints surface rate-limit and capacity errors through distinct HTTP headers and JSON payload structures.

#### 3.1 Quota Exhaustion (`HTTP 429 Too Many Requests`)
- **gRPC / Google Cloud Status**: `"status": "RESOURCE_EXHAUSTED"` or error message containing `"Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute_per_project_per_base_model"`.
- **Anthropic on Vertex Status**: `"type": "rate_limit_error"` inside the JSON error envelope.
- **Immediate Trip Policy**: Because a `429` indicates deterministic token/request bucket depletion for the current minute window, a **single `HTTP 429` immediately trips the breaker to `OPEN`** (`quota_trip_threshold = 1`) rather than wasting latency on futile immediate retries against the same endpoint.
- **`Retry-After` Header Extraction**:
  - Inspects `Retry-After` (standard integer seconds or HTTP-date) and `x-ratelimit-reset-requests` / `google.rpc.RetryInfo` metadata.
  - Adds a jitter buffer (`+ random.uniform(0.5, 2.0)` seconds) to prevent thundering-herd synchronization across multi-agent worker threads.

#### 3.2 Service Unavailability & Overloaded Capacity (`HTTP 503 Service Unavailable` / `HTTP 529`)
- **gRPC / Google Cloud Status**: `"status": "UNAVAILABLE"` or `"System overloaded"`.
- **Anthropic Partner Status**: `"type": "overloaded_error"` (often surfaced with `HTTP 503` or `HTTP 529`).
- **Mid-Stream SSE Interruption**: Detects error frames injected mid-stream in Server-Sent Events (SSE) before any irreversible tool execution side effects have occurred.
- **Sliding-Window Policy**: Allows 1 micro-retry with jittered backoff (`250ms`); if a second `503` occurs within `sliding_window_seconds` (`30s`), the circuit trips to `OPEN` and fails over to the next tier.

#### 3.3 Non-Tripping Errors (Excluded from Circuit Breaker)
Client-side semantic or authentication errors **never** trip the quota circuit breaker:
- `HTTP 400 Bad Request` (Invalid tool schema or malformed prompt)
- `HTTP 401 Unauthorized` / `HTTP 403 Permission Denied` (Handled by Wave 3 Auth & IAM Guards)
- `HTTP 404 Not Found` (Invalid model identifier)

---

### 4. Multi-Tier Fallback Chain (Tier 1 to Tier 3)

When a primary model endpoint trips to `OPEN`, the `FallbackRouter` traverses a deterministic, capability-preserving fallback hierarchy across three tiers:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          W4.1 Multi-Tier Fallback Topology                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Primary Flagship Model & Primary Region (Zero Capability Degradation)       │
│   • Target: User-configured flagship (e.g., `anthropic/claude-sonnet-4-6@us-east5`  │
│             or `google/gemini-2.5-pro@us-central1`)                                 │
│   • Action on 429/503: Trip (`model_id`, `primary_region`) -> Advance to Tier 1B    │
│   • Tier 1B (Cross-Region Same Model): Failover to secondary GCP region             │
│     (e.g., `us-east5` -> `us-central1` -> `europe-west1`) with independent quota.   │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │ All Tier 1 regions OPEN
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: Cross-Family Equivalent Reasoning Model (Peer Capability Failover)          │
│   • Target: Equivalent frontier coding model from alternate publisher family        │
│     (e.g., `anthropic/claude-sonnet-4-6` ──► `google/gemini-2.5-pro@us-central1`    │
│      or `google/gemini-2.5-pro` ──► `xai/grok-4.20-reasoning@us-central1`)          │
│   • Protocol Bridge: Normalizes system prompt, tool declarations, and message       │
│     history across Anthropic Messages API <-> Vertex Gemini `contents` format.      │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │ All Tier 2 endpoints OPEN
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: High-Quota Workhorse / Global Endpoint (Guaranteed Continuity Survivor)     │
│   • Target: Ultra-high-RPM/TPM model on global or multi-region routing              │
│     (e.g., `google/gemini-2.5-flash@global` or `google/gemini-3.8-flash@us-central1`)│
│   • Purpose: Ensures autonomous TDD repair loops and sub-agent tool dispatches      │
│     complete without crashing the session during severe regional outages.           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1 Tier Specification Matrix

| Tier | Tier Name | Example Endpoints | Quota Independence | Context & Tool Normalization |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Primary & Cross-Region Replica** | 1a. `anthropic/claude-sonnet-4-6@us-east5`<br>1b. `anthropic/claude-sonnet-4-6@us-central1`<br>1c. `anthropic/claude-sonnet-4-6@europe-west1` | High (Vertex AI enforces per-region quota buckets) | Identical publisher protocol; zero payload translation required. |
| **Tier 2** | **Cross-Publisher Peer Frontier** | 2a. `google/gemini-2.5-pro@us-central1`<br>2b. `google/gemini-2.5-pro@europe-west4` | Complete (Separate publisher & base model quota) | Lossless translation via `CrossFamilyPayloadNormalizer` (converts `tool_use`/`tool_result` $\leftrightarrow$ `functionCall`/`functionResponse`). |
| **Tier 3** | **High-Throughput Flash Survivor** | 3a. `google/gemini-2.5-flash@global`<br>3b. `google/gemini-2.5-flash@us-central1` | Maximum (Global pool + high TPM ceiling) | Adapts thinking budget (`thinking_budget` scaled to Flash limits) and preserves full tool schemas. |

---

### 5. Technical Interface: `QuotaCircuitBreaker` & `FallbackRouter`

The resilience subsystem resides in `src/cli_agent/resilience/circuit_breaker.py` and `src/cli_agent/resilience/fallback_router.py`.

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


class CircuitState(str, Enum):
    """Deterministic states for a (model_id, region) endpoint circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class ModelEndpoint:
    """Identifies a specific model deployment in a specific GCP region."""

    model_id: str  # e.g., "anthropic/claude-sonnet-4-6"
    region: str  # e.g., "us-east5", "us-central1", "global"
    tier: int  # 1 = Primary/Cross-Region, 2 = Peer Frontier, 3 = Flash Survivor

    @property
    def key(self) -> str:
        return f"{self.model_id}@{self.region}"


@dataclass
class CircuitBreakerConfig:
    """Configuration parameters governing trip thresholds and recovery timers."""

    failure_threshold_503: int = 2
    sliding_window_seconds: float = 30.0
    base_cooldown_seconds: float = 60.0
    max_cooldown_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter_Range_seconds: Tuple[float, float] = (0.5, 2.0)


@dataclass
class EndpointHealthSnapshot:
    """Read-only telemetry snapshot emitted to the W4.3 Split-Pane TUI HUD."""

    endpoint: ModelEndpoint
    state: CircuitState
    consecutive_failures: int
    cooldown_remaining_seconds: float
    last_status_code: Optional[int] = None
    last_error_reason: Optional[str] = None


class UpstreamQuotaError(Exception):
    """Raised when an endpoint returns HTTP 429 or HTTP 503."""

    def __init__(
        self,
        status_code: int,
        message: str,
        endpoint: ModelEndpoint,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.retry_after = retry_after


class ExhaustedFallbackChainError(RuntimeError):
    """Raised only when every endpoint across Tiers 1, 2, and 3 is OPEN."""

    def __init__(self, snapshots: Sequence[EndpointHealthSnapshot]) -> None:
        self.snapshots = list(snapshots)
        summary = ", ".join(
            f"{s.endpoint.key}={s.state.value}({s.cooldown_remaining_seconds:.1f}s)"
            for s in self.snapshots
        )
        super().__init__(f"All fallback model tiers exhausted: [{summary}]")


class QuotaCircuitBreaker:
    """
    Stateful circuit breaker managing health and cooldown transitions for
    individual (model_id, region) Vertex AI endpoints.
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CircuitBreakerConfig()
        self._clock = clock
        self._states: Dict[str, CircuitState] = {}
        self._failure_timestamps: Dict[str, List[float]] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._Half_open_probe_in_flight: Dict[str, bool] = {}
        self._consecutive_trips: Dict[str, int] = {}
        self._last_status: Dict[str, Tuple[int, str]] = {}

    def get_state(self, endpoint: ModelEndpoint) -> CircuitState:
        """Evaluates and returns the current state, auto-transitioning OPEN -> HALF_OPEN if cooldown expired."""
        ...

    def allow_request(self, endpoint: ModelEndpoint) -> bool:
        """
        Returns True if the endpoint is CLOSED, or if it just transitioned to
        HALF_OPEN and no other probe request is currently in flight.
        """
        ...

    def record_success(self, endpoint: ModelEndpoint) -> None:
        """
        Resets failure counters, clears exponential backoff multipliers,
        and transitions HALF_OPEN -> CLOSED.
        """
        ...

    def record_failure(
        self,
        endpoint: ModelEndpoint,
        status_code: int,
        error_body: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> CircuitState:
        """
        Inspects HTTP 429 / 503 errors, parses Retry-After headers, updates
        sliding-window counters, and transitions to OPEN when thresholds are met.
        """
        ...

    @staticmethod
    def parse_retry_after(
        headers: Optional[Dict[str, str]], error_body: str = ""
    ) -> Optional[float]:
        """Extracts explicit retry delay in seconds from HTTP headers or gRPC RetryInfo payload."""
        ...

    def get_health_snapshot(self, endpoint: ModelEndpoint) -> EndpointHealthSnapshot:
        """Returns current health metrics for TUI status bar and `/doctor` diagnostics."""
        ...


class FallbackRouter:
    """
    Orchestrates request execution across Tier 1 -> Tier 2 -> Tier 3 endpoints,
    automatically normalizing payloads when crossing model families.
    """

    def __init__(
        self,
        circuit_breaker: QuotaCircuitBreaker,
        fallback_chain: Sequence[ModelEndpoint],
        on_failover_callback: Optional[Callable[[ModelEndpoint, ModelEndpoint, int], None]] = None,
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.fallback_chain = list(fallback_chain)
        self.on_failover_callback = on_failover_callback

    def select_endpoint(self) -> ModelEndpoint:
        """
        Iterates through the configured Tier 1 -> Tier 3 chain and returns the
        first endpoint where `circuit_breaker.allow_request(endpoint)` is True.
        Raises `ExhaustedFallbackChainError` if all tiers are OPEN.
        """
        ...

    def execute_with_fallback(
        self,
        request_fn: Callable[[ModelEndpoint, Dict[str, Any]], Iterable[Dict[str, Any]]],
        canonical_payload: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        """
        Executes a streaming or unary LLM call against the highest-priority healthy
        endpoint. On HTTP 429 or 503 prior to token commitment, records failure,
        emits telemetry, normalizes the payload for the next target family, and
        transparently retries on the next available tier.
        """
        ...
```

---

### 6. Unit Testing Strategy (`tests/unit/resilience/`)

To guarantee deterministic `< 200ms` failover without relying on live GCP quotas or wall-clock `time.sleep` delays, all unit tests inject a controllable `FakeMonotonicClock` into `QuotaCircuitBreaker`.

#### 6.1 Core Test Suites & Scenarios

1. **`test_state_transitions_on_429_and_503.py` (State Machine & Thresholds)**
   - **Immediate Trip on `HTTP 429`**: Verify that a single `429 RESOURCE_EXHAUSTED` response on `claude-sonnet-4-6@us-east5` transitions the state from `CLOSED` $\rightarrow$ `OPEN` immediately.
   - **Sliding-Window `HTTP 503` Threshold**: Verify that a single `503` keeps the circuit `CLOSED`, a second `503` within `30s` trips it to `OPEN`, whereas a second `503` at `t = 35s` (outside the sliding window) does not trip the circuit.
   - **Non-Tripping Client Errors**: Verify that `HTTP 400`, `401`, `403`, and `404` exceptions propagate directly to the caller without incrementing failure counters or opening the circuit.

2. **`test_cooldown_and_half_open_probing.py` (Timers, Headers & Recovery)**
   - **`Retry-After` Header Compliance**: Pass `headers={"Retry-After": "45"}` on a `429` failure and assert `cooldown_remaining_seconds` is initialized to `45.0 + jitter` instead of the `60.0s` default.
   - **`OPEN` $\rightarrow$ `HALF_OPEN` Single-Probe Gate**: Advance `FakeMonotonicClock` past `cooldown_until`. Verify the first `allow_request()` returns `True` (transitions to `HALF_OPEN`), while a concurrent second `allow_request()` returns `False` while the probe is in flight.
   - **Probe Success vs. Probe Failure**:
     - When `record_success()` is called in `HALF_OPEN`, assert state returns to `CLOSED` and backoff multiplier resets to `1x`.
     - When `record_failure(429)` is called in `HALF_OPEN`, assert state returns to `OPEN` with doubled exponential cooldown (`120s`, capped at `300s`).

3. **`test_multi_tier_fallback_router.py` (Tier 1 $\rightarrow$ Tier 2 $\rightarrow$ Tier 3 Routing)**
   - **Cross-Region Failover (Tier 1a $\rightarrow$ Tier 1b)**: Simulate `429` on `claude-sonnet-4-6@us-east5` and assert `FallbackRouter` immediately dispatches to `claude-sonnet-4-6@us-central1` and invokes `on_failover_callback`.
   - **Cross-Family Failover & Payload Normalization (Tier 1 $\rightarrow$ Tier 2)**: Trip all Tier 1 Claude regions; assert router selects `google/gemini-2.5-pro@us-central1` and transforms Anthropic `tool_use` / `tool_result` blocks into Vertex Gemini `functionCall` / `functionResponse` parts without data loss.
   - **Cascade to Tier 3 & Total Exhaustion**: Trip Tiers 1 and 2; assert execution succeeds on Tier 3 (`google/gemini-2.5-flash@global`). Finally, trip Tier 3 and verify `ExhaustedFallbackChainError` is raised containing structured `EndpointHealthSnapshot` diagnostics for every tier.
