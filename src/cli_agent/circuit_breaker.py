"""
Model Quota Circuit Breaker & Multi-Tier Fallback Router (Wave 4 Milestone 1).

Provides zero-downtime inference continuity on Google Cloud Vertex AI by
isolating degraded (model_id, region) endpoints on HTTP 429 / 503 and
transparently failing over across Tier 1 (Cross-Region) -> Tier 2 (Frontier Peer) ->
Tier 3 (High-Throughput Flash Survivor).
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any


class CircuitState(StrEnum):
    """Deterministic states for an individual (model_id, region) endpoint circuit."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class ModelEndpoint:
    """Identifies a specific model deployment in a specific GCP region."""

    model_id: str
    region: str
    tier: int = 1

    @property
    def key(self) -> str:
        return f"{self.model_id}@{self.region}"


@dataclass
class CircuitBreakerConfig:
    """Configuration governing failure thresholds, sliding windows, and cooldown timers."""

    failure_threshold_503: int = 2
    sliding_window_seconds: float = 30.0
    base_cooldown_seconds: float = 60.0
    max_cooldown_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter_range_seconds: tuple[float, float] = (0.5, 2.0)


@dataclass
class EndpointHealthSnapshot:
    """Telemetry snapshot emitted to TUI HUD and /doctor diagnostics."""

    endpoint: ModelEndpoint
    state: CircuitState
    consecutive_failures: int
    cooldown_remaining_seconds: float
    last_status_code: int | None = None
    last_error_reason: str | None = None


class UpstreamQuotaError(Exception):
    """Raised when an upstream model endpoint returns HTTP 429 or 503."""

    def __init__(
        self,
        status_code: int,
        message: str,
        endpoint: ModelEndpoint,
        retry_after: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.retry_after = retry_after
        self.headers = headers or {}


class ExhaustedFallbackChainError(RuntimeError):
    """Raised when all configured endpoints across all fallback tiers are in OPEN state."""

    def __init__(self, snapshots: Sequence[EndpointHealthSnapshot]) -> None:
        self.snapshots = list(snapshots)
        summary = ", ".join(
            f"{s.endpoint.key}={s.state.value}({s.cooldown_remaining_seconds:.1f}s)"
            for s in self.snapshots
        )
        super().__init__(f"All fallback model tiers exhausted: [{summary}]")


class QuotaCircuitBreaker:
    """Stateful circuit breaker managing health and cooldown transitions for

    individual (model_id, region) endpoints.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CircuitBreakerConfig()
        self._clock = clock
        self._states: dict[str, CircuitState] = {}
        self._failure_timestamps: dict[str, list[float]] = {}
        self._cooldown_until: dict[str, float] = {}
        self._half_open_probe_in_flight: dict[str, bool] = {}
        self._consecutive_trips: dict[str, int] = {}
        self._last_status: dict[str, tuple[int, str]] = {}

    def get_state(self, endpoint: ModelEndpoint) -> CircuitState:
        """Returns the current state, auto-transitioning OPEN -> HALF_OPEN when cooldown elapses."""
        key = endpoint.key
        state = self._states.get(key, CircuitState.CLOSED)
        if state == CircuitState.OPEN:
            cooldown_until = self._cooldown_until.get(key, 0.0)
            if self._clock() >= cooldown_until:
                state = CircuitState.HALF_OPEN
                self._states[key] = state
                self._half_open_probe_in_flight[key] = False
        return state

    def allow_request(self, endpoint: ModelEndpoint) -> bool:
        """Returns True if the endpoint is CLOSED or an available HALF_OPEN probe candidate."""
        key = endpoint.key
        state = self.get_state(endpoint)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            if not self._half_open_probe_in_flight.get(key, False):
                self._half_open_probe_in_flight[key] = True
                return True
            return False
        return False

    def record_success(self, endpoint: ModelEndpoint) -> None:
        """Resets failure counters and transitions HALF_OPEN -> CLOSED."""
        key = endpoint.key
        self._states[key] = CircuitState.CLOSED
        self._failure_timestamps[key] = []
        self._cooldown_until[key] = 0.0
        self._half_open_probe_in_flight[key] = False
        self._consecutive_trips[key] = 0

    def record_failure(
        self,
        endpoint: ModelEndpoint,
        status_code: int,
        error_body: str = "",
        headers: dict[str, str] | None = None,
    ) -> CircuitState:
        """Records error and transitions endpoint to OPEN if quota or capacity threshold is breached."""
        key = endpoint.key
        now = self._clock()
        self._last_status[key] = (status_code, error_body[:200])

        # HTTP 429: deterministic per-minute quota exhaustion -> trips immediately
        if status_code == 429:
            return self._trip(endpoint, status_code, error_body, headers)

        # HTTP 503 / 529: capacity overload -> sliding-window threshold
        if status_code in (503, 529):
            cutoff = now - self.config.sliding_window_seconds
            timestamps = [t for t in self._failure_timestamps.get(key, []) if t >= cutoff]
            timestamps.append(now)
            self._failure_timestamps[key] = timestamps
            if len(timestamps) >= self.config.failure_threshold_503:
                return self._trip(endpoint, status_code, error_body, headers)
            return self.get_state(endpoint)

        # Client-side / semantic errors (400, 401, 403, 404) do not trip breaker
        if self._states.get(key) == CircuitState.HALF_OPEN:
            self._half_open_probe_in_flight[key] = False
        return self.get_state(endpoint)

    def _trip(
        self,
        endpoint: ModelEndpoint,
        status_code: int,
        error_body: str,
        headers: dict[str, str] | None,
    ) -> CircuitState:
        key = endpoint.key
        now = self._clock()
        trips = self._consecutive_trips.get(key, 0) + 1
        self._consecutive_trips[key] = trips

        explicit_retry = self.parse_retry_after(headers, error_body)
        if explicit_retry is not None:
            cooldown = explicit_retry
        else:
            exp_factor = self.config.backoff_multiplier ** (trips - 1)
            raw_cooldown = self.config.base_cooldown_seconds * exp_factor
            cooldown = min(self.config.max_cooldown_seconds, raw_cooldown)

        jitter = random.uniform(*self.config.jitter_range_seconds)
        cooldown += jitter

        self._cooldown_until[key] = now + cooldown
        self._states[key] = CircuitState.OPEN
        self._half_open_probe_in_flight[key] = False
        return CircuitState.OPEN

    @staticmethod
    def parse_retry_after(
        headers: dict[str, str] | None,
        error_body: str = "",
    ) -> float | None:
        """Parses Retry-After header (seconds or HTTP date) or gRPC RetryInfo payload."""
        if headers:
            for header_name, header_val in headers.items():
                if header_name.lower() in ("retry-after", "x-ratelimit-reset-requests"):
                    header_str = header_val.strip()
                    try:
                        return float(header_str)
                    except ValueError:
                        try:
                            target_dt = parsedate_to_datetime(header_str)
                            diff = target_dt.timestamp() - time.time()
                            return max(0.0, diff)
                        except Exception:
                            pass

        if error_body:
            try:
                data = json.loads(error_body)
                if isinstance(data, dict):
                    # Check gRPC retryDelay e.g. "15s" or {"seconds": 15}
                    details = data.get("error", {}).get("details", [])
                    if isinstance(details, list):
                        for detail in details:
                            if isinstance(detail, dict) and "retryDelay" in detail:
                                delay_val = detail["retryDelay"]
                                if isinstance(delay_val, str) and delay_val.endswith("s"):
                                    return float(delay_val[:-1])
                                if isinstance(delay_val, (int, float)):
                                    return float(delay_val)
                    # Check rate_limit_error or retry_after in root
                    if "retry_after" in data:
                        return float(data["retry_after"])
            except Exception:
                pass

            # Regex search for e.g. "retry after 45 seconds" or "quota resets in 45s"
            match = re.search(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*s", error_body, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def get_health_snapshot(self, endpoint: ModelEndpoint) -> EndpointHealthSnapshot:
        """Returns read-only telemetry snapshot for TUI and diagnostics."""
        key = endpoint.key
        state = self.get_state(endpoint)
        cooldown_remaining = max(0.0, self._cooldown_until.get(key, 0.0) - self._clock())
        failures = self._consecutive_trips.get(key, 0)
        last_stat = self._last_status.get(key)
        return EndpointHealthSnapshot(
            endpoint=endpoint,
            state=state,
            consecutive_failures=failures,
            cooldown_remaining_seconds=cooldown_remaining,
            last_status_code=last_stat[0] if last_stat else None,
            last_error_reason=last_stat[1] if last_stat else None,
        )


class CrossFamilyPayloadNormalizer:
    """Lossless payload adapter across Anthropic Messages format and Vertex Gemini format."""

    @staticmethod
    def anthropic_to_gemini(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converts Anthropic messages (tool_use / tool_result) to Gemini contents (parts)."""
        gemini_contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "user" if msg.get("role") in ("user", "system") else "model"
            content = msg.get("content", "")
            parts: list[dict[str, Any]] = []

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item_type == "tool_use":
                            parts.append(
                                {
                                    "functionCall": {
                                        "name": item.get("name", ""),
                                        "args": item.get("input", {}),
                                    }
                                }
                            )
                        elif item_type == "tool_result":
                            parts.append(
                                {
                                    "functionResponse": {
                                        "name": item.get("tool_use_id", ""),
                                        "response": {"content": item.get("content", "")},
                                    }
                                }
                            )
            gemini_contents.append({"role": role, "parts": parts})
        return gemini_contents

    @staticmethod
    def gemini_to_anthropic(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converts Gemini contents (functionCall / functionResponse) to Anthropic messages."""
        anthropic_messages: list[dict[str, Any]] = []
        for content in contents:
            role = "assistant" if content.get("role") == "model" else "user"
            parts = content.get("parts", [])
            content_list: list[dict[str, Any]] = []

            for part in parts:
                if "text" in part:
                    content_list.append({"type": "text", "text": part["text"]})
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    content_list.append(
                        {
                            "type": "tool_use",
                            "id": f"call_{fc.get('name', 'fn')}",
                            "name": fc.get("name", ""),
                            "input": fc.get("args", {}),
                        }
                    )
                elif "functionResponse" in part:
                    fr = part["functionResponse"]
                    content_list.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": fr.get("name", "call_fn"),
                            "content": json.dumps(fr.get("response", {})),
                        }
                    )

            anthropic_messages.append({"role": role, "content": content_list})
        return anthropic_messages


class FallbackRouter:
    """Orchestrates request dispatch across Tier 1 -> Tier 2 -> Tier 3 endpoints."""

    def __init__(
        self,
        circuit_breaker: QuotaCircuitBreaker,
        fallback_chain: Sequence[ModelEndpoint],
        on_failover_callback: (
            Callable[[ModelEndpoint, ModelEndpoint, int, str], None] | None
        ) = None,
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.fallback_chain = list(fallback_chain)
        self.on_failover_callback = on_failover_callback

    def select_endpoint(self) -> ModelEndpoint:
        """Returns the highest-priority healthy endpoint in the fallback chain."""
        for endpoint in self.fallback_chain:
            if self.circuit_breaker.allow_request(endpoint):
                return endpoint
        snapshots = [self.circuit_breaker.get_health_snapshot(ep) for ep in self.fallback_chain]
        raise ExhaustedFallbackChainError(snapshots)

    def execute_with_fallback(
        self,
        request_fn: Callable[[ModelEndpoint, dict[str, Any]], Any],
        canonical_payload: dict[str, Any],
    ) -> Any:
        """Executes a synchronous request with automatic failover on quota or capacity failure."""
        tried_endpoints: set[str] = set()

        while True:
            candidate: ModelEndpoint | None = None
            for ep in self.fallback_chain:
                if ep.key not in tried_endpoints and self.circuit_breaker.allow_request(ep):
                    candidate = ep
                    break

            if candidate is None:
                snapshots = [
                    self.circuit_breaker.get_health_snapshot(ep) for ep in self.fallback_chain
                ]
                raise ExhaustedFallbackChainError(snapshots)

            tried_endpoints.add(candidate.key)

            try:
                result = request_fn(candidate, canonical_payload)
                self.circuit_breaker.record_success(candidate)
                return result
            except UpstreamQuotaError as exc:
                self.circuit_breaker.record_failure(
                    candidate,
                    exc.status_code,
                    str(exc),
                    exc.headers,
                )
                next_ep: ModelEndpoint | None = None
                for ep in self.fallback_chain:
                    if ep.key not in tried_endpoints and self.circuit_breaker.allow_request(ep):
                        next_ep = ep
                        break
                if next_ep and self.on_failover_callback:
                    self.on_failover_callback(candidate, next_ep, exc.status_code, str(exc))
                continue
