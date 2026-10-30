"""Unit tests for Model Quota Circuit Breaker & Fallback Router (W4.1)."""

import json
from unittest.mock import MagicMock

import pytest

from cli_agent.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    CrossFamilyPayloadNormalizer,
    ExhaustedFallbackChainError,
    FallbackRouter,
    ModelEndpoint,
    QuotaCircuitBreaker,
    UpstreamQuotaError,
)


class FakeClock:
    """Controllable monotonic clock for deterministic time travel in tests."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def deterministic_config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold_503=2,
        sliding_window_seconds=30.0,
        base_cooldown_seconds=60.0,
        max_cooldown_seconds=300.0,
        backoff_multiplier=2.0,
        jitter_range_seconds=(0.0, 0.0),  # Zero jitter for deterministic assertions
    )


@pytest.fixture
def breaker(
    deterministic_config: CircuitBreakerConfig,
    fake_clock: FakeClock,
) -> QuotaCircuitBreaker:
    return QuotaCircuitBreaker(config=deterministic_config, clock=fake_clock)


@pytest.fixture
def ep_primary() -> ModelEndpoint:
    return ModelEndpoint("claude-sonnet-4-6", "us-east5", tier=1)


@pytest.fixture
def ep_cross_region() -> ModelEndpoint:
    return ModelEndpoint("claude-sonnet-4-6", "us-central1", tier=1)


@pytest.fixture
def ep_tier2_gemini() -> ModelEndpoint:
    return ModelEndpoint("gemini-2.5-pro", "us-central1", tier=2)


@pytest.fixture
def ep_tier3_flash() -> ModelEndpoint:
    return ModelEndpoint("gemini-2.5-flash", "global", tier=3)


def test_initial_state_closed(breaker: QuotaCircuitBreaker, ep_primary: ModelEndpoint) -> None:
    assert breaker.get_state(ep_primary) == CircuitState.CLOSED
    assert breaker.allow_request(ep_primary) is True


def test_immediate_trip_on_429(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    state = breaker.record_failure(
        ep_primary,
        status_code=429,
        error_body="Quota exceeded for Requests Per Minute",
    )
    assert state == CircuitState.OPEN
    assert breaker.get_state(ep_primary) == CircuitState.OPEN
    assert breaker.allow_request(ep_primary) is False

    snapshot = breaker.get_health_snapshot(ep_primary)
    assert snapshot.state == CircuitState.OPEN
    assert snapshot.consecutive_failures == 1
    assert snapshot.cooldown_remaining_seconds == 60.0
    assert snapshot.last_status_code == 429


def test_sliding_window_503_threshold(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    # First 503 at t=1000
    state1 = breaker.record_failure(ep_primary, status_code=503, error_body="Overloaded")
    assert state1 == CircuitState.CLOSED
    assert breaker.allow_request(ep_primary) is True

    # Advance time within sliding window (t=1015)
    fake_clock.advance(15.0)
    state2 = breaker.record_failure(ep_primary, status_code=503, error_body="Overloaded")
    assert state2 == CircuitState.OPEN
    assert breaker.allow_request(ep_primary) is False


def test_sliding_window_503_expired(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    # First 503 at t=1000
    breaker.record_failure(ep_primary, status_code=503)
    assert breaker.get_state(ep_primary) == CircuitState.CLOSED

    # Advance beyond sliding window (t=1035 > 30s)
    fake_clock.advance(35.0)
    # Second 503 should not trip because first 503 has expired
    breaker.record_failure(ep_primary, status_code=503)
    assert breaker.get_state(ep_primary) == CircuitState.CLOSED
    assert breaker.allow_request(ep_primary) is True


def test_non_tripping_client_errors(
    breaker: QuotaCircuitBreaker, ep_primary: ModelEndpoint
) -> None:
    for code in (400, 401, 403, 404):
        state = breaker.record_failure(ep_primary, status_code=code, error_body="Bad Request")
        assert state == CircuitState.CLOSED
        assert breaker.allow_request(ep_primary) is True


def test_retry_after_header_parsing(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    breaker.record_failure(
        ep_primary,
        status_code=429,
        headers={"Retry-After": "45"},
    )
    snapshot = breaker.get_health_snapshot(ep_primary)
    assert snapshot.cooldown_remaining_seconds == 45.0


def test_retry_after_grpc_json_parsing(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
) -> None:
    body = json.dumps(
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"retryDelay": "18.5s"}],
            }
        }
    )
    breaker.record_failure(ep_primary, status_code=429, error_body=body)
    snapshot = breaker.get_health_snapshot(ep_primary)
    assert snapshot.cooldown_remaining_seconds == 18.5


def test_open_to_half_open_and_single_probe_gate(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    breaker.record_failure(ep_primary, status_code=429)
    assert breaker.get_state(ep_primary) == CircuitState.OPEN

    # Advance past cooldown (60s)
    fake_clock.advance(65.0)
    assert breaker.get_state(ep_primary) == CircuitState.HALF_OPEN

    # First probe request allowed
    assert breaker.allow_request(ep_primary) is True
    # Concurrent second probe request rejected while probe is in flight
    assert breaker.allow_request(ep_primary) is False


def test_half_open_probe_success_recovery(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    breaker.record_failure(ep_primary, status_code=429)
    fake_clock.advance(65.0)
    assert breaker.allow_request(ep_primary) is True

    breaker.record_success(ep_primary)
    assert breaker.get_state(ep_primary) == CircuitState.CLOSED
    assert breaker.allow_request(ep_primary) is True

    snapshot = breaker.get_health_snapshot(ep_primary)
    assert snapshot.consecutive_failures == 0


def test_half_open_probe_failure_exponential_backoff(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    fake_clock: FakeClock,
) -> None:
    # First trip: 60s cooldown
    breaker.record_failure(ep_primary, status_code=429)
    fake_clock.advance(65.0)
    assert breaker.allow_request(ep_primary) is True

    # Probe fails: 2nd trip doubles cooldown to 120s
    breaker.record_failure(ep_primary, status_code=429)
    assert breaker.get_state(ep_primary) == CircuitState.OPEN

    snapshot = breaker.get_health_snapshot(ep_primary)
    assert snapshot.consecutive_failures == 2
    assert snapshot.cooldown_remaining_seconds == 120.0


def test_fallback_router_cross_region_failover(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    ep_cross_region: ModelEndpoint,
) -> None:
    callback = MagicMock()
    chain = [ep_primary, ep_cross_region]
    router = FallbackRouter(breaker, chain, on_failover_callback=callback)

    def mock_request(ep: ModelEndpoint, payload: dict[str, object]) -> str:
        if ep == ep_primary:
            raise UpstreamQuotaError(429, "Rate limit exceeded", ep)
        return "response-from-cross-region"

    result = router.execute_with_fallback(mock_request, {"prompt": "hello"})
    assert result == "response-from-cross-region"
    assert breaker.get_state(ep_primary) == CircuitState.OPEN
    assert breaker.get_state(ep_cross_region) == CircuitState.CLOSED
    callback.assert_called_once_with(
        ep_primary,
        ep_cross_region,
        429,
        "Rate limit exceeded",
    )


def test_fallback_router_exhaustion_raises_structured_error(
    breaker: QuotaCircuitBreaker,
    ep_primary: ModelEndpoint,
    ep_cross_region: ModelEndpoint,
) -> None:
    breaker.record_failure(ep_primary, status_code=429)
    breaker.record_failure(ep_cross_region, status_code=429)

    router = FallbackRouter(breaker, [ep_primary, ep_cross_region])

    with pytest.raises(ExhaustedFallbackChainError) as exc_info:
        router.select_endpoint()

    error = exc_info.value
    assert len(error.snapshots) == 2
    assert "All fallback model tiers exhausted" in str(error)


def test_cross_family_payload_normalizer_anthropic_to_gemini() -> None:
    anthropic_msgs = [
        {"role": "user", "content": "Analyze code"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_search_1",
                    "name": "code_search",
                    "input": {"query": "CircuitBreaker"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_search_1",
                    "content": "found match on line 12",
                }
            ],
        },
    ]

    gemini_contents = CrossFamilyPayloadNormalizer.anthropic_to_gemini(anthropic_msgs)
    assert len(gemini_contents) == 3
    assert gemini_contents[0]["role"] == "user"
    assert gemini_contents[0]["parts"][0]["text"] == "Analyze code"

    assert gemini_contents[1]["role"] == "model"
    fc = gemini_contents[1]["parts"][0]["functionCall"]
    assert fc["name"] == "code_search"
    assert fc["args"] == {"query": "CircuitBreaker"}

    assert gemini_contents[2]["role"] == "user"
    fr = gemini_contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "call_search_1"
    assert fr["response"]["content"] == "found match on line 12"


def test_cross_family_payload_normalizer_gemini_to_anthropic() -> None:
    gemini_contents = [
        {"role": "user", "parts": [{"text": "Generate test"}]},
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "run_test",
                        "args": {"path": "tests/test_cb.py"},
                    }
                }
            ],
        },
    ]

    anthropic_msgs = CrossFamilyPayloadNormalizer.gemini_to_anthropic(gemini_contents)
    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[0]["content"][0]["text"] == "Generate test"

    assert anthropic_msgs[1]["role"] == "assistant"
    tool_use = anthropic_msgs[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "run_test"
    assert tool_use["input"] == {"path": "tests/test_cb.py"}
