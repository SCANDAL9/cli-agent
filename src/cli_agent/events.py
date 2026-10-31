"""
High-Performance Asynchronous Pub/Sub EventBus (Wave 5 Milestone 2).

Provides lock-free, zero-blocking reactive telemetry pub/sub with bounded
ring-buffer backpressure protection and 60 FPS frame coalescing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .circuit_breaker import CircuitState
from .pr import SymbolDelta


class EventTopic(StrEnum):
    """Canonical Pub/Sub telemetry topics in the Wave 5 reactive runtime."""

    TOKEN_STREAM = "token.stream"
    SWARM_NODE_STATE = "swarm.node_state"
    WORKTREE_DIFF_UPDATED = "worktree.diff_updated"
    SUBPROC_LOG = "subproc.log"
    CIRCUIT_HEALTH = "circuit.health"
    TUI_CONTROL = "tui.control"


@dataclass(frozen=True, slots=True)
class TokenStreamPayload:
    """Payload for `token.stream` events."""

    agent_id: str
    delta: str
    is_thought: bool = False
    is_complete: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class SwarmNodeStatePayload:
    """Payload for `swarm.node_state` events."""

    plan_id: str
    node_id: str
    role: str  # "architect" | "worker" | "qa"
    status: str  # "PENDING" | "READY" | "RUNNING" | "COMPLETED" | "FAILED" | "SKIPPED"
    active_tool: str | None = None
    worktree_branch: str | None = None
    dependencies: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorktreeDiffPayload:
    """Payload for `worktree.diff_updated` events."""

    worktree_id: str
    branch_name: str
    file_path: str
    diff_text: str
    insertions: int = 0
    deletions: int = 0
    symbol_delta: SymbolDelta | None = None
    lsp_diagnostics_count: int = 0


@dataclass(frozen=True, slots=True)
class SubprocLogPayload:
    """Payload for `subproc.log` events."""

    source: str  # "TDD" | "LSP" | "MCP" | "CI" | "GIT"
    line: str
    level: str = "INFO"  # "DEBUG" | "INFO" | "WARN" | "ERROR"
    worker_id: str | None = None


@dataclass(frozen=True, slots=True)
class CircuitHealthPayload:
    """Payload for `circuit.health` events."""

    model_id: str
    region: str
    tier: int
    state: CircuitState
    failure_count: int = 0
    cooldown_remaining_s: float = 0.0
    failover_from: str | None = None
    vad_active: bool = False


@dataclass(frozen=True, slots=True)
class TUIControlPayload:
    """Payload for `tui.control` bi-directional user actions emitted from TUIApp."""

    action: str  # "SUBMIT_PROMPT" | "CANCEL_SWARM" | "SWITCH_WORKTREE" | "TOGGLE_VOICE"
    argument: str = ""


PayloadT = (
    TokenStreamPayload
    | SwarmNodeStatePayload
    | WorktreeDiffPayload
    | SubprocLogPayload
    | CircuitHealthPayload
    | TUIControlPayload
    | dict[str, Any]
)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Immutable message envelope routed through the EventBus."""

    topic: EventTopic
    payload: PayloadT
    timestamp: float = field(default_factory=time.monotonic)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class RingBufferSubscription:
    """Bounded lock-free ring-buffer subscriber with drop-oldest backpressure semantics."""

    def __init__(
        self,
        subscription_id: str = "",
        topics: frozenset[str] | None = None,
        capacity: int = 4096,
        maxlen: int | None = None,
    ) -> None:
        effective_capacity = maxlen if maxlen is not None else capacity
        if effective_capacity <= 0:
            raise ValueError(f"Subscription capacity must be > 0, got {effective_capacity}")
        self.subscription_id = subscription_id or uuid.uuid4().hex[:8]
        self.topics = topics
        self._ring: deque[EventEnvelope] = deque(maxlen=effective_capacity)
        self._wake_event = asyncio.Event()
        self.dropped_count: int = 0
        self._closed: bool = False

    @property
    def is_closed(self) -> bool:
        """Returns True if this subscription has been closed."""
        return self._closed

    @property
    def pending_count(self) -> int:
        """Returns the number of buffered unconsumed envelopes in the ring buffer."""
        return len(self._ring)

    def push_nowait(self, envelope: EventEnvelope) -> None:
        """Appends an envelope with O(1) drop-oldest backpressure semantics."""
        if self._closed:
            return
        if self._ring.maxlen is not None and len(self._ring) >= self._ring.maxlen:
            self.dropped_count += 1
        self._ring.append(envelope)
        self._wake_event.set()

    def drain(self) -> list[EventEnvelope]:
        """Drains and returns all buffered envelopes, clearing the wake signal."""
        self._wake_event.clear()
        if not self._ring:
            return []
        items = list(self._ring)
        self._ring.clear()
        return items

    def drain_nowait(self) -> list[EventEnvelope]:
        """Alias for drain() for parity with non-blocking APIs."""
        return self.drain()

    async def wait_and_drain(self, timeout: float | None = None) -> list[EventEnvelope]:
        """Awaits until at least one event is buffered, then drains all envelopes."""
        if not self._ring and not self._closed:
            if timeout is not None:
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
                except TimeoutError:
                    return []
            else:
                await self._wake_event.wait()
        return self.drain()

    def close(self) -> None:
        """Closes the subscription and wakes any pending consumers."""
        self._closed = True
        self._wake_event.set()

    async def __aiter__(self) -> AsyncIterator[EventEnvelope]:
        """Asynchronously yields envelopes one-by-one as they arrive."""
        while not self._closed or self.pending_count > 0:
            events = await self.wait_and_drain()
            if not events and self._closed:
                break
            for ev in events:
                yield ev


def coalesce_frame_events(envelopes: Sequence[EventEnvelope]) -> list[EventEnvelope]:
    """Coalesces high-frequency events within a 16.67ms (60 FPS) frame window."""
    if len(envelopes) <= 1:
        return list(envelopes)

    token_accum: dict[tuple[str, bool], TokenStreamPayload] = {}
    node_states: dict[tuple[str, str], EventEnvelope] = {}
    diff_updates: dict[tuple[str, str], EventEnvelope] = {}
    circuit_states: dict[tuple[str, str], EventEnvelope] = {}
    ordered_passthrough: list[EventEnvelope] = []

    for env in envelopes:
        if env.topic == EventTopic.TOKEN_STREAM and isinstance(env.payload, TokenStreamPayload):
            p = env.payload
            key = (p.agent_id, p.is_thought)
            if key in token_accum:
                prev = token_accum[key]
                token_accum[key] = TokenStreamPayload(
                    agent_id=p.agent_id,
                    delta=prev.delta + p.delta,
                    is_thought=p.is_thought,
                    is_complete=prev.is_complete or p.is_complete,
                    input_tokens=max(prev.input_tokens, p.input_tokens),
                    output_tokens=max(prev.output_tokens, p.output_tokens),
                    model_id=p.model_id or prev.model_id,
                )
            else:
                token_accum[key] = p

        elif env.topic == EventTopic.SWARM_NODE_STATE and isinstance(
            env.payload, SwarmNodeStatePayload
        ):
            node_states[(env.payload.plan_id, env.payload.node_id)] = env

        elif env.topic == EventTopic.WORKTREE_DIFF_UPDATED and isinstance(
            env.payload, WorktreeDiffPayload
        ):
            diff_updates[(env.payload.worktree_id, env.payload.file_path)] = env

        elif env.topic == EventTopic.CIRCUIT_HEALTH and isinstance(
            env.payload, CircuitHealthPayload
        ):
            circuit_states[(env.payload.model_id, env.payload.region)] = env

        else:
            # SUBPROC_LOG and TUI_CONTROL preserve exact chronological ordering
            ordered_passthrough.append(env)

    coalesced: list[EventEnvelope] = [
        EventEnvelope(topic=EventTopic.TOKEN_STREAM, payload=tp) for tp in token_accum.values()
    ]
    coalesced.extend(node_states.values())
    coalesced.extend(diff_updates.values())
    coalesced.extend(circuit_states.values())
    coalesced.extend(ordered_passthrough)
    return coalesced


class EventBus:
    """High-performance asynchronous Pub/Sub telemetry broker with 60 FPS coalescing."""

    def __init__(
        self,
        default_capacity: int = 4096,
        fps_limit: int = 60,
        default_maxlen: int | None = None,
    ) -> None:
        self.capacity = default_maxlen if default_maxlen is not None else default_capacity
        self.fps_limit = fps_limit
        self._subscriptions: set[RingBufferSubscription] = set()
        self._topic_subscribers: dict[str, set[RingBufferSubscription]] = {}
        self._wildcard_subscribers: set[RingBufferSubscription] = set()

    def subscribe(
        self,
        topic: EventTopic | str | Iterable[EventTopic | str] | None = None,
        capacity: int = 4096,
        maxlen: int | None = None,
        topics: Iterable[EventTopic | str] | None = None,
    ) -> RingBufferSubscription:
        """Registers a ring-buffer subscription for a specific topic, topics list, or wildcard '*'."""
        effective_cap = maxlen if maxlen is not None else capacity
        target_topics: set[str] = set()

        raw_topics = topics if topics is not None else topic
        if raw_topics is None:
            target_topics.add("*")
        elif isinstance(raw_topics, (str, EventTopic)):
            target_topics.add(str(raw_topics))
        else:
            for t in raw_topics:
                target_topics.add(str(t))

        is_wildcard = "*" in target_topics
        sub = RingBufferSubscription(
            subscription_id=uuid.uuid4().hex[:8],
            topics=frozenset(target_topics) if not is_wildcard else None,
            capacity=effective_cap,
        )
        self._subscriptions.add(sub)
        if is_wildcard:
            self._wildcard_subscribers.add(sub)
        else:
            for t in target_topics:
                self._topic_subscribers.setdefault(t, set()).add(sub)
        return sub

    def unsubscribe(self, subscription: RingBufferSubscription) -> None:
        """Unregisters and closes a subscriber."""
        self._subscriptions.discard(subscription)
        self._wildcard_subscribers.discard(subscription)
        for topic_set in self._topic_subscribers.values():
            topic_set.discard(subscription)
        subscription.close()

    def publish_nowait(
        self,
        topic: EventTopic | str,
        payload: PayloadT,
    ) -> EventEnvelope:
        """Dispatches an event envelope to all matching subscriber ring buffers in O(1)."""
        topic_str = str(topic)
        try:
            topic_enum = topic if isinstance(topic, EventTopic) else EventTopic(topic_str)
        except ValueError as err:
            raise ValueError(f"Unknown event topic: {topic_str}") from err

        envelope = EventEnvelope(topic=topic_enum, payload=payload)

        # Collect unique subscribers matching either wildcard or specific topic
        matched_subs: set[RingBufferSubscription] = set(self._wildcard_subscribers)
        if topic_str in self._topic_subscribers:
            matched_subs.update(self._topic_subscribers[topic_str])

        for sub in matched_subs:
            sub.push_nowait(envelope)

        return envelope

    async def publish(
        self,
        topic: EventTopic | str,
        payload: PayloadT,
    ) -> EventEnvelope:
        """Asynchronous wrapper around publish_nowait."""
        return self.publish_nowait(topic, payload)

    async def iter_coalesced_frames(
        self,
        subscription: RingBufferSubscription,
        stop_event: asyncio.Event | None = None,
        fps_limit: int | None = None,
    ) -> AsyncIterator[list[EventEnvelope]]:
        """Yields coalesced event frame batches at the configured frame rate."""
        effective_fps = fps_limit if fps_limit is not None else self.fps_limit
        interval = (1.0 / effective_fps) if effective_fps > 0 else 0.0

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if subscription.is_closed and subscription.pending_count == 0:
                break

            # Idle state: wait on wake_event or stop_event with 0.0% CPU
            if subscription.pending_count == 0:
                if stop_event is None:
                    await subscription._wake_event.wait()
                else:
                    wake_task = asyncio.create_task(subscription._wake_event.wait())
                    stop_task = asyncio.create_task(stop_event.wait())
                    done, pending = await asyncio.wait(
                        [wake_task, stop_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                    if stop_event.is_set():
                        break

            if subscription.is_closed and subscription.pending_count == 0:
                break

            # First-Event Immediate Edge Trigger: flush first burst without delay
            first_batch = subscription.drain()
            if first_batch:
                yield coalesce_frame_events(first_batch)

            if interval > 0:
                await asyncio.sleep(interval)

            # Subsequent bursts within active frame window are coalesced
            while subscription.pending_count > 0:
                if stop_event is not None and stop_event.is_set():
                    break
                burst = subscription.drain()
                if burst:
                    yield coalesce_frame_events(burst)
                if interval > 0:
                    await asyncio.sleep(interval)


_global_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Returns the process-wide EventBus singleton."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(bus: EventBus | None) -> None:
    """Sets or resets the process-wide EventBus singleton."""
    global _global_event_bus
    _global_event_bus = bus


get_global_event_bus = get_event_bus
set_global_event_bus = set_event_bus
