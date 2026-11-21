"""
Unit tests for Asynchronous Pub/Sub EventBus and TUIApp Integration (Wave 5 Milestone 2).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError

import pytest

from cli_agent.circuit_breaker import CircuitState, ModelEndpoint
from cli_agent.events import (
    CircuitHealthPayload,
    EventBus,
    EventEnvelope,
    EventTopic,
    RingBufferSubscription,
    SubprocLogPayload,
    SwarmNodeStatePayload,
    TokenStreamPayload,
    TUIControlPayload,
    WorktreeDiffPayload,
    coalesce_frame_events,
    get_event_bus,
    get_global_event_bus,
    set_event_bus,
    set_global_event_bus,
)
from cli_agent.pr import SymbolDelta
from cli_agent.tui import DiffViewMode, LogSource, TUIApp

# ---------------------------------------------------------------------------
# 1. Strictly Typed Payload Models
# ---------------------------------------------------------------------------


def test_token_stream_payload_immutability() -> None:
    p = TokenStreamPayload(
        agent_id="worker-1",
        delta="def foo():",
        is_thought=False,
        is_complete=False,
        input_tokens=100,
        output_tokens=20,
        model_id="gemini-2.5-pro",
    )
    assert p.agent_id == "worker-1"
    assert p.delta == "def foo():"
    assert p.model_id == "gemini-2.5-pro"
    with pytest.raises(FrozenInstanceError):
        # Immutable frozen dataclass
        p.delta = "bar"  # type: ignore[misc]


def test_swarm_node_state_payload() -> None:
    p = SwarmNodeStatePayload(
        plan_id="plan-1",
        node_id="node-a",
        role="worker",
        status="RUNNING",
        active_tool="edit_file",
        worktree_branch="feat/test",
        dependencies=("node-root",),
        error=None,
    )
    assert p.plan_id == "plan-1"
    assert p.node_id == "node-a"
    assert p.status == "RUNNING"
    assert p.dependencies == ("node-root",)


def test_worktree_diff_payload() -> None:
    delta = SymbolDelta(
        file_path="src/app.py",
        change_type="MODIFIED",
        modified_symbols=("main",),
    )
    p = WorktreeDiffPayload(
        worktree_id="wt-123",
        branch_name="branch-a",
        file_path="src/app.py",
        diff_text="@@ -1 +1 @@\n-x\n+y",
        insertions=1,
        deletions=1,
        symbol_delta=delta,
        lsp_diagnostics_count=0,
    )
    assert p.worktree_id == "wt-123"
    assert p.symbol_delta is not None
    assert p.symbol_delta.modified_symbols == ("main",)


def test_subproc_log_payload() -> None:
    p = SubprocLogPayload(
        source="TDD",
        line="pytest passed in 0.12s",
        level="INFO",
        worker_id="worker-2",
    )
    assert p.source == "TDD"
    assert p.line == "pytest passed in 0.12s"
    assert p.level == "INFO"
    assert p.worker_id == "worker-2"


def test_circuit_health_payload() -> None:
    p = CircuitHealthPayload(
        model_id="gemini-3.8-flash",
        region="us-central1",
        tier=3,
        state=CircuitState.HALF_OPEN,
        failure_count=2,
        cooldown_remaining_s=15.5,
        failover_from="gemini-2.5-pro",
        vad_active=True,
    )
    assert p.model_id == "gemini-3.8-flash"
    assert p.state == CircuitState.HALF_OPEN
    assert p.vad_active is True


def test_tui_control_payload() -> None:
    p = TUIControlPayload(action="CANCEL_SWARM", argument="user_interrupt")
    assert p.action == "CANCEL_SWARM"
    assert p.argument == "user_interrupt"


def test_event_envelope_metadata() -> None:
    now = time.monotonic()
    env = EventEnvelope(
        topic=EventTopic.TUI_CONTROL,
        payload=TUIControlPayload(action="TOGGLE_VOICE"),
    )
    assert env.topic == EventTopic.TUI_CONTROL
    assert env.timestamp >= now
    assert len(env.event_id) == 12


# ---------------------------------------------------------------------------
# 2. RingBufferSubscription Semantics
# ---------------------------------------------------------------------------


def test_ring_buffer_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be > 0"):
        RingBufferSubscription(capacity=0)


def test_ring_buffer_push_drain_fifo() -> None:
    sub = RingBufferSubscription(capacity=10)
    assert sub.pending_count == 0
    assert sub.dropped_count == 0

    env1 = EventEnvelope(
        topic=EventTopic.TOKEN_STREAM,
        payload=TokenStreamPayload(agent_id="a", delta="1"),
    )
    env2 = EventEnvelope(
        topic=EventTopic.TOKEN_STREAM,
        payload=TokenStreamPayload(agent_id="a", delta="2"),
    )

    sub.push_nowait(env1)
    sub.push_nowait(env2)
    assert sub.pending_count == 2

    drained = sub.drain()
    assert len(drained) == 2
    assert drained[0].payload.delta == "1"  # type: ignore[union-attr]
    assert drained[1].payload.delta == "2"  # type: ignore[union-attr]
    assert sub.pending_count == 0

    # drain when empty returns []
    assert sub.drain() == []
    assert sub.drain_nowait() == []


def test_ring_buffer_drop_oldest_backpressure() -> None:
    sub = RingBufferSubscription(capacity=3)
    envelopes = [
        EventEnvelope(
            topic=EventTopic.TOKEN_STREAM,
            payload=TokenStreamPayload(agent_id="a", delta=str(i)),
        )
        for i in range(7)
    ]

    for env in envelopes:
        sub.push_nowait(env)

    # We pushed 7 items into capacity 3.
    # 4 oldest should be dropped: 0, 1, 2, 3
    # Ring should contain 4, 5, 6
    assert sub.dropped_count == 4
    assert sub.pending_count == 3

    drained = sub.drain()
    assert len(drained) == 3
    assert [e.payload.delta for e in drained] == ["4", "5", "6"]  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ring_buffer_wait_and_drain_timeout() -> None:
    sub = RingBufferSubscription(capacity=10)
    # Empty ring buffer timeout
    result = await sub.wait_and_drain(timeout=0.01)
    assert result == []


@pytest.mark.asyncio
async def test_ring_buffer_wait_and_drain_with_event() -> None:
    sub = RingBufferSubscription(capacity=10)
    env = EventEnvelope(
        topic=EventTopic.TOKEN_STREAM,
        payload=TokenStreamPayload(agent_id="a", delta="ok"),
    )

    async def delayed_push() -> None:
        await asyncio.sleep(0.01)
        sub.push_nowait(env)

    task = asyncio.create_task(delayed_push())
    drained = await sub.wait_and_drain()
    await task
    assert len(drained) == 1
    assert drained[0].payload.delta == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ring_buffer_async_iterator() -> None:
    sub = RingBufferSubscription(capacity=10)

    async def producer() -> None:
        for i in range(3):
            await asyncio.sleep(0.005)
            sub.push_nowait(
                EventEnvelope(
                    topic=EventTopic.TOKEN_STREAM,
                    payload=TokenStreamPayload(agent_id="a", delta=str(i)),
                )
            )
        await asyncio.sleep(0.01)
        sub.close()

    p_task = asyncio.create_task(producer())

    collected: list[str] = []
    async for env in sub:
        collected.append(env.payload.delta)  # type: ignore[union-attr]

    await p_task
    assert collected == ["0", "1", "2"]
    assert sub.is_closed is True

    # Pushing when closed is no-op
    sub.push_nowait(
        EventEnvelope(
            topic=EventTopic.TOKEN_STREAM,
            payload=TokenStreamPayload(agent_id="a", delta="noop"),
        )
    )
    assert sub.pending_count == 0


# ---------------------------------------------------------------------------
# 3. 60 FPS Frame Coalescing Algorithm
# ---------------------------------------------------------------------------


def test_coalesce_frame_events_empty_and_single() -> None:
    assert coalesce_frame_events([]) == []

    single = EventEnvelope(
        topic=EventTopic.TOKEN_STREAM,
        payload=TokenStreamPayload(agent_id="a", delta="x"),
    )
    res = coalesce_frame_events([single])
    assert len(res) == 1
    assert res[0] is single


def test_coalesce_frame_events_token_concatenation() -> None:
    envelopes = [
        EventEnvelope(
            topic=EventTopic.TOKEN_STREAM,
            payload=TokenStreamPayload(
                agent_id="worker-1",
                delta="Hello ",
                is_thought=False,
                input_tokens=10,
                output_tokens=2,
            ),
        ),
        EventEnvelope(
            topic=EventTopic.TOKEN_STREAM,
            payload=TokenStreamPayload(
                agent_id="worker-1",
                delta="world",
                is_thought=False,
                input_tokens=10,
                output_tokens=4,
                is_complete=True,
                model_id="gemini-2.5-flash",
            ),
        ),
        # Thought tokens for same agent shouldn't collide with non-thought
        EventEnvelope(
            topic=EventTopic.TOKEN_STREAM,
            payload=TokenStreamPayload(
                agent_id="worker-1",
                delta="thinking...",
                is_thought=True,
            ),
        ),
    ]

    coalesced = coalesce_frame_events(envelopes)
    assert len(coalesced) == 2

    token_payloads = [e.payload for e in coalesced if isinstance(e.payload, TokenStreamPayload)]
    non_thought = next(p for p in token_payloads if not p.is_thought)
    thought = next(p for p in token_payloads if p.is_thought)

    assert non_thought.delta == "Hello world"
    assert non_thought.is_complete is True
    assert non_thought.output_tokens == 4
    assert non_thought.model_id == "gemini-2.5-flash"

    assert thought.delta == "thinking..."
    assert thought.is_thought is True


def test_coalesce_frame_events_latest_state_dedup() -> None:
    envelopes = [
        # Swarm node transitions: RUNNING -> COMPLETED
        EventEnvelope(
            topic=EventTopic.SWARM_NODE_STATE,
            payload=SwarmNodeStatePayload(
                plan_id="p1", node_id="n1", role="worker", status="RUNNING"
            ),
        ),
        EventEnvelope(
            topic=EventTopic.SWARM_NODE_STATE,
            payload=SwarmNodeStatePayload(
                plan_id="p1", node_id="n1", role="worker", status="COMPLETED"
            ),
        ),
        # Worktree diff snapshots
        EventEnvelope(
            topic=EventTopic.WORKTREE_DIFF_UPDATED,
            payload=WorktreeDiffPayload(
                worktree_id="wt1",
                branch_name="b1",
                file_path="foo.py",
                diff_text="diff 1",
            ),
        ),
        EventEnvelope(
            topic=EventTopic.WORKTREE_DIFF_UPDATED,
            payload=WorktreeDiffPayload(
                worktree_id="wt1",
                branch_name="b1",
                file_path="foo.py",
                diff_text="diff 2 (latest)",
            ),
        ),
        # Circuit health
        EventEnvelope(
            topic=EventTopic.CIRCUIT_HEALTH,
            payload=CircuitHealthPayload(
                model_id="m1",
                region="r1",
                tier=1,
                state=CircuitState.HALF_OPEN,
            ),
        ),
        EventEnvelope(
            topic=EventTopic.CIRCUIT_HEALTH,
            payload=CircuitHealthPayload(
                model_id="m1",
                region="r1",
                tier=1,
                state=CircuitState.CLOSED,
            ),
        ),
    ]

    coalesced = coalesce_frame_events(envelopes)
    assert len(coalesced) == 3

    node_ev = next(e for e in coalesced if e.topic == EventTopic.SWARM_NODE_STATE)
    assert isinstance(node_ev.payload, SwarmNodeStatePayload)
    assert node_ev.payload.status == "COMPLETED"

    diff_ev = next(e for e in coalesced if e.topic == EventTopic.WORKTREE_DIFF_UPDATED)
    assert isinstance(diff_ev.payload, WorktreeDiffPayload)
    assert diff_ev.payload.diff_text == "diff 2 (latest)"

    cb_ev = next(e for e in coalesced if e.topic == EventTopic.CIRCUIT_HEALTH)
    assert isinstance(cb_ev.payload, CircuitHealthPayload)
    assert cb_ev.payload.state == CircuitState.CLOSED


def test_coalesce_frame_events_ordered_passthrough() -> None:
    envelopes = [
        EventEnvelope(
            topic=EventTopic.SUBPROC_LOG,
            payload=SubprocLogPayload(source="TDD", line="line 1"),
        ),
        EventEnvelope(
            topic=EventTopic.TUI_CONTROL,
            payload=TUIControlPayload(action="CYCLE_DIFF_FILE", argument="+1"),
        ),
        EventEnvelope(
            topic=EventTopic.SUBPROC_LOG,
            payload=SubprocLogPayload(source="TDD", line="line 2"),
        ),
    ]

    coalesced = coalesce_frame_events(envelopes)
    assert len(coalesced) == 3
    assert [e.payload.line for e in coalesced if isinstance(e.payload, SubprocLogPayload)] == [  # type: ignore[attr-defined]
        "line 1",
        "line 2",
    ]


# ---------------------------------------------------------------------------
# 4. EventBus Pub/Sub & Frame Pacing
# ---------------------------------------------------------------------------


def test_event_bus_subscribe_exact_and_wildcard() -> None:
    bus = EventBus()
    sub_tokens = bus.subscribe(EventTopic.TOKEN_STREAM)
    sub_wildcard = bus.subscribe("*")
    sub_multiple = bus.subscribe(topics=[EventTopic.SUBPROC_LOG, EventTopic.CIRCUIT_HEALTH])

    # Publish token
    bus.publish_nowait(
        EventTopic.TOKEN_STREAM,
        TokenStreamPayload(agent_id="a", delta="hello"),
    )

    assert sub_tokens.pending_count == 1
    assert sub_wildcard.pending_count == 1
    assert sub_multiple.pending_count == 0

    # Publish log
    bus.publish_nowait(
        EventTopic.SUBPROC_LOG,
        SubprocLogPayload(source="MCP", line="connected"),
    )

    assert sub_tokens.pending_count == 1
    assert sub_wildcard.pending_count == 2
    assert sub_multiple.pending_count == 1

    bus.unsubscribe(sub_tokens)
    assert sub_tokens.is_closed is True


def test_event_bus_invalid_topic_publish() -> None:
    bus = EventBus()
    with pytest.raises(ValueError, match="Unknown event topic"):
        bus.publish_nowait("unregistered.topic", {})


@pytest.mark.asyncio
async def test_event_bus_publish_async_wrapper() -> None:
    bus = EventBus()
    sub = bus.subscribe("swarm.node_state")
    env = await bus.publish(
        EventTopic.SWARM_NODE_STATE,
        SwarmNodeStatePayload(plan_id="p1", node_id="n1", role="qa", status="READY"),
    )
    assert env.topic == EventTopic.SWARM_NODE_STATE
    assert sub.pending_count == 1


@pytest.mark.asyncio
async def test_event_bus_iter_coalesced_frames() -> None:
    bus = EventBus(fps_limit=60)
    sub = bus.subscribe(EventTopic.TOKEN_STREAM)
    stop_event = asyncio.Event()

    frames: list[list[EventEnvelope]] = []

    async def frame_collector() -> None:
        async for frame in bus.iter_coalesced_frames(sub, stop_event=stop_event, fps_limit=100):
            frames.append(frame)
            if len(frames) >= 2:
                stop_event.set()

    collector_task = asyncio.create_task(frame_collector())

    # Burst 1: immediate flush edge-trigger
    bus.publish_nowait(
        EventTopic.TOKEN_STREAM,
        TokenStreamPayload(agent_id="arch", delta="chunk 1 "),
    )
    await asyncio.sleep(0.015)

    # Burst 2: multiple tokens coalesced into one frame
    bus.publish_nowait(
        EventTopic.TOKEN_STREAM,
        TokenStreamPayload(agent_id="arch", delta="chunk 2 "),
    )
    bus.publish_nowait(
        EventTopic.TOKEN_STREAM,
        TokenStreamPayload(agent_id="arch", delta="chunk 3"),
    )

    await asyncio.wait_for(collector_task, timeout=2.0)

    assert len(frames) >= 2
    first_frame = frames[0]
    assert len(first_frame) == 1
    assert first_frame[0].payload.delta == "chunk 1 "  # type: ignore[union-attr]

    second_frame = frames[1]
    assert len(second_frame) == 1
    assert second_frame[0].payload.delta == "chunk 2 chunk 3"  # type: ignore[union-attr]


def test_global_event_bus_accessors() -> None:
    orig = get_global_event_bus()
    assert orig is not None
    custom = EventBus()
    set_global_event_bus(custom)
    assert get_event_bus() is custom
    assert get_global_event_bus() is custom

    # Reset
    set_event_bus(orig)
    assert get_global_event_bus() is orig


# ---------------------------------------------------------------------------
# 5. TUIEventBusBridge & TUIApp Panes Integration
# ---------------------------------------------------------------------------


def test_tui_bridge_forward_telemetry() -> None:
    app = TUIApp()
    bridge = app.bridge

    # 1. token.stream -> ChatPane & StatusHUD
    env_token = EventEnvelope(
        topic=EventTopic.TOKEN_STREAM,
        payload=TokenStreamPayload(
            agent_id="agent-1",
            delta="Hello terminal!",
            is_complete=True,
            input_tokens=1500,
            output_tokens=320,
        ),
    )
    bridge.apply_envelope(env_token)
    assert len(app.chat_pane.messages) == 1
    assert app.chat_pane.messages[0]["content"] == "Hello terminal!"
    assert app.status_hud.tokens_in == 1500
    assert app.status_hud.tokens_out == 320

    # 2. swarm.node_state -> ChatPane hierarchy & StatusHUD active swarms
    env_swarm_lead = EventEnvelope(
        topic=EventTopic.SWARM_NODE_STATE,
        payload=SwarmNodeStatePayload(
            plan_id="plan-x",
            node_id="lead-arch",
            role="architect",
            status="RUNNING",
        ),
    )
    env_swarm_worker = EventEnvelope(
        topic=EventTopic.SWARM_NODE_STATE,
        payload=SwarmNodeStatePayload(
            plan_id="plan-x",
            node_id="worker-1",
            role="worker",
            status="RUNNING",
            active_tool="search_code",
            worktree_branch="feat/w5",
            dependencies=("lead-arch",),
        ),
    )
    bridge.apply_envelope(env_swarm_lead)
    bridge.apply_envelope(env_swarm_worker)

    assert app.chat_pane.swarm_root is not None
    assert app.chat_pane.swarm_root.agent_id == "lead-arch"
    assert len(app.chat_pane.swarm_root.children) == 1
    assert app.chat_pane.swarm_root.children[0].agent_id == "worker-1"
    assert app.status_hud.active_swarms == 2

    # 3. worktree.diff_updated -> DiffViewerPane
    delta = SymbolDelta("src/events.py", "ADDED", ("EventBus",))
    env_diff = EventEnvelope(
        topic=EventTopic.WORKTREE_DIFF_UPDATED,
        payload=WorktreeDiffPayload(
            worktree_id="wt-1",
            branch_name="feat/w5",
            file_path="src/events.py",
            diff_text="+class EventBus:\n+    pass",
            symbol_delta=delta,
        ),
    )
    bridge.apply_envelope(env_diff)
    assert app.diff_pane.active_file == "src/events.py"
    assert app.diff_pane.diffs["src/events.py"] == "+class EventBus:\n+    pass"
    assert app.diff_pane.symbol_deltas["src/events.py"] is delta

    # 4. subproc.log -> ProcessLogPane
    env_log = EventEnvelope(
        topic=EventTopic.SUBPROC_LOG,
        payload=SubprocLogPayload(source="TDD", line="PASSED tests/test_events.py"),
    )
    bridge.apply_envelope(env_log)
    assert len(app.log_pane.entries) == 1
    assert app.log_pane.entries[0][0] == LogSource.TDD
    assert app.log_pane.entries[0][1] == "PASSED tests/test_events.py"

    # Unknown source falls back to LogSource.ALL
    env_log_custom = EventEnvelope(
        topic=EventTopic.SUBPROC_LOG,
        payload=SubprocLogPayload(source="CUSTOM_TOOL", line="custom message"),
    )
    bridge.apply_envelope(env_log_custom)
    assert app.log_pane.entries[1][0] == LogSource.ALL

    # 5. circuit.health -> StatusHUD
    env_cb = EventEnvelope(
        topic=EventTopic.CIRCUIT_HEALTH,
        payload=CircuitHealthPayload(
            model_id="gemini-exp-1206",
            region="europe-west4",
            tier=1,
            state=CircuitState.HALF_OPEN,
            failure_count=1,
            cooldown_remaining_s=12.4,
            vad_active=True,
        ),
    )
    bridge.apply_envelope(env_cb)
    assert app.status_hud.endpoint == ModelEndpoint("gemini-exp-1206", "europe-west4", 1)
    assert app.status_hud.circuit_snapshot.state == CircuitState.HALF_OPEN
    assert app.status_hud.circuit_snapshot.consecutive_failures == 1
    assert app.status_hud.circuit_snapshot.cooldown_remaining_seconds == 12.4
    assert app.status_hud.vad_listening is True


def test_tui_bridge_reverse_control_path() -> None:
    bus = EventBus()
    app = TUIApp(event_bus=bus)

    control_sub = bus.subscribe(EventTopic.TUI_CONTROL)

    # Prompt submission
    prompts_submitted: list[str] = []
    app.on_submit_prompt = lambda p: prompts_submitted.append(p)
    app.submit_prompt("Refactor events architecture")
    assert prompts_submitted == ["Refactor events architecture"]

    ctrl_events = control_sub.drain()
    assert len(ctrl_events) == 1
    p0 = ctrl_events[0].payload
    assert isinstance(p0, TUIControlPayload)
    assert p0.action == "SUBMIT_PROMPT"
    assert p0.argument == "Refactor events architecture"

    # Key dispatch controls
    app.handle_key("Ctrl+D")
    assert app.diff_pane.mode == DiffViewMode.AST_OUTLINE

    app.handle_key("]")
    app.handle_key("[")
    app.handle_key("Ctrl+C")
    app.handle_key("/voice")

    dispatched = control_sub.drain()
    actions = [e.payload.action for e in dispatched if isinstance(e.payload, TUIControlPayload)]
    assert actions == [
        "TOGGLE_DIFF_MODE",
        "CYCLE_DIFF_FILE",
        "CYCLE_DIFF_FILE",
        "CANCEL_SWARM",
        "TOGGLE_VOICE",
    ]


@pytest.mark.asyncio
async def test_tui_bridge_async_lifecycle() -> None:
    bus = EventBus(fps_limit=120)
    app = TUIApp(event_bus=bus)

    app.start()
    assert app.is_running is True
    assert app.bridge._bridge_task is not None

    # Publish an event while bridge is running
    bus.publish_nowait(
        EventTopic.TOKEN_STREAM,
        TokenStreamPayload(agent_id="arch", delta="Async stream", is_complete=True),
    )

    # Let bridge consume frame
    await asyncio.sleep(0.04)

    assert len(app.chat_pane.messages) == 1
    assert app.chat_pane.messages[0]["content"] == "Async stream"

    app.stop()
    assert app.is_running is False
    assert app.bridge.subscription.is_closed is True
