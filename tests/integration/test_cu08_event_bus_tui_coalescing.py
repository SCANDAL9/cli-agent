"""
Integration test for CU-08: High-Frequency Async EventBus Telemetry & 60 FPS Bi-Directional TUI Coalescing
(Wave 5 Milestone W5.1 & W5.2).
"""

from __future__ import annotations

import pytest

from cli_agent.circuit_breaker import CircuitState
from cli_agent.events import (
    CircuitHealthPayload,
    EventBus,
    EventTopic,
    SubprocLogPayload,
    SwarmNodeStatePayload,
    TokenStreamPayload,
    WorktreeDiffPayload,
    coalesce_frame_events,
)
from cli_agent.tui import TUIApp, TUIEventBusBridge


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU08EventBusTUICoalescing:
    async def test_high_frequency_telemetry_storm_coalesces_cleanly(self) -> None:
        bus = EventBus(fps_limit=60)
        sub = bus.subscribe()  # Wildcard subscription
        app = TUIApp()
        bridge = TUIEventBusBridge(app=app, event_bus=bus)

        # 1. Telemetry Storm: 100 token chunks across two workers
        for i in range(50):
            bus.publish_nowait(
                EventTopic.TOKEN_STREAM,
                TokenStreamPayload(agent_id="w1", delta=f"tok1_{i} "),
            )
            bus.publish_nowait(
                EventTopic.TOKEN_STREAM,
                TokenStreamPayload(agent_id="w2", delta=f"tok2_{i} "),
            )

        # 2. 6 rapid swarm node transitions
        statuses = ["PENDING", "READY", "RUNNING", "COMPLETED"]
        for s in statuses:
            bus.publish_nowait(
                EventTopic.SWARM_NODE_STATE,
                SwarmNodeStatePayload(plan_id="p1", node_id="node_x", role="worker", status=s),
            )

        # 3. 5 rapid diff updates
        for d in range(5):
            bus.publish_nowait(
                EventTopic.WORKTREE_DIFF_UPDATED,
                WorktreeDiffPayload(
                    worktree_id="wt-1",
                    branch_name="swarm/p1/node_x",
                    file_path="src/app.py",
                    diff_text=f"+ line {d}\n",
                    insertions=d,
                ),
            )

        # 4. 20 log lines from LSP and TDD
        for k in range(20):
            bus.publish_nowait(
                EventTopic.SUBPROC_LOG,
                SubprocLogPayload(source="LSP" if k % 2 == 0 else "TDD", line=f"log message {k}"),
            )

        # 5. Circuit health transitions
        bus.publish_nowait(
            EventTopic.CIRCUIT_HEALTH,
            CircuitHealthPayload(
                model_id="gemini-4-argon",
                region="us-central1",
                tier=1,
                state=CircuitState.CLOSED,
            ),
        )
        bus.publish_nowait(
            EventTopic.CIRCUIT_HEALTH,
            CircuitHealthPayload(
                model_id="gemini-4-argon",
                region="us-central1",
                tier=1,
                state=CircuitState.HALF_OPEN,
            ),
        )

        # Drain raw burst
        raw_events = sub.drain()
        assert len(raw_events) == 100 + 4 + 5 + 20 + 2

        # Coalesce frame
        coalesced = coalesce_frame_events(raw_events)

        # Assert token streams coalesced into exactly 2 envelopes (one per worker)
        token_envs = [e for e in coalesced if e.topic == EventTopic.TOKEN_STREAM]
        assert len(token_envs) == 2
        w1_token = next(e for e in token_envs if e.payload.agent_id == "w1")  # type: ignore[union-attr]
        assert len(w1_token.payload.delta.split()) == 50  # type: ignore[union-attr]

        # Assert only latest node_state is preserved
        node_envs = [e for e in coalesced if e.topic == EventTopic.SWARM_NODE_STATE]
        assert len(node_envs) == 1
        assert node_envs[0].payload.status == "COMPLETED"  # type: ignore[union-attr]

        # Assert only latest diff_update is preserved
        diff_envs = [e for e in coalesced if e.topic == EventTopic.WORKTREE_DIFF_UPDATED]
        assert len(diff_envs) == 1
        assert diff_envs[0].payload.insertions == 4  # type: ignore[union-attr]

        # Assert chronological order of all 20 subproc logs is preserved
        log_envs = [e for e in coalesced if e.topic == EventTopic.SUBPROC_LOG]
        assert len(log_envs) == 20
        assert log_envs[0].payload.line == "log message 0"  # type: ignore[union-attr]
        assert log_envs[19].payload.line == "log message 19"  # type: ignore[union-attr]

        # Apply coalesced frame through TUIEventBusBridge
        bridge.apply_coalesced_batch(coalesced)
        assert app.status_hud.circuit_snapshot is not None
        assert app.status_hud.circuit_snapshot.state == CircuitState.HALF_OPEN

        # Bi-directional control verification: dispatch CANCEL_SWARM
        ctrl_sub = bus.subscribe(EventTopic.TUI_CONTROL)
        bridge.publish_control("CANCEL_SWARM", argument="user requested cancel")
        ctrl_events = ctrl_sub.drain()
        assert len(ctrl_events) == 1
        assert ctrl_events[0].payload.action == "CANCEL_SWARM"  # type: ignore[union-attr]
        assert ctrl_events[0].payload.argument == "user requested cancel"  # type: ignore[union-attr]
