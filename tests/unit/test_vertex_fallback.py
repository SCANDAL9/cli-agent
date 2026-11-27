"""Unit tests for VertexAgentClient stream_chat fallback router failover on 429."""

import json

import httpx
import pytest

from cli_agent.circuit_breaker import (
    CircuitState,
    FallbackRouter,
    ModelEndpoint,
    QuotaCircuitBreaker,
)
from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import (
    VertexAgentClient,
    _capture_tool_edit_snapshot,
    _notify_tool_result,
)


class _MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-token-fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_chat_fallback_on_429(httpx_mock) -> None:
    breaker = QuotaCircuitBreaker()
    ep_tier1 = ModelEndpoint("gemini-3.8-flash", "global", tier=1)
    ep_tier2 = ModelEndpoint("gemini-2.5-pro", "us-central1", tier=2)
    router = FallbackRouter(breaker, [ep_tier1, ep_tier2])

    # Tier 1 returns 429 rate limit
    httpx_mock.add_response(
        status_code=429,
        text='{"error": {"message": "Quota exceeded for gemini-3.8-flash"}}',
    )
    # Tier 2 returns successful 200 SSE stream
    sse_data = (
        'data: {"candidates": [{"content": {"parts": [{"text": "Fallback "}]}}]}\n\n'
        'data: {"candidates": [{"content": {"parts": [{"text": "succeeded!"}]}}]}\n\n'
        "data: [DONE]\n\n"
    )
    httpx_mock.add_response(
        status_code=200,
        text=sse_data,
        headers={"content-type": "text/event-stream"},
    )

    client = VertexAgentClient(
        project_id="test-project",
        location="global",
        model_name="gemini-3.8-flash",
        token_provider=_MockFixedTokenProvider(),
        fallback_router=router,
        circuit_breaker=breaker,
    )

    messages = [{"role": "user", "parts": [{"text": "Hello fallback"}]}]
    tokens: list[str] = []
    async for chunk in client.stream_chat(messages):
        tokens.append(chunk)

    assert "".join(tokens) == "Fallback succeeded!"
    assert client.model_name == "gemini-2.5-pro"
    assert client.location == "us-central1"
    assert breaker.get_state(ep_tier1) == CircuitState.OPEN
    assert breaker.get_state(ep_tier2) == CircuitState.CLOSED


@pytest.mark.unit
def test_tool_edit_snapshot_reads_only_authorized_text_targets(tmp_path):
    file_path = tmp_path / "edited.py"
    file_path.write_text("value = 1\n", encoding="utf-8")

    snapshot = _capture_tool_edit_snapshot("edit_file", {"path": str(file_path)})

    assert snapshot == (str(file_path), "value = 1\n")
    assert _capture_tool_edit_snapshot("read_file", {"path": str(file_path)}) is None
    assert _capture_tool_edit_snapshot(
        "apply_code_action",
        {"path": str(file_path), "action_index": 0, "action_title": "Add import"},
    ) == (str(file_path), "value = 1\n")
    assert (
        _capture_tool_edit_snapshot(
            "apply_code_action", {"path": str(file_path), "line": 0, "character": 0}
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_blocks_unapproved_code_action_before_tool_execution(monkeypatch, tmp_path):
    from cli_agent import vertex as vertex_module
    from cli_agent.security import SecurityPolicy

    file_path = tmp_path / "source.py"
    file_path.write_text("value = 1\n", encoding="utf-8")
    call_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "apply_code_action",
                                "args": {
                                    "path": str(file_path),
                                    "line": 0,
                                    "character": 0,
                                    "server_name": "pyright",
                                    "action_index": 0,
                                    "action_title": "Add import",
                                    "selection_id": "f353c4e202ad12345fa0",
                                },
                            }
                        }
                    ]
                }
            }
        ]
    }
    final_payload = {"candidates": [{"content": {"parts": [{"text": "approval required"}]}}]}
    responses = [call_payload, final_payload]
    executed: list[str] = []

    def handler(_request):
        return httpx.Response(200, json=responses.pop(0))

    def execute(name, _arguments):
        executed.append(name)
        file_path.write_text("unsafe = True\n", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(vertex_module, "execute_tool", execute)
    client = VertexAgentClient(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        token_provider=_MockFixedTokenProvider(),
        security_policy=SecurityPolicy(auto_approve=False),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        output = await client.chat(
            [{"role": "user", "parts": [{"text": "apply the selected fix"}]}],
            tools_declarations=[],
            max_turns=2,
        )
    finally:
        await client._http_client.aclose()

    assert output == "approval required"
    assert executed == []
    assert file_path.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.unit
def test_tool_result_callback_renders_diff_without_changing_model_result(tmp_path):
    observed = []
    file_path = tmp_path / "a.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    before = _capture_tool_edit_snapshot("edit_file", {"path": str(file_path)})
    file_path.write_text("x = 2\n", encoding="utf-8")
    result = {"status": "success"}
    arguments = {"path": "a.py"}

    returned = _notify_tool_result(
        "edit_file",
        arguments,
        result,
        lambda *values: observed.append(values),
        before,
    )

    assert observed == [
        (
            "edit_file",
            arguments,
            {"path": str(file_path), "before": "x = 1\n", "after": "x = 2\n"},
        )
    ]
    assert returned == {"status": "success"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_chat_sends_file_diff_to_ui_callback_only(monkeypatch, tmp_path):
    from cli_agent import vertex as vertex_module

    file_path = tmp_path / "edited.py"
    file_path.write_text("value = 1\n", encoding="utf-8")
    call_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "edit_file",
                                "args": {
                                    "path": str(file_path),
                                    "target": "value = 1",
                                    "replacement": "value = 2",
                                },
                            }
                        }
                    ]
                }
            }
        ]
    }
    final_payload = {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
    responses = [
        f"data: {json.dumps(call_payload)}\n\ndata: [DONE]\n\n",
        f"data: {json.dumps(final_payload)}\n\ndata: [DONE]\n\n",
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=responses.pop(0),
            headers={"content-type": "text/event-stream"},
        )

    def execute(name, arguments):
        file_path.write_text("value = 2\n", encoding="utf-8")
        return {"status": "success", "path": str(file_path)}

    monkeypatch.setattr(vertex_module, "execute_tool", execute)
    client = VertexAgentClient(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        token_provider=_MockFixedTokenProvider(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    callbacks = []

    try:
        output = [
            part
            async for part in client.stream_chat(
                [{"role": "user", "parts": [{"text": "edit file"}]}],
                tools_declarations=[],
                on_tool_result=lambda *values: callbacks.append(values),
                max_turns=2,
            )
        ]
    finally:
        await client._http_client.aclose()

    assert output == ["done"]
    assert callbacks == [
        (
            "edit_file",
            {"path": str(file_path), "target": "value = 1", "replacement": "value = 2"},
            {"path": str(file_path), "before": "value = 1\n", "after": "value = 2\n"},
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_sends_file_diff_to_ui_callback_only(monkeypatch, tmp_path):
    from cli_agent import vertex as vertex_module

    file_path = tmp_path / "edited.py"
    file_path.write_text("value = 1\n", encoding="utf-8")
    responses = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "edit_file",
                                    "args": {
                                        "path": str(file_path),
                                        "target": "value = 1",
                                        "replacement": "value = 2",
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        },
        {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
    ]

    def handler(request):
        return httpx.Response(200, json=responses.pop(0))

    def execute(name, arguments):
        file_path.write_text("value = 2\n", encoding="utf-8")
        return {"status": "success", "path": str(file_path)}

    monkeypatch.setattr(vertex_module, "execute_tool", execute)
    client = VertexAgentClient(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        token_provider=_MockFixedTokenProvider(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    callbacks = []

    try:
        output = await client.chat(
            [{"role": "user", "parts": [{"text": "edit file"}]}],
            tools_declarations=[],
            on_tool_result=lambda *values: callbacks.append(values),
            max_turns=2,
        )
    finally:
        await client._http_client.aclose()

    assert output == "done"
    assert callbacks == [
        (
            "edit_file",
            {"path": str(file_path), "target": "value = 1", "replacement": "value = 2"},
            {"path": str(file_path), "before": "value = 1\n", "after": "value = 2\n"},
        )
    ]


@pytest.mark.unit
def test_edit_snapshot_skips_invalid_unwritable_and_unreadable_targets(monkeypatch, tmp_path):
    from cli_agent import vertex as vertex_module

    assert _capture_tool_edit_snapshot("read_file", {"path": "anything.py"}) is None
    assert _capture_tool_edit_snapshot("edit_file", {}) is None
    assert _capture_tool_edit_snapshot("edit_file", {"path": 4}) is None

    monkeypatch.setattr(vertex_module, "_check_write_allowed", lambda path: "read only")
    assert _capture_tool_edit_snapshot("edit_file", {"path": str(tmp_path / "file.py")}) is None

    monkeypatch.setattr(vertex_module, "_check_write_allowed", lambda path: None)
    missing_path = tmp_path / "missing.py"
    assert _capture_tool_edit_snapshot("write_file", {"path": str(missing_path)}) == (
        str(missing_path),
        "",
    )

    directory = tmp_path / "directory"
    directory.mkdir()
    assert _capture_tool_edit_snapshot("edit_file", {"path": str(directory)}) is None

    binary_path = tmp_path / "binary.py"
    binary_path.write_bytes(b"\xff")
    assert _capture_tool_edit_snapshot("edit_file", {"path": str(binary_path)}) is None

    monkeypatch.setattr(
        vertex_module,
        "_resolve_tool_path",
        lambda path: (_ for _ in ()).throw(PermissionError("outside workspace")),
    )
    assert _capture_tool_edit_snapshot("edit_file", {"path": "../outside.py"}) is None


@pytest.mark.unit
def test_edit_result_notifier_preserves_failed_unchanged_or_unreadable_results(tmp_path):
    observed = []
    file_path = tmp_path / "same.py"
    file_path.write_text("same\n", encoding="utf-8")
    previous = (str(file_path), "same\n")

    def callback(*values):
        observed.append(values)

    success = {"status": "success"}

    assert _notify_tool_result("edit_file", {}, success, None, previous) is success
    assert _notify_tool_result("edit_file", {}, success, callback, None) is success
    assert _notify_tool_result("edit_file", {}, "not-a-mapping", callback, previous) == (
        "not-a-mapping"
    )
    failed = {"error": "permission denied"}
    assert _notify_tool_result("edit_file", {}, failed, callback, previous) is failed
    assert _notify_tool_result("edit_file", {}, success, callback, previous) is success
    assert observed == []

    file_path.unlink()
    assert _notify_tool_result("edit_file", {}, success, callback, previous) is success
    assert observed == []

    file_path.write_text("changed\n", encoding="utf-8")
    assert (
        _notify_tool_result(
            "edit_file",
            {},
            success,
            lambda *values: (_ for _ in ()).throw(RuntimeError("renderer failed")),
            previous,
        )
        is success
    )
