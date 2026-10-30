"""
Unit tests for the Universal Model Context Protocol (MCP) Client module.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cli_agent.mcp import (
    MCPServerClient,
    MCPServerConfig,
    MCPTool,
    _expand_env_vars,
    convert_json_schema_to_gemini,
    set_active_mcp_registry,
)
from cli_agent.tools import execute_tool


@pytest.mark.unit
class TestMCPSchemaAndEnv:
    def test_expand_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_TOKEN", "mock_dummy_token_123")
        monkeypatch.setenv("TEST_PORT", "5432")

        env_dict = {
            "AUTH_HEADER": "Bearer ${TEST_TOKEN}",
            "DB_PORT": "${TEST_PORT}",
            "FALLBACK": "${UNSET_VAR_EXAMPLE}",
        }
        expanded = _expand_env_vars(env_dict)
        assert expanded["AUTH_HEADER"] == "Bearer mock_dummy_token_123"
        assert expanded["DB_PORT"] == "5432"
        assert expanded["FALLBACK"] == ""

    def test_convert_json_schema_to_gemini(self) -> None:
        raw_schema = {
            "type": "object",
            "description": "Create a new issue in GitHub",
            "properties": {
                "title": {"type": "string", "description": "Issue title"},
                "number": {"type": "integer"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "details": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                    },
                },
            },
            "required": ["title"],
        }

        gemini_schema = convert_json_schema_to_gemini(raw_schema)
        assert gemini_schema["type"] == "OBJECT"
        assert gemini_schema["description"] == "Create a new issue in GitHub"
        assert gemini_schema["required"] == ["title"]

        props = gemini_schema["properties"]
        assert props["title"]["type"] == "STRING"
        assert props["number"]["type"] == "INTEGER"
        assert props["labels"]["type"] == "ARRAY"
        assert props["labels"]["items"]["type"] == "STRING"
        assert props["details"]["type"] == "OBJECT"
        assert props["details"]["properties"]["severity"]["type"] == "STRING"

        # Empty schema
        assert convert_json_schema_to_gemini({})["type"] == "OBJECT"


@pytest.mark.unit
class TestMCPRegistryConfig:
    def test_load_configs_from_file(self, tmp_path: Path) -> None:
        from cli_agent.mcp import MCPRegistry

        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "token-1"},
                },
                "postgres": {
                    "command": "uvx",
                    "args": ["mcp-server-postgres"],
                },
            }
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        registry = MCPRegistry(config_path=config_file)
        configs = registry.load_configs()
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert "github" in names
        assert "postgres" in names

    def test_load_configs_non_existent_file(self, tmp_path: Path) -> None:
        from cli_agent.mcp import MCPRegistry

        registry = MCPRegistry(config_path=tmp_path / "absent.json")
        assert registry.load_configs() == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestMCPServerClientMocked:
    async def test_mcp_client_handshake_and_tool_call(self) -> None:
        from unittest.mock import MagicMock

        cfg = MCPServerConfig(name="mock_srv", command="mock-cmd", args=[])
        client = MCPServerClient(cfg)

        queue: asyncio.Queue[bytes] = asyncio.Queue()

        def handle_stdin_write(data: bytes) -> None:
            parsed = json.loads(data.decode("utf-8"))
            req_id = parsed.get("id")
            if req_id is None:
                return
            method = parsed.get("method")
            if method == "initialize":
                res = {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05"}}
            elif method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echoes text back",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"msg": {"type": "string"}},
                                },
                            }
                        ]
                    },
                }
            elif method == "tools/call":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"text": "Echo: hello"}]},
                }
            else:
                res = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            queue.put_nowait(json.dumps(res).encode("utf-8") + b"\n")

        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock(side_effect=handle_stdin_write)
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=queue.get)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await client.connect()
            assert client.is_connected is True

            tools = await client.list_tools()
            assert len(tools) == 1
            assert tools[0].full_name == "mcp__mock_srv__echo"
            assert tools[0].description == "Echoes text back"

            call_res = await client.call_tool("echo", {"msg": "hello"})
            assert call_res["status"] == "success"
            assert "Echo: hello" in call_res["content"]

            await client.disconnect()
            assert client.is_connected is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestMCPRegistryAndToolDispatch:
    async def test_registry_tool_declarations_and_dispatch(self) -> None:
        from cli_agent.mcp import MCPRegistry

        registry = MCPRegistry()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.call_tool = AsyncMock(
            return_value={"status": "success", "content": "Database row created"}
        )

        registry.servers["db"] = mock_client
        registry.tools["mcp__db__insert_record"] = MCPTool(
            server_name="db",
            name="insert_record",
            full_name="mcp__db__insert_record",
            description="Inserts a DB row",
            input_schema={"type": "object", "properties": {"table": {"type": "string"}}},
        )

        # Test Gemini declarations translation
        decls = registry.get_gemini_declarations()
        assert len(decls) == 1
        assert decls[0]["name"] == "mcp__db__insert_record"
        assert "[db]" in decls[0]["description"]
        assert decls[0]["parameters"]["type"] == "OBJECT"

        # Test tool execution dispatch
        res = await registry.execute_tool("mcp__db__insert_record", {"table": "users"})
        assert res["status"] == "success"
        assert "Database row created" in res["content"]

        # Test unreg tool
        err_res = await registry.execute_tool("mcp__unknown__tool", {})
        assert "is not registered" in err_res["error"]

    async def test_tools_py_routing_mcp(self) -> None:
        from cli_agent.mcp import MCPRegistry

        registry = MCPRegistry()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.call_tool = AsyncMock(
            return_value={"status": "success", "content": "MCP Docker container listed"}
        )

        registry.servers["docker"] = mock_client
        registry.tools["mcp__docker__ps"] = MCPTool(
            server_name="docker",
            name="ps",
            full_name="mcp__docker__ps",
            description="List containers",
            input_schema={},
        )

        set_active_mcp_registry(registry)
        try:
            res = execute_tool("mcp__docker__ps", {})
            # If coroutine, await it
            if hasattr(res, "__await__"):
                res = await res
            assert res.get("status") == "success"
            assert "MCP Docker" in res.get("content", "")

            # Tool with no active registry
            set_active_mcp_registry(None)
            unreg = execute_tool("mcp__docker__ps", {})
            assert "No active MCP registry" in unreg.get("error", "")
        finally:
            set_active_mcp_registry(None)
