"""
Model Context Protocol (MCP) Client and Federation Registry for cli-agent.
Supports stdio JSON-RPC 2.0 communication, tool discovery, automatic schema
translation into Vertex AI Gemini function declarations, and runtime dispatch.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MCP_CONFIG_PATH = Path.home() / ".config" / "cli-agent" / "mcp.json"


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None


@dataclass
class MCPTool:
    server_name: str
    name: str
    full_name: str
    description: str
    input_schema: dict[str, Any]


def _expand_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Expands ${VAR_NAME} environment variables in configuration strings."""
    expanded: dict[str, str] = {}
    pattern = re.compile(r"\$\{([^}]+)\}")

    for k, v in env_dict.items():

        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        expanded[k] = pattern.sub(repl, v)
    return expanded


def convert_json_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Translates standard JSON Schema types into Vertex AI Gemini schema formats:
    e.g. 'string' -> 'STRING', 'integer' -> 'INTEGER', 'object' -> 'OBJECT'.
    """
    if not schema:
        return {"type": "OBJECT", "properties": {}}

    gemini_schema: dict[str, Any] = {}

    schema_type = schema.get("type", "object")
    if isinstance(schema_type, str):
        gemini_schema["type"] = schema_type.upper()
    elif isinstance(schema_type, list):
        # Pick primary type
        first = [t for t in schema_type if t != "null"]
        gemini_schema["type"] = first[0].upper() if first else "STRING"
    else:
        gemini_schema["type"] = "OBJECT"

    if "description" in schema:
        gemini_schema["description"] = schema["description"]

    if "properties" in schema and isinstance(schema["properties"], dict):
        gemini_schema["properties"] = {
            prop_name: convert_json_schema_to_gemini(prop_def)
            for prop_name, prop_def in schema["properties"].items()
        }

    if "required" in schema and isinstance(schema["required"], list):
        gemini_schema["required"] = schema["required"]

    if "items" in schema and isinstance(schema["items"], dict):
        gemini_schema["items"] = convert_json_schema_to_gemini(schema["items"])

    return gemini_schema


class MCPServerClient:
    """Manages a JSON-RPC 2.0 stdio subprocess connection to an MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self.is_connected = False

    async def connect(self) -> None:
        """Spawns server subprocess and performs protocol handshake."""
        env = os.environ.copy()
        env.update(_expand_env_vars(self.config.env))

        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._reader_task = asyncio.create_task(self._read_responses())

        # Protocol Handshake: initialize
        init_res = await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cli-agent", "version": "0.1.0"},
            },
        )
        if "error" in init_res:
            raise RuntimeError(
                f"MCP initialize failed for server '{self.config.name}': {init_res['error']}"
            )

        # Send notifications/initialized
        await self._send_notification("notifications/initialized", {})
        self.is_connected = True

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        raw = json.dumps(msg) + "\n"
        self.process.stdin.write(raw.encode("utf-8"))
        await self.process.stdin.drain()

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Sends a JSON-RPC 2.0 request and awaits the matched response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"Server '{self.config.name}' is not running.")

        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = future

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        raw = json.dumps(msg) + "\n"
        self.process.stdin.write(raw.encode("utf-8"))
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(req_id, None)

    async def _read_responses(self) -> None:
        """Continuously reads stdout JSON-RPC lines and fulfills pending futures."""
        if not self.process or not self.process.stdout:
            return

        while True:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                data = json.loads(text)
                req_id = data.get("id")
                if req_id is not None and req_id in self._pending_requests:
                    future = self._pending_requests[req_id]
                    if not future.done():
                        future.set_result(data)
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def list_tools(self) -> list[MCPTool]:
        """Queries the server for registered tools."""
        res = await self.send_request("tools/list")
        result = res.get("result", {})
        raw_tools = result.get("tools", [])

        tools: list[MCPTool] = []
        for t in raw_tools:
            name = t.get("name", "")
            full_name = f"mcp__{self.config.name}__{name}"
            tools.append(
                MCPTool(
                    server_name=self.config.name,
                    name=name,
                    full_name=full_name,
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Calls an MCP tool by its server-scoped name."""
        res = await self.send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        if "error" in res:
            return {"error": res["error"]}

        result = res.get("result", {})
        # Extract text content if present
        content_items = result.get("content", [])
        text_outputs = [item.get("text", "") for item in content_items if "text" in item]

        return {
            "status": "success",
            "content": "\n".join(text_outputs) if text_outputs else str(result),
            "isError": result.get("isError", False),
        }

    async def disconnect(self) -> None:
        """Terminates subprocess and cleans up background tasks."""
        self.is_connected = False
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


class MCPRegistry:
    """Manages active MCP servers and translates tool definitions for Gemini."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or DEFAULT_MCP_CONFIG_PATH
        self.servers: dict[str, MCPServerClient] = {}
        self.tools: dict[str, MCPTool] = {}

    def load_configs(self) -> list[MCPServerConfig]:
        """Loads server configurations from mcp.json file."""
        if not self.config_path.is_file():
            return []

        try:
            with open(self.config_path, encoding="utf-8") as f:
                payload = json.load(f)
            servers_raw = payload.get("mcpServers", {})
            configs: list[MCPServerConfig] = []
            for name, item in servers_raw.items():
                configs.append(
                    MCPServerConfig(
                        name=name,
                        command=item.get("command", ""),
                        args=item.get("args", []),
                        env=item.get("env", {}),
                        url=item.get("url"),
                    )
                )
            return configs
        except Exception:
            return []

    async def connect_all(self) -> int:
        """Connects to all configured servers and registers their tools."""
        configs = self.load_configs()
        connected_count = 0

        for cfg in configs:
            if not cfg.command:
                continue
            client = MCPServerClient(cfg)
            try:
                await client.connect()
                self.servers[cfg.name] = client
                server_tools = await client.list_tools()
                for tool in server_tools:
                    self.tools[tool.full_name] = tool
                connected_count += 1
            except Exception:
                pass

        return connected_count

    def get_gemini_declarations(self) -> list[dict[str, Any]]:
        """Generates Vertex AI Gemini tool declarations for all federated MCP tools."""
        declarations: list[dict[str, Any]] = []
        for tool in self.tools.values():
            param_schema = convert_json_schema_to_gemini(tool.input_schema)
            declarations.append(
                {
                    "name": tool.full_name,
                    "description": f"[{tool.server_name}] {tool.description}".strip(),
                    "parameters": param_schema,
                }
            )
        return declarations

    async def execute_tool(self, full_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatches an MCP tool call to the corresponding server."""
        tool = self.tools.get(full_name)
        if not tool:
            return {"error": f"MCP Tool '{full_name}' is not registered."}

        client = self.servers.get(tool.server_name)
        if not client or not client.is_connected:
            return {"error": f"MCP Server '{tool.server_name}' is not connected."}

        return await client.call_tool(tool.name, arguments)

    async def shutdown_all(self) -> None:
        """Disconnects all active server connections."""
        for client in self.servers.values():
            await client.disconnect()
        self.servers.clear()
        self.tools.clear()


# Global active registry singleton
_active_mcp_registry: MCPRegistry | None = None


def set_active_mcp_registry(registry: MCPRegistry | None) -> None:
    """Sets or clears the active MCP registry."""
    global _active_mcp_registry
    _active_mcp_registry = registry


def get_active_mcp_registry() -> MCPRegistry | None:
    """Returns the active MCP registry."""
    return _active_mcp_registry
