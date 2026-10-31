"""
Vertex AI Gemini 3.8 Flash Client with Tool Calling Agent Loop.
Authenticates via cached OAuth2 tokens and interacts with Google Cloud Vertex AI global endpoint.
Supports non-blocking async execution, streaming SSE, and automated tool dispatching.
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx

from .circuit_breaker import (
    ExhaustedFallbackChainError,
    FallbackRouter,
    ModelEndpoint,
    QuotaCircuitBreaker,
)
from .code_graph import CodeGraphStore
from .memory import MemoryStore, resolve_project_key
from .models import detect_publisher
from .security import SecurityPolicy
from .token_provider import TokenProvider
from .tools import (
    GEMINI_TOOLS_DECLARATIONS,
    execute_tool,
    set_code_graph_store,
    set_memory_store,
)

DEFAULT_PROJECT = os.getenv("GCP_PROJECT", "my-gcp-project")
DEFAULT_LOCATION = os.getenv("GEMINI_LOCATION", "global")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class VertexAgentClient:
    def __init__(
        self,
        project_id: str = DEFAULT_PROJECT,
        location: str = DEFAULT_LOCATION,
        model_name: str = DEFAULT_MODEL,
        memory_store: MemoryStore | None = None,
        code_graph_store: CodeGraphStore | None = None,
        token_provider: TokenProvider | None = None,
        security_policy: SecurityPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        fallback_router: FallbackRouter | None = None,
        circuit_breaker: QuotaCircuitBreaker | None = None,
    ):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.memory = memory_store or MemoryStore()
        set_memory_store(self.memory)
        self.code_graph_store = code_graph_store or CodeGraphStore()
        set_code_graph_store(self.code_graph_store)
        self.token_provider = token_provider or TokenProvider()
        self.security_policy = security_policy or SecurityPolicy(auto_approve=True)
        self._http_client = http_client
        self.publisher = detect_publisher(self.model_name)
        self._configure_endpoints()
        self.circuit_breaker = circuit_breaker
        self.fallback_router = fallback_router
        self.system_instruction = self._build_system_instruction()

    def _configure_endpoints(self) -> None:
        """Sets up API endpoints based on the active model publisher."""
        raw_name = self.model_name
        clean_model = raw_name.split("/")[-1].split("@")[0]

        if self.publisher == "anthropic":
            loc_prefix = f"{self.location}-" if self.location != "global" else ""
            base = f"https://{loc_prefix}aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/anthropic/models/{clean_model}"
            self.endpoint = f"{base}:rawPredict"
            self.stream_endpoint = f"{base}:streamRawPredict"
        elif self.publisher in ("xai", "mistralai"):
            loc_prefix = f"{self.location}-" if self.location != "global" else ""
            base = f"https://{loc_prefix}aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/{self.publisher}/models/{clean_model}"
            self.endpoint = f"{base}:rawPredict"
            self.stream_endpoint = f"{base}:streamRawPredict"
        else:
            base = f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{clean_model}"
            self.endpoint = f"{base}:generateContent"
            self.stream_endpoint = f"{base}:streamGenerateContent"

    def switch_model(self, model_name: str) -> None:
        """Dynamically hot-swaps the underlying model and publisher endpoints."""
        self.model_name = model_name
        self.publisher = detect_publisher(model_name)
        self._configure_endpoints()

    def _build_system_instruction(self) -> str:
        project_key = resolve_project_key()
        recent_mems = self.memory.get_recent_memories(limit=8, project_key=project_key)
        memory_context = ""
        if recent_mems:
            memory_context = f"\n### RECENT PERSISTENT MEMORIES ({project_key} & GLOBAL):\n"
            for m in recent_mems:
                memory_context += f"- [{m['type'].upper()}] {m['title']}: {m['content']}\n"

        return f"""You are 'cli-agent', an elite Senior Software Architect and Autonomous Developer Assistant running directly on the user's Linux workstation.
You have access to powerful local system tools and persistent SQLite memory.

Your Core Capabilities:
1. Inspecting local files: Use `read_file` to read source code or configs.
2. Exploring projects: Use `list_directory` to understand file trees and layouts.
3. Code search: Use `search_code` to locate functions, classes, imports, or patterns across directories.
4. Active Code Modification:
   - Use `write_file` to create new files or overwrite existing ones.
   - Use `edit_file` to perform exact, surgical string replacements in existing files.
5. System & Terminal Execution:
   - Use `run_command` to execute tests, linters, git commands, and shell utilities.
6. Persistent Memory (like Engram):
   - Use `search_memory` when the user asks about previous decisions, past discussions, or prior setup.
   - Use `save_memory` proactively when an important architectural decision is made, a bug is solved with non-obvious root cause, or a user convention is established.
7. Structural Code Intelligence (Tree-sitter AST & Code Graph):
   - Use `get_symbol_outline` to inspect a file's skeleton (classes, methods, docstrings, signatures) before reading full implementations (saves ~90% tokens).
   - Use `find_symbol_definition` to jump directly to where a class, method, or function is defined across the codebase.
   - Use `find_symbol_references` to trace all callers, imports, and usages before refactoring.

Guidelines:
- When asked about a file or code, ALWAYS read it first with `read_file` instead of guessing.
- Before modifying a file, read the relevant lines to ensure target replacements are exact.
- Be concise, direct, technically rigorous, and pragmatic.
- Provide answers formatted in clean Markdown.
{memory_context}
"""

    async def _get_headers(self) -> dict[str, str]:
        token = await self.token_provider.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        max_turns: int = 10,
        tools_declarations: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        """Executes the agent tool-calling loop until a final text response is produced."""
        headers = await self._get_headers()
        working_contents = list(messages)
        active_instruction = system_instruction or self.system_instruction
        if tools_declarations is not None:
            active_tools = tools_declarations
        else:
            from .mcp import get_active_mcp_registry

            reg = get_active_mcp_registry()
            mcp_decls = reg.get_gemini_declarations() if reg else []
            active_tools = list(GEMINI_TOOLS_DECLARATIONS) + mcp_decls

        client_owner = False
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            client_owner = True

        try:
            for _ in range(max_turns):
                payload: dict[str, Any] = {
                    "contents": working_contents,
                    "system_instruction": {"parts": [{"text": active_instruction}]},
                    "generationConfig": {"temperature": 0.2},
                }
                if active_tools:
                    payload["tools"] = [{"function_declarations": active_tools}]

                resp = await client.post(self.endpoint, headers=headers, json=payload)
                if resp.status_code == 429:
                    err_msg = ""
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", {}).get("message", "")
                    except Exception:
                        pass
                    remediation = f"\nDetail: {err_msg}" if err_msg else ""
                    tip = (
                        "\nTip: Third-party models (Claude, Grok) require requesting quota in GCP Quotas console. "
                        "Switch back with '/model gemini-3.8-flash'."
                        if self.publisher != "google"
                        else ""
                    )
                    raise RuntimeError(
                        f"Vertex AI rate limit exceeded (HTTP 429).{remediation}{tip}"
                    )
                if resp.status_code != 200:
                    raise RuntimeError(f"Vertex AI API error ({resp.status_code}): {resp.text}")

                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "No response candidates returned by model."

                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                # Check for tool/function calls
                function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

                if not function_calls:
                    # Model returned pure text response
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    return "".join(text_parts).strip()

                # Record model's tool call turn to conversation history
                working_contents.append(content)

                # Execute all requested functions
                tool_response_parts = []
                for fc in function_calls:
                    fn_name = fc["name"]
                    fn_args = fc.get("args", {})

                    if on_tool_call:
                        on_tool_call(fn_name, fn_args)

                    allowed, reason = self.security_policy.check_permission(fn_name, fn_args)
                    if not allowed:
                        result = {"error": f"Tool execution rejected by security policy: {reason}"}
                    else:
                        tool_res = execute_tool(fn_name, fn_args)
                        result = await tool_res if asyncio.iscoroutine(tool_res) else tool_res

                    tool_response_parts.append(
                        {
                            "functionResponse": {
                                "name": fn_name,
                                "response": {"result": result},
                            }
                        }
                    )

                # Send function responses back to the model
                working_contents.append({"role": "user", "parts": tool_response_parts})

            return "Agent reached maximum tool iterations limit."
        finally:
            if client_owner:
                await client.aclose()

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        max_turns: int = 50,
        tools_declarations: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Executes the agent loop while yielding text chunks incrementally via SSE.
        Executes tool calls transparently during the stream.
        """
        headers = await self._get_headers()
        working_contents = list(messages)
        active_instruction = system_instruction or self.system_instruction
        if tools_declarations is not None:
            active_tools = tools_declarations
        else:
            from .mcp import get_active_mcp_registry

            reg = get_active_mcp_registry()
            mcp_decls = reg.get_gemini_declarations() if reg else []
            active_tools = list(GEMINI_TOOLS_DECLARATIONS) + mcp_decls

        client_owner = False
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            client_owner = True

        try:
            from .context import prune_tool_responses

            for _ in range(max_turns):
                pruned_contents = prune_tool_responses(working_contents, max_chars=4000)
                payload: dict[str, Any] = {
                    "contents": pruned_contents,
                    "system_instruction": {"parts": [{"text": active_instruction}]},
                    "generationConfig": {"temperature": 0.2},
                }
                if active_tools:
                    payload["tools"] = [{"function_declarations": active_tools}]

                function_calls: list[dict[str, Any]] = []
                full_content_parts: list[dict[str, Any]] = []

                try:
                    async with client.stream(
                        "POST",
                        f"{self.stream_endpoint}?alt=sse",
                        headers=headers,
                        json=payload,
                        timeout=DEFAULT_TIMEOUT,
                    ) as response:
                        if response.status_code in (404, 429, 503):
                            err_body = await response.aread()
                            err_text = err_body.decode(errors="replace")
                            if (
                                self.circuit_breaker is not None
                                and self.fallback_router is not None
                            ):
                                current_ep = ModelEndpoint(self.model_name, self.location)
                                self.circuit_breaker.record_failure(
                                    current_ep,
                                    status_code=response.status_code,
                                    error_body=err_text,
                                    headers=dict(response.headers),
                                )
                                try:
                                    fallback_ep = self.fallback_router.select_endpoint()
                                except ExhaustedFallbackChainError:
                                    fallback_ep = None

                                if fallback_ep is not None and fallback_ep.key != current_ep.key:
                                    self.location = fallback_ep.region
                                    self.switch_model(fallback_ep.model_id)
                                    continue

                            if response.status_code == 429:
                                err_detail = ""
                                try:
                                    err_json = json.loads(err_text)
                                    err_detail = err_json.get("error", {}).get("message", "")
                                except Exception:
                                    err_detail = err_text
                                remediation = f"\nDetail: {err_detail}" if err_detail else ""
                                tip = (
                                    "\nTip: Third-party models (Claude, Grok) require requesting quota in GCP Quotas console. "
                                    "Switch back with '/model gemini-3.8-flash'."
                                    if self.publisher != "google"
                                    else ""
                                )
                                raise RuntimeError(
                                    f"Vertex AI rate limit exceeded (HTTP 429).{remediation}{tip}"
                                )
                            raise RuntimeError(
                                f"Vertex AI API error ({response.status_code}): {err_text}"
                            )
                        if response.status_code != 200:
                            err_body = await response.aread()
                            raise RuntimeError(
                                f"Vertex AI API error ({response.status_code}): {err_body.decode()}"
                            )

                        if self.circuit_breaker is not None:
                            self.circuit_breaker.record_success(
                                ModelEndpoint(self.model_name, self.location)
                            )

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if not raw or raw == "[DONE]":
                                continue

                            chunk_data = json.loads(raw)
                            candidates = chunk_data.get("candidates", [])
                            if not candidates:
                                continue

                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                full_content_parts.append(p)
                                if "text" in p:
                                    yield p["text"]
                                elif "functionCall" in p:
                                    function_calls.append(p["functionCall"])
                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    if self.circuit_breaker is not None and self.fallback_router is not None:
                        current_ep = ModelEndpoint(self.model_name, self.location)
                        self.circuit_breaker.record_failure(
                            current_ep,
                            status_code=504,
                            error_body=str(net_err),
                            headers={},
                        )
                        try:
                            fallback_ep = self.fallback_router.select_endpoint()
                        except ExhaustedFallbackChainError:
                            fallback_ep = None

                        if fallback_ep is not None and fallback_ep.key != current_ep.key:
                            self.location = fallback_ep.region
                            self.switch_model(fallback_ep.model_id)
                            continue
                    raise RuntimeError(f"Vertex AI network/timeout error: {net_err}") from net_err

                if not function_calls:
                    return

                # Record tool call turn to conversation history
                working_contents.append({"role": "model", "parts": full_content_parts})

                tool_response_parts = []
                for fc in function_calls:
                    fn_name = fc["name"]
                    fn_args = fc.get("args", {})

                    if on_tool_call:
                        on_tool_call(fn_name, fn_args)

                    allowed, reason = self.security_policy.check_permission(fn_name, fn_args)
                    if not allowed:
                        result = {"error": f"Tool execution rejected by security policy: {reason}"}
                    else:
                        tool_res = execute_tool(fn_name, fn_args)
                        result = await tool_res if asyncio.iscoroutine(tool_res) else tool_res

                    tool_response_parts.append(
                        {
                            "functionResponse": {
                                "name": fn_name,
                                "response": {"result": result},
                            }
                        }
                    )

                working_contents.append({"role": "user", "parts": tool_response_parts})

        finally:
            if client_owner:
                await client.aclose()
