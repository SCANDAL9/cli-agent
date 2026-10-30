# Core Architecture & Performance Specification

## 1. Problem Statement & Baseline Analysis

In the current implementation of [`VertexAgentClient`](../../src/cli_agent/vertex.py#L18-L149):
1. **Blocking Subprocess Execution:** `_get_token()` invokes `subprocess.run(["gcloud", "auth", "print-access-token"])` synchronously inside an asynchronous workflow. Each invocation spawns a new Python process that invokes the gcloud CLI, incurring 400ms to 1200ms overhead on every single agent iteration.
2. **Lack of Token Caching:** Google Cloud OAuth2 access tokens have a lifetime of 3600 seconds (1 hour). Spawning the subprocess repeatedly per turn causes unnecessary CPU utilization and latency spikes.
3. **Monolithic Response Blocking:** Responses are requested via `:generateContent`. The user receives no feedback during lengthy generation turns until the entire text block completes.
4. **No Resiliency Handling:** Network hiccups or HTTP 429 (quota rate limits) / HTTP 503 (service unavailable) trigger immediate unhandled exceptions, crashing the active REPL session.

---

## 2. Proposed Architectural Design

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as REPL Interface
    participant Agent as Agent Orchestrator
    participant TP as TokenProvider (Cached)
    participant API as Vertex AI (:streamGenerateContent)

    User->>CLI: Enters message / prompt
    CLI->>Agent: chat(messages)
    Agent->>TP: get_token()
    alt Token valid (< 50 min old)
        TP-->>Agent: Cached Token
    else Token expired or missing
        TP->>TP: Refresh token (async)
        TP-->>Agent: Fresh Token
    end
    Agent->>API: POST :streamGenerateContent (SSE Stream)
    loop Server-Sent Events
        API-->>Agent: Chunk (delta text / function call)
        alt is Text Part
            Agent-->>CLI: Yield delta chunk (Real-time render)
        else is Function Call Part
            Agent->>Agent: Buffer function call parameters
        end
    end
    alt Tool calls present
        Agent->>Agent: Execute tools concurrently
        Agent->>API: Send tool results & continue stream
    else Completed
        Agent-->>CLI: Finalize Markdown render
    end
```

---

## 3. Detailed Component Implementation

### 3.1. Asynchronous Token Provider with Proactive Invalidation

A singleton or injected provider managing OAuth2 credentials with automatic refresh and in-memory TTL caching.

```python
import asyncio
import time
from typing import Optional


class TokenProvider:
    """Manages GCP OAuth2 access tokens with in-memory caching and non-blocking refresh."""

    def __init__(self, ttl_buffer_seconds: int = 300):
        self._cached_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._ttl_buffer: int = ttl_buffer_seconds
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            now = time.time()
            if self._cached_token and now < (self._expires_at - self._ttl_buffer):
                return self._cached_token

            self._cached_token = await self._fetch_token()
            # Standard gcloud access tokens expire after 3600s
            self._expires_at = now + 3600
            return self._cached_token

    async def _fetch_token(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "gcloud",
            "auth",
            "print-access-token",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"GCloud authentication failed: {stderr.decode().strip()}")
        token = stdout.decode().strip()
        if not token:
            raise RuntimeError("Received empty access token from gcloud.")
        return token
```

### 3.2. Streaming Content Integration (`:streamGenerateContent`)

Switch endpoint from `:generateContent` to `:streamGenerateContent?alt=sse` to process Server-Sent Events incrementally.

Key Benefits:
- **Time to First Token (TTFT):** Drops from >2000ms to ~350ms.
- **Visual Feedback:** User perceives instant responsiveness through live streaming tokens.
- **Handling Mixed Payloads:** Streaming payloads may yield partial text chunks or full function call objects.

```python
async def stream_chat_turn(
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    payload: dict,
    on_chunk: Callable[[str], None],
) -> tuple[str, list[dict]]:
    """Streams model output via SSE, yielding text fragments to UI while aggregating tool calls."""
    collected_text = []
    function_calls = []

    async with client.stream(
        "POST", f"{endpoint}?alt=sse", headers=headers, json=payload, timeout=60.0
    ) as response:
        if response.status_code != 200:
            err_body = await response.aread()
            raise RuntimeError(f"Vertex API error ({response.status_code}): {err_body.decode()}")

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw_data = line[len("data: ") :].strip()
            if not raw_data:
                continue

            chunk = json.loads(raw_data)
            candidates = chunk.get("candidates", [])
            if not candidates:
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    text_delta = p["text"]
                    collected_text.append(text_delta)
                    on_chunk(text_delta)
                elif "functionCall" in p:
                    function_calls.append(p["functionCall"])

    return "".join(collected_text), function_calls
```

### 3.3. HTTP Resiliency: Exponential Backoff & Connection Pooling

- Implement persistent `httpx.AsyncClient(http2=True, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10))` lifecycle.
- Add retries with jitter for `429 Too Many Requests`, `503 Service Unavailable`, and network resets (`httpx.TransportError`).

---

## 4. Verification & Testing Strategy

1. **Token Cache Unit Test:**
   - Verify token is requested once across 10 concurrent tasks.
   - Verify refresh happens after TTL expiration.
2. **Stream Parser Unit Test:**
   - Mock SSE stream with alternating text chunks and tool invocation payloads.
   - Assert all delta callbacks are triggered in sequence.
3. **Resilience Test:**
   - Simulate 429 response status code with `Retry-After` header.
   - Ensure exponential backoff adheres to expected delay before resolving.
