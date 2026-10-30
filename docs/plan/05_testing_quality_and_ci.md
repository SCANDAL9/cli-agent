# Testing, Quality & Continuous Integration Specification (UmlStudio Standard)

## 1. Architectural Philosophy & Quality Model

Following the quality framework established in **UmlStudio** (governed by **ISO/IEC 25010** and formal Use Case Traceability), `cli-agent`'s testing infrastructure is elevated from basic assertions to an enterprise-grade verification harness.

### Quality Characteristics Mapping (ISO/IEC 25010)

| ISO/IEC 25010 Characteristic | Application in `cli-agent` | Verification Strategy |
| :--- | :--- | :--- |
| **Functional Suitability** | 100% correct tool execution, exact file edits, schema-compliant Vertex AI payloads | Strict contract validation, JSON schema validation for function calls |
| **Reliability (Fault Tolerance)** | Graceful recovery from HTTP 429/503, invalid JSON responses, command timeouts | Error injection tests, network drop simulation |
| **Maintainability (Testability)** | High modularity, zero test interdependence, strict typing | Pytest fixtures, `mypy --strict`, 100% mocked external dependencies |
| **Performance Efficiency** | Non-blocking event loop, sub-millisecond in-memory token lookup | Benchmark tests on FTS5 search and TokenProvider concurrency |
| **Security** | Sandboxed command execution, human-in-the-loop permission gates | Security policy unit & boundary test suites |

---

## 2. Use Case (CU) Traceability Matrix

Every integration test directly verifies a discrete business Use Case (CU):

| ID | Use Case Name | Test Suite | Objective |
| :--- | :--- | :--- | :--- |
| **CU-01** | Interactive & One-Shot REPL Journey | `tests/integration/test_cu01_repl_journey.py` | Verify multi-turn user conversation lifecycle and history maintenance. |
| **CU-02** | Agentic Tool Calling & Dispatch Loop | `tests/integration/test_cu02_tool_orchestration.py` | Verify model requesting tools -> execution -> sending results back -> final output. |
| **CU-03** | Cross-Session Persistent Memory Sync | `tests/integration/test_cu03_persistent_memory.py` | Verify memory saved in Session A is correctly recalled and ranked in Session B. |
| **CU-04** | Resiliency, Rate Limits & Token Refresh | `tests/integration/test_cu04_resilience_and_recovery.py` | Verify automatic recovery on 429 quota exhaustion and mid-session token refresh. |
| **CU-05** | Security Policy & Destructive Tool Gate | `tests/integration/test_cu05_security_and_permissions.py` | Verify interactive blocking prompt on mutating actions (`write`, `edit`, `exec`). |

---

## 3. Directory Layout & Test Suite Hierarchy

```
tests/
├── conftest.py                          # Global fixtures: temp SQLite memory, mock tokens, event loops
├── helpers/
│   ├── fake_vertex_server.py           # In-memory httpx mock transport with recorded Gemini payloads
│   └── assertions.py                   # Custom domain assertions for tools and memory
├── unit/                               # Fast, isolated, deterministic (<5ms per test, 0 network, 0 disk leaks)
│   ├── test_memory_store.py            # FTS5 search ranking, triggers, schema migrations, edge cases
│   ├── test_token_provider.py          # Non-blocking async token fetch, TTL caching, expiry
│   ├── test_tools_filesystem.py        # read_file, write_file, edit_file with edge cases (unicode, permissions)
│   ├── test_tools_execution.py         # run_command timeouts, truncations, working dir
│   ├── test_vertex_protocol.py         # Gemini JSON schema serialization, SSE event parsing
│   └── test_security_policy.py         # Permission checks, confirmation gates, pattern blocking
└── integration/                         # End-to-End Use Case workflows (Traceability CU-01 to CU-05)
    ├── test_cu01_repl_journey.py
    ├── test_cu02_tool_orchestration.py
    ├── test_cu03_persistent_memory.py
    ├── test_cu04_resilience_and_recovery.py
    └── test_cu05_security_and_permissions.py
```

---

## 4. Concrete Test Specifications

### 4.1. Unit: FTS5 SQLite Ranking & Isolation (`tests/unit/test_memory_store.py`)

```python
import pytest
from pathlib import Path
from cli_agent.memory import MemoryStore


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "memory_test.db")


class TestMemoryStoreUnit:
    def test_fts5_ranked_search_prioritizes_exact_title(self, memory_store: MemoryStore):
        memory_store.save_memory(
            title="Database Architecture",
            content="General discussion about SQL engines",
            memory_type="decision",
        )
        memory_store.save_memory(
            title="Frontend Layout",
            content="Database details mentioned casually in passing",
            memory_type="discovery",
        )

        results = memory_store.search_memory(query="Database")
        assert len(results) >= 1
        assert results[0]["title"] == "Database Architecture"

    def test_special_characters_in_query_do_not_crash_fts5(self, memory_store: MemoryStore):
        """Ensures queries with unmatched quotes, asterisks, or punctuation don't trigger syntax errors."""
        memory_store.save_memory(title="Special Chars", content="Testing 'quotes' and *wildcards*")
        # Should not raise sqlite3.OperationalError
        results = memory_store.search_memory(query="quotes' AND *broken(")
        assert isinstance(results, list)
```

### 4.2. Integration: CU-02 Tool Orchestration (`tests/integration/test_cu02_tool_orchestration.py`)

```python
import pytest
from cli_agent.vertex import VertexAgentClient
from cli_agent.memory import MemoryStore


@pytest.mark.asyncio
class TestCU02ToolOrchestration:
    async def test_agent_completes_read_and_edit_cycle(self, httpx_mock, tmp_path):
        target_file = tmp_path / "app.py"
        target_file.write_text("API_KEY = 'old_secret'\n", encoding="utf-8")

        # Mock Turn 1: Model asks to edit the file
        httpx_mock.add_response(
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "edit_file",
                                        "args": {
                                            "path": str(target_file),
                                            "target": "old_secret",
                                            "replacement": "new_secret",
                                        },
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )

        # Mock Turn 2: Model receives tool result and outputs final message
        httpx_mock.add_response(
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Successfully rotated the secret key."}]}}
                ]
            }
        )

        client = VertexAgentClient(memory_store=MemoryStore(db_path=tmp_path / "mem.db"))
        client._get_token = lambda: "mock-token"

        response = await client.chat(
            [{"role": "user", "parts": [{"text": "rotate API_KEY in app.py"}]}]
        )

        assert response == "Successfully rotated the secret key."
        assert "API_KEY = 'new_secret'" in target_file.read_text(encoding="utf-8")
```

---

## 5. Coverage, Linting & Quality Gates

### 5.1. Strict Pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = """
  --strict-markers
  --strict-config
  --tb=short
  --cov=cli_agent
  --cov-report=term-missing:skip-covered
  --cov-report=html:coverage_html
  --cov-fail-under=85
"""
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "unit: Fast isolated unit tests without network or filesystem side-effects",
    "integration: End-to-end Use Case tests tracing CU specifications",
    "slow: Tests with subprocesses or timing validations",
]

[tool.coverage.run]
branch = true
source = ["src/cli_agent"]
omit = ["src/cli_agent/__main__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### 5.2. Pre-Commit Verification Script (`scripts/verify.sh`)

Mirroring UmlStudio's zero-compromise CI discipline:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Checking Code Formatting with Ruff ==="
uv run ruff format --check .

echo "=== 2. Running Ruff Linter ==="
uv run ruff check .

echo "=== 3. Static Type Analysis with Mypy ==="
uv run mypy src/ tests/

echo "=== 4. Executing Pytest Harness (Unit + Integration with Coverage) ==="
uv run pytest

echo "✅ All Quality Gates Passed (ISO/IEC 25010 Conformance Verified)!"
```
