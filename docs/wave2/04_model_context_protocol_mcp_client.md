# Specification: Model Context Protocol (MCP) Client

## 1. Context & Motivation

Hardcoding every tool into [`tools.py`](../../src/cli_agent/tools.py) does not scale:
- Developers need database querying (PostgreSQL, SQLite, BigQuery), issue tracker integration (GitHub, Jira), container management (Docker, Kubernetes), and web search (Brave, Google).
- The industry has converged on **Model Context Protocol (MCP)**, an open standard by Anthropic adopted universally across AI developer tooling.

Wave 2 turns `cli-agent` into a **Universal MCP Client**.

---

## 2. MCP Client Architecture

```
                    ┌─────────────────────────────────────────┐
                    │               cli-agent                 │
                    │        (Universal MCP Client)           │
                    └───────────────────┬─────────────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │ JSON-RPC 2.0 (stdio)     │ JSON-RPC 2.0 (stdio)     │ SSE (HTTP)
             ▼                          ▼                          ▼
┌─────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
│   GitHub MCP Server     ││  Postgres MCP Server    ││  Remote Team MCP Server │
│ - list_pull_requests    ││ - query_database        ││ - internal_knowledge    │
│ - create_issue          ││ - describe_table        ││ - deployment_status     │
└─────────────────────────┘└─────────────────────────┘└─────────────────────────┘
```

---

## 3. Configuration & Server Registry (`~/.config/cli-agent/mcp.json`)

Developers register MCP servers via standard JSON configuration:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "command": "uvx",
      "args": ["mcp-server-postgres", "postgresql://localhost/mydb"]
    },
    "docker": {
      "command": "uvx",
      "args": ["mcp-server-docker"]
    }
  }
}
```

---

## 4. Automatic Protocol Translation to Gemini 3.8

When `cli-agent` starts up:
1. **Handshake & Discovery:** Sends `initialize` and `tools/list` over JSON-RPC 2.0.
2. **Schema Translation:** Automatically converts MCP JSON Schema definitions into `GEMINI_TOOLS_DECLARATIONS` format.
3. **Execution Dispatch:** When Gemini yields `functionCall` with prefix `mcp__github__create_issue`:
   - Routes request to the active GitHub stdio process.
   - Awaits `tools/call` response and returns result to model.
