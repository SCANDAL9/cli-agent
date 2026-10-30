# cli-agent

An autonomous AI developer agent running directly on developer workstations via Google Cloud Vertex AI (Gemini 3.8 Flash). Features non-blocking async execution, local filesystem inspection and editing, sandboxed shell execution, SQLite memory with FTS5, and an interactive terminal REPL powered by `prompt_toolkit` and `Rich`.

---

## Features

- **Interactive REPL & One-Shot Mode:** Seamless terminal UX with `prompt_toolkit`, persistent command history, and tab autocompletion.
- **Real-Time Streaming:** Server-Sent Events (SSE) streaming for instant token-by-token response rendering.
- **Surgical Code & System Tools:**
  - `read_file`: Inspect lines with line-numbered ranges.
  - `write_file`: Create new files with parent directory creation.
  - `edit_file`: Atomic, targeted string replacements without full-file rewrites.
  - `run_command`: Sandboxed shell execution with timeout protection and output truncation.
  - `search_code`: Fast regex/pattern search across source trees.
- **Security & Safety Gate (HITL):** Human-in-the-loop permission prompt `[y/N/always]` for mutating actions and pattern detection for dangerous shell commands.
- **Project-Scoped Persistent Memory:** SQLite FTS5 full-text search engine automatically scoped by git repository or workspace directory.
- **ISO/IEC 25010 Quality Standard:** Modeled after enterprise verification standards with Use Case (CU) test traceability and strict quality gates.

---

## Quickstart

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Authenticated Google Cloud SDK (`gcloud auth login`)

### Installation & Execution

```bash
# Sync dependencies
uv sync

# Run interactive agent REPL
uv run cli-agent

# Run one-shot prompt
uv run cli-agent "Inspect pyproject.toml and summarize dependencies"

# Run with auto-approved permissions
uv run cli-agent --auto-approve
```

---

## Interactive Slash Commands

| Command | Action |
| :--- | :--- |
| `/memories` | Display persistent SQLite memories for the current workspace. |
| `/compact` | Compact conversation history into a synthetic summary block. |
| `/doctor` | Run diagnostic health checks on auth tokens, database, and workspace. |
| `/model <name>` | Inspect or dynamically switch the active model (e.g. `gemini-3.1-pro`). |
| `/clear` | Flush current chat context while keeping persistent memories. |
| `/help` | Print list of available interactive commands. |
| `/exit` | Quit the agent and persist high-level session accomplishments. |

---

## Configuration

Defaults can be defined in `~/.config/cli-agent/config.toml`:

```toml
[defaults]
project = "my-gcp-project"
location = "global"
model = "gemini-3.8-flash"
temperature = 0.2
auto_approve = false
```

Environment variable overrides:
- `GCP_PROJECT`: Google Cloud Project ID.
- `GEMINI_LOCATION`: Vertex AI region endpoint (default: `global`).
- `GEMINI_MODEL`: Model name (default: `gemini-3.8-flash`).
- `GCLOUD_ACCESS_TOKEN`: Custom OAuth2 token (overrides gcloud subprocess).

---

## Quality & Verification

Run the full verification suite (Ruff format, Ruff linter, Mypy strict, and Pytest with >=85% coverage gate):

```bash
./scripts/verify.sh
```
