# cli-agent

An autonomous AI developer agent running directly on developer workstations via Google Cloud Vertex AI (Gemini 3.8 Flash, Gemini 3.1 Pro, Claude, etc.). Features non-blocking async execution, surgical code editing, AST structural intelligence, sandboxed shell execution, ephemeral Git worktrees, SQLite memory with FTS5, and an interactive terminal REPL powered by `prompt_toolkit` and `Rich`.

---

## Table of Contents

- [Features](#features)
- [Quickstart & CLI Commands](#quickstart--cli-commands)
  - [Prerequisites](#prerequisites)
  - [Session Management Options](#session-management-options)
  - [Execution & Mode Options](#execution--mode-options)
  - [One-Shot and Pipe Mode](#one-shot-and-pipe-mode)
- [Interactive REPL & Slash Commands](#interactive-repl--slash-commands)
  - [Autonomous Permissions (`/auto`)](#autonomous-permissions-auto)
  - [Session Continuation & History](#session-continuation--history)
  - [Code Intelligence & Search (`/index`, `/search`)](#code-intelligence--search-index-search)
  - [Git Sandbox Worktree (`/sandbox`)](#git-sandbox-worktree-sandbox)
  - [Multi-Agent Swarms & TDD Engine (`/team`, `/tdd`)](#multi-agent-swarms--tdd-engine-team-tdd)
  - [Pull Request & CI Automation (`/pr`)](#pull-request--ci-automation-pr)
  - [Model Garden & Diagnostics (`/model`, `/doctor`, `/setup`)](#model-garden--diagnostics-model-doctor-setup)
- [Safety & Human-in-the-Loop (HITL) Gate](#safety--human-in-the-loop-hitl-gate)
- [Surgical Code & System Tools](#surgical-code--system-tools)
- [Configuration](#configuration)
- [Quality & Verification](#quality--verification)

---

## Features

- **Interactive REPL & One-Shot Mode:** Terminal UX with `prompt_toolkit`, persistent command history, and dynamic tab autocompletion.
- **Dynamic Session Resumption:** Seamlessly pause, list, switch, or resume previous conversations across terminal sessions.
- **Hot-Togglable Autonomous Mode:** Toggle permission gates on the fly with `/auto` or select `[y] Yes / [n] No / [a] Always` per tool call.
- **Surgical File Operations:** In-place atomic string replacements (`edit_file`) without rewriting entire files.
- **Tree-sitter Code Intelligence:** Structural AST indexing, outline extraction, symbol definition lookup, references tracing, and impact analysis.
- **Hybrid Code Search:** Combines BM25 keyword matching (FTS5) with dense vector embeddings via Reciprocal Rank Fusion (RRF).
- **Ephemeral Git Sandboxes:** Isolate experimental code mutations in throwaway Git worktrees with atomic diff reviews before merging.
- **Autonomous Multi-Agent Swarm (ODD):** Decompose complex goals into isolated Feature Worker and QA Verifier subagents.
- **Self-Healing TDD Loop:** Automatic Red-Green-Refactor test repair loop with regression detection.
- **PR & CI Synthesizer:** Automatic Conventional Commit and ISO/IEC 25010 compliant Pull/Merge Request generation and CI watch/healing.
- **Persistent SQLite Memory:** Project-scoped and global full-text search (FTS5) for past architectural decisions, discoveries, and conventions.

---

## Quickstart & CLI Commands

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Authenticated Google Cloud SDK (`gcloud auth login`)

### Installation & Basic Launch

```bash
# Sync dependencies
uv sync

# Launch standard interactive REPL
uv run cli-agent
```

### Session Management Options

Resume conversations and manage persistent workspace sessions directly from the terminal:

| Flag / Option | Description |
| :--- | :--- |
| `-c`, `--continue` | Resume the most recent conversation session in the current workspace. |
| `--resume [SESSION_ID]` | Resume a specific conversation session by ID (or latest if ID is omitted). |
| `--session [SESSION_ID]` | Alias for `--resume`. |
| `--sessions`, `--list-sessions` | Display all saved sessions for the current workspace and exit. |

```bash
# Resume the latest session in current repository
uv run cli-agent --continue

# Resume a specific session by ID
uv run cli-agent --resume 20250228-143022-abcd

# List all recorded sessions for this workspace
uv run cli-agent --sessions
```

### Execution & Mode Options

| Flag / Option | Description |
| :--- | :--- |
| `--auto-approve` | Automatically approve safe and medium-risk actions (file edits, tests, commands) without interactive prompts. |
| `--model <MODEL_NAME>` | Override the default model (e.g. `gemini-3.8-flash`, `gemini-3.1-pro`, `claude-3-7-sonnet`). |
| `--max-turns <N>` | Set a hard limit on conversation loop turns. |
| `--tui` | Launch the fullscreen 4-zone Split-Pane Terminal User Interface. |
| `--voice` | Start a bidirectional Gemini Live voice streaming session. |
| `-h`, `--help` | Show CLI help overview and exit. |

```bash
# Run with pre-approved permissions for autonomous code editing
uv run cli-agent --auto-approve

# Run using a specific model
uv run cli-agent --model gemini-3.1-pro
```

### One-Shot and Pipe Mode

You can run single commands without entering the interactive REPL:

```bash
# Direct command-line argument
uv run cli-agent "Inspect pyproject.toml and summarize dependencies"

# Auto-approved one-shot refactoring
uv run cli-agent --auto-approve "Add docstrings to all functions in src/cli_agent/config.py"

# Standard input pipe
cat error.log | uv run cli-agent "Explain the root cause of this stack trace"
```

---

## Interactive REPL & Slash Commands

Inside the interactive chat, type `/` to access rich slash commands with `<TAB>` autocompletion:

### Autonomous Permissions (`/auto`)

Control the security policy in real-time without restarting your chat:

| Command | Action |
| :--- | :--- |
| `/auto on` | **Enable autonomous mode:** Pre-approves safe and medium-risk actions (`edit_file`, `write_file`, test execution). High-risk commands (`rm -rf`, destructive git) will still prompt. |
| `/auto off` | **Disable autonomous mode:** Prompts for confirmation on every mutating action. |
| `/auto` | Toggle between autonomous and interactive approval mode. |

### Session Continuation & History

| Command | Action |
| :--- | :--- |
| `/sessions` | List all saved conversation sessions in the current workspace. |
| `/session` | Show active session ID, turn count, workspace key, and active model. |
| `/resume <session_id>` | Switch to or resume another saved conversation session. |
| `/clear` | Clear the active conversation history and start a fresh session ID (preserves SQLite memories). |
| `/compact` | Manually compress conversation history into an intelligent summary block to optimize context tokens. |
| `/exit` | Exit the REPL and persist session summary. |

### Code Intelligence & Search (`/index`, `/search`)

| Command | Action |
| :--- | :--- |
| `/index` | Parse and index all workspace AST symbols, references, and code vector chunks. |
| `/search <query>` | Hybrid code search combining BM25 keyword matching with dense vector embeddings via Reciprocal Rank Fusion (RRF). |

### Git Sandbox Worktree (`/sandbox`)

Isolate modifications in an ephemeral Git worktree so your active working directory remains 100% clean until you approve the diff:

| Command | Action |
| :--- | :--- |
| `/sandbox on` | Create and activate an isolated Git worktree branch (`cli-agent-sandbox-*`). |
| `/sandbox status` | Show current sandbox branch, directory, and modification summary. |
| `/sandbox diff` | Render colorized syntax-highlighted git diff of all modifications made in the sandbox. |
| `/sandbox apply` | Review change summary and merge the sandbox patch into your main workspace. |
| `/sandbox discard` | Discard all sandbox modifications cleanly and restore main workspace. |

### Multi-Agent Swarms & TDD Engine (`/team`, `/tdd`)

| Command | Action |
| :--- | :--- |
| `/team <goal>` | Launch an autonomous multi-agent swarm following the ODD pipeline: **Architect** (planning) -> **Feature Worker** (surgical edits) -> **QA Verifier** (test validation). |
| `/tdd <test_target>` | Trigger an autonomous Red-Green-Refactor repair loop on a failing test (e.g. `/tdd tests/unit/test_config.py`). Reproduces failure, applies minimal fix, confirms green, and checks for regressions. |

### Pull Request & CI Automation (`/pr`)

| Command | Action |
| :--- | :--- |
| `/pr preview` | Synthesize Conventional Commit messages and ISO/IEC 25010 compliant PR markdown body based on current git diff. |
| `/pr create` | Generate PR and publish/update it directly on GitHub (`gh`) or GitLab (`glab`). |
| `/pr watch` | Monitor remote CI pipeline status and trigger self-healing loops if checks fail. |

### Model Garden & Diagnostics (`/model`, `/doctor`, `/setup`)

| Command | Action |
| :--- | :--- |
| `/model` | Open interactive table of available models from Google Vertex AI Model Garden. |
| `/model <name>` | Dynamically switch the active model (e.g. `/model gemini-3.1-pro` or index number). |
| `/model refresh` | Force-refresh the remote Model Garden catalog. |
| `/doctor` | Run deep diagnostic health checks on auth tokens, SQLite database, AST Code Graph, sandbox engine, and workspace. |
| `/doctor --fix` | Automatically attempt repair of missing Google Cloud APIs or permissions. |
| `/setup` | Guided Google Cloud setup wizard: switch projects and enable Vertex AI API with one click. |
| `/memories` | Display all persistent project and global SQLite memories. |
| `/mcp [list\|reload]` | Inspect active Model Context Protocol (MCP) federated servers and tools, or reload configurations. |
| `/tui` | Switch to the 4-zone fullscreen Terminal UI. |
| `/voice` | Start live audio voice session with Gemini Live. |
| `/help` | Print the command reference. |

---

## Safety & Human-in-the-Loop (HITL) Gate

`cli-agent` enforces a strict risk-tiered security policy:

```
⚠️  Permission Request (MEDIUM)
Action: edit_file(path='src/cli_agent/vertex.py', ...)
  [y] Yes   [n] No   [a] Always (this session)
Select an option [y/n/a] (default 'n'):
```

- **Safe (Read-Only):** `read_file`, `list_directory`, `search_code`, AST symbol inspection, and memory searches are executed immediately.
- **Medium (Mutations):** `edit_file`, `write_file`, and standard terminal commands. Prompts the user with `[y/n/a]`:
  - `y` (Yes): Approves this specific execution.
  - `n` (No / Enter): Denies execution (safe default).
  - `a` (Always): Grants permanent approval for this tool and target file/command for the remainder of the session.
- **High (Destructive):** Dangerous commands (`rm -rf`, `git reset --hard`, disk wiping, fork bombs) **always require explicit confirmation**, even when `--auto-approve` or `/auto on` is active.

---

## Surgical Code & System Tools

The agent interacts with your workstation via atomic tools:

### File Editing & Exploration
- **`read_file`**: Read specific line ranges (`start_line`, `end_line`) with line numbers without saturating context.
- **`write_file`**: Create new files or overwrite existing files, creating parent directories automatically.
- **`edit_file`**: Atomic, exact string replacement (`target` -> `replacement`). Fails cleanly if the target string is non-unique or absent, preventing corrupted edits.
- **`list_directory`**: Explore directory trees with depth limits, ignoring `.git`, `node_modules`, and `.venv`.
- **`search_code`**: Fast regex and pattern search across files in the workspace.

### System & Shell
- **`run_command`**: Execute bash commands with timeout enforcement, working directory control, and output truncation.

### Structural Code Intelligence (Tree-sitter AST)
- **`get_symbol_outline`**: Extract skeleton outlines (classes, functions, methods, docstrings) without reading entire bodies (~90% token reduction).
- **`find_symbol_definition`**: Pinpoint exact file path and line numbers for any symbol in the AST index.
- **`find_symbol_references`**: Trace all callers, imports, and usages across the workspace.
- **`analyze_symbol_impact`**: Comprehensive impact analysis reporting incoming callers, dependent files, and risk level (`LOW`, `MEDIUM`, `HIGH`).

### Persistent Memory (SQLite FTS5)
- **`save_memory`**: Persist decisions, discoveries, conventions, or bug fixes across sessions.
- **`search_memory`**: Query persistent memories using full-text search.

### Declarative Workflows

Workflows can be defined as TOML or YAML files in `.cli-agent/workflows/`; user-level
recipes live in `~/.config/cli-agent/workflows/`. Workspace definitions override
user recipes, which override the four bundled recipes. Use:

```text
/workflow list
/workflow describe strict-tdd
/workflow run strict-tdd feature_description="Add retry handling" target_module=src/cli_agent/client.py test_name=test_retry_limit
/workflow abort
```

`/workflow list` shows each recipe's description and inputs. `/workflow describe <id>`
shows step dependencies and verification gates. Bundled recipes cover strict TDD
(including a Python AST parse gate), code health (secret scan, formatting/linting,
type checking, and bounded LSP diagnostics), security review, and dependency auditing
(outdated/unused packages plus known vulnerability advisories). The dependency audit
uses [`uvx` and `uv run --with`](https://docs.astral.sh/uv/guides/tools/) to obtain its
audit tools, including [pip-audit](https://github.com/pypa/pip-audit), and may contact
configured package and advisory services; the normal workflow permission prompt must
approve each command.

Each run starts from an isolated Git worktree that includes tracked edits and
visible untracked files from the current workspace. Successful runs leave their
branch, checkpoint tag, and worktree for inspection; failures and aborts remove
only that run's checkpoint, leaving the original workspace untouched. Shell and verification
commands pass through the existing permission policy. Runs and step outcomes are
recorded in the configured storage backend.

---

## Configuration

Default settings can be customized in `~/.config/cli-agent/config.toml`:

```toml
[defaults]
project = "my-gcp-project"
location = "global"
model = "gemini-3.8-flash"
temperature = 0.2
auto_approve = false
timeout = 300.0
```

### Environment Variable Overrides

- `GCP_PROJECT`: Google Cloud Project ID.
- `GEMINI_LOCATION`: Vertex AI endpoint location (default: `global`).
- `GEMINI_MODEL`: Model name (default: `gemini-3.8-flash`).
- `GCLOUD_ACCESS_TOKEN`: Custom OAuth2 access token.
- `CLI_AGENT_TIMEOUT`: HTTP client timeout in seconds (default: `300.0`).

### Storage Backends

SQLite remains the zero-configuration default. To use Turso, install the optional
driver and provide both the database URL and token:

```bash
pip install 'cli-agent[turso]'
export CLI_AGENT_STORAGE_BACKEND=turso
export TURSO_DATABASE_URL='libsql://your-database.turso.io'
export TURSO_AUTH_TOKEN='your-database-token'
```

The default `TURSO_MODE=replica` keeps a local embedded replica and synchronizes
it with Turso. Set `TURSO_MODE=remote` to connect directly without a local
replica; `TURSO_SYNC_INTERVAL` controls replica sync frequency (default: 30
seconds), and `TURSO_REPLICA_PATH` optionally selects the replica file. The
authentication token is read only from `TURSO_AUTH_TOKEN`, never from the TOML
configuration. On first Turso use, the legacy `~/.config/cli-agent/memory.db`
is migrated transactionally; the source database is left unchanged.

---

## Quality & Verification

`cli-agent` conforms to ISO/IEC 25010 enterprise verification standards with strict quality gates:

```bash
# Run full verification suite (Ruff format, Ruff linter, Mypy strict, and Pytest >=85% coverage)
./scripts/verify.sh

# Run only test suite
uv run pytest
```
