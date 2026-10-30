# CLI User Experience & Developer Experience (DX) Specification

## 1. Problem Statement & UX Deficiencies

The current REPL implementation in [`main.py`](../../src/cli_agent/main.py#L65):
- Relies on basic `console.input()`, which lacks:
  - Persistent shell history across terminal restarts (Up/Down arrow recalls).
  - Multiline prompt submission (pasting a stack trace or code snippet triggers immediate premature submission).
  - Autocompletion for slash commands (`/memories`, `/clear`, `/exit`).
  - Vi / Emacs editing keybindings.
- Emits output via a static spinner that blanks out while the model is thinking, preventing perceived real-time interaction.
- Does not support standard flags like `--version`, `--model`, `--auto-approve`, or `--project`.

---

## 2. Upgraded Architecture with `prompt_toolkit`

```
┌────────────────────────────────────────────────────────┐
│                   Terminal / PTY                       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               prompt_toolkit PromptSession             │
│  - Persistent History (~/.config/cli-agent/history)    │
│  - Slash-command Completer (/exit, /clear, /memories)  │
│  - Multiline Input Mode (Alt+Enter for newline)        │
│  - Syntax Highlighting for code input                  │
└───────────────────────────┬────────────────────────────┘
                            │ Prompt String
                            ▼
┌────────────────────────────────────────────────────────┐
│               Rich Live Streaming Display              │
│  - Real-time token Markdown rendering (Live context)   │
│  - Tool execution badges with elapsed timings          │
│  - Syntax-highlighted code diff panels                 │
└────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Details

### 3.1. `prompt_toolkit` Session Setup

Add `prompt_toolkit` to dependencies in `pyproject.toml`.

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style

SLASH_COMMANDS = ["/memories", "/clear", "/exit", "/help", "/compact", "/model", "/doctor"]
command_completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)

custom_style = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "continuation": "ansibrightblack",
    }
)

session: PromptSession = PromptSession(
    history=FileHistory(Path.home() / ".config" / "cli-agent" / "history"),
    completer=command_completer,
    style=custom_style,
    multiline=False,  # Toggleable via keybinding or flag
)
```

### 3.2. Real-Time Streaming Output with Rich `Live`

Using `rich.live.Live` to render markdown as tokens stream in from Vertex AI:

```python
from rich.live import Live
from rich.markdown import Markdown


async def stream_render_response(token_generator):
    full_text = ""
    with Live(Markdown(""), refresh_per_second=12, console=console) as live:
        async for token in token_generator:
            full_text += token
            live.update(Markdown(full_text))
    return full_text
```

### 3.3. Slash Command Ecosystem

Expand available interactive slash commands:

| Command | Action |
| :--- | :--- |
| `/memories` | Renders a Rich table of SQLite persistent memories for current project. |
| `/clear` | Flushes active conversation history while preserving SQLite memory. |
| `/compact` | Triggers immediate context window summarization and pruning. |
| `/doctor` | Verifies gcloud authentication, API endpoint connectivity, and database health. |
| `/model <name>` | Hot-swaps the underlying model (e.g. `gemini-3.8-flash` to `gemini-3.1-pro`). |
| `/exit` or `/quit` | Cleanly exits with automatic session memory summarization. |

### 3.4. Unified CLI Configuration (`config.toml`)

Allow users to define defaults in `~/.config/cli-agent/config.toml`:

```toml
[defaults]
model = "gemini-3.8-flash"
location = "global"
project = "my-gcp-project"
temperature = 0.2
auto_approve_readonly = true
max_tool_iterations = 10

[theme]
prompt_color = "cyan"
agent_color = "magenta"
border_style = "rounded"
```
