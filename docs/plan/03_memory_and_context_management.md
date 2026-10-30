# Memory & Context Window Management Specification

## 1. Baseline Review & Problem Statement

In [`memory.py`](../../src/cli_agent/memory.py):
1. **Global Unscoped Storage:** All memories are dumped into `~/.config/cli-agent/memory.db`. If a developer works on `project-a` (Go backend) and `project-b` (React frontend), discoveries and architecture decisions bleed across unrelated workspaces.
2. **Unbounded Context Window in REPL:** In [`main.py`](../../src/cli_agent/main.py#L51-L104), `conversation_history` grows monotonically across turns without token count checks. After 15–20 complex turns with tool outputs, payload sizes inflate, latency increases, and costs grow unnecessarily.
3. **Absence of Session Lifecycle Hook:** On `/exit` or terminal closure, the session evaporates. Discoveries made during the session that were not explicitly saved via `save_memory` are permanently lost.

---

## 2. Proposed Database Schema Evolution

### Schema Migration v2

```sql
-- Upgraded memories table with workspace tracking and soft tags
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,       -- Normalized git root basename or absolute path hash
    scope TEXT DEFAULT 'project',    -- 'project' or 'global'
    type TEXT NOT NULL,              -- 'decision' | 'bugfix' | 'discovery' | 'convention' | 'config'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,              -- JSON containing commit hash, active branch, author
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_key);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);

-- FTS5 full text search table including project_key filter
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title,
    content,
    project_key UNINDEXED,
    content='memories',
    content_rowid='id'
);
```

---

## 3. Project Scoping Logic

To automatically identify the active project without manual developer configuration:

```python
import subprocess
from pathlib import Path


def resolve_project_key(cwd: Optional[Path] = None) -> str:
    """Resolves canonical project identifier via git root or directory path."""
    target_dir = cwd or Path.cwd()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        git_root = Path(res.stdout.strip())
        return git_root.name
    except Exception:
        return target_dir.name
```

### Contextual Recall Protocol:
- When querying `search_memory`:
  - Search filter: `project_key = :current_project OR scope = 'global'`
- When priming the system instruction:
  - Load top 5 recent memories for the *current project*.
  - Load top 3 global developer preferences.

---

## 4. Conversation History Compaction & Sliding Window

To prevent context window explosion while preserving critical decisions:

```
[System Instruction & Primed Memories] (Static Header)
               │
               ▼
[Compacted Summary of Turns 1..N-6] (Generated via LLM if turns > 12)
               │
               ▼
[Active Recent Turns N-5..N] (Raw Full Fidelity Messages & Tool Responses)
```

### Algorithm:
1. Track approximate tokens per message (e.g. 1 token ~= 4 chars).
2. If total session tokens exceed `MAX_CONTEXT_TOKENS` (e.g., 64,000) or turn count exceeds `12`:
   - Take the oldest `6` turns.
   - Run a background summarization prompt: `"Summarize key facts, files edited, and decisions from these turns in 3 bullet points."`
   - Replace those turns with a single synthetic `{"role": "user", "parts": [{"text": "[Previous context summary: ...]"}]}`.
   - Keep the most recent `6` turns intact.

---

## 5. Automatic Session Summarization on Exit

When `/exit` or `SIGINT` is received, prompt or auto-execute a session summarizer:

```python
async def finalize_session(history: list[dict], memory_store: MemoryStore, project_key: str):
    """Inspects conversation history and persists high-level accomplishments if meaningful work occurred."""
    if len(history) < 4:
        return

    # Trigger lightweight extraction turn
    summary = await generate_session_summary(history)
    if summary and summary.get("accomplishments"):
        memory_store.save_memory(
            project_key=project_key,
            title=f"Session Summary: {summary['title']}",
            content=summary["body"],
            memory_type="session_summary",
        )
```
