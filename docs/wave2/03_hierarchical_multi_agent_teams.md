# Specification: Hierarchical Multi-Agent Teams (Swarm)

## 1. Architectural Philosophy: Decomposing Cognitive Load

Complex features cannot be reliably implemented by a single agent loop:
- **Context Contamination:** Output from linting, build failures, and intermediate files fills the LLM context window.
- **Role Confusion:** An agent that plans, codes, and verifies simultaneously tends to rubber-stamp its own bugs.

Wave 2 introduces **Hierarchical Multi-Agent Teams** structured according to Organic Driven Development (ODD):

```
                        ┌─────────────────────────────────────────┐
                        │      Lead Architect / Planner Agent     │
                        │    - Requirements & Architecture Plan   │
                        │    - Task Graph & Dependency Ordering   │
                        └───────────────────┬─────────────────────┘
                                            │ Dispatches Subtasks
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Coding Worker A (Core)  │                 │   Coding Worker B (API)   │
       │  - Scoped to module X     │                 │  - Scoped to module Y     │
       │  - Clean 4k token context │                 │  - Clean 4k token context │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │ Submits Code Diffs
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │       Independent QA / Reviewer Agent   │
                        │  - Zero code mutation tools             │
                        │  - Runs test suites & linters           │
                        │  - Rejects if regressions detected      │
                        └─────────────────────────────────────────┘
```

---

## 2. Agent Roles & Privilege Matrix

| Agent Role | Tools Permitted | Objective |
| :--- | :--- | :--- |
| **Lead Architect** | `read_file`, `list_directory`, `search_code`, `delegate_task` | Break user request into discrete task units; synthesize final report. |
| **Feature Worker** | `read_file`, `write_file`, `edit_file`, `search_code` | Implement assigned task unit in its assigned isolated scope. |
| **QA / Verifier** | `read_file`, `run_command` (tests/linters only) | Independently verify correctness; produce PASS/FAIL report with logs. |

---

## 3. Delegation Protocol & Inter-Agent Messaging

### Subagent Spawning Interface:
```python
async def spawn_worker(
    task_id: str,
    role: str,
    objective: str,
    allowed_files: list[str],
    context_files: list[str],
) -> WorkerResult:
    """Spawns an isolated subagent worker with a bounded context and approved surfaces."""
```

### Safety & Recursion Guards:
- **Max Delegation Depth:** Strictly capped at `3` levels to prevent runaway agent spawning.
- **Budget / Turn Limits:** Subagent workers have a maximum of `8` turns before forced reporting.
- **Context Isolation:** Subagents do NOT inherit parent conversation history; they receive only the prompt, relevant symbol signatures, and target files.
