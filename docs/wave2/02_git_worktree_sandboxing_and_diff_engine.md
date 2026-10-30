# Specification: Git Worktree Sandboxing & Atomic Diff Engine

## 1. Problem Statement: The Danger of In-Place Mutations

In Wave 1, `write_file` and `edit_file` modify files directly in the user's working directory:
- **Dirty Tree Collision:** If the user has uncommitted WIP changes, agent edits can cause merge conflicts or corrupt uncommitted work.
- **Half-Baked Breakage:** If an agent gets stuck halfway through a 5-file refactor, the user is left with a broken build state and failing tests.
- **Destructive Shell Commands:** Running `run_command` directly in the active workspace risks accidental deletion of critical artifacts.

---

## 2. Sandboxing Architecture: Ephemeral Git Worktrees

Instead of touching the user's active directory, `cli-agent` spawns an **ephemeral Git worktree** in an isolated runtime path:

```
                          Developer's Active Workspace
                          (/workspace/projects/my-app)
                                       │
                         [Initiate Complex Task / Refactor]
                                       │
                                       ▼
                       Ephemeral Git Worktree Branch
                (/tmp/cli-agent/worktrees/agent-session-42)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
           Agent Coding Cycle                    Automated Test Suite
           - write_file (sandbox)                - uv run pytest
           - edit_file  (sandbox)                - npm test
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                             All Tests Pass (Green)
                                       │
                                       ▼
                         Interactive Patch Review
                    ┌─────────────────────────────────────┐
                    │  Unified Color Diff Review          │
                    │  [a] Accept & Merge to Main Tree    │
                    │  [d] Discard Sandbox Changes        │
                    └─────────────────────────────────────┘
```

---

## 3. Worktree Lifecycle Manager Protocol

### 3.1. Worktree Creation
```bash
# Branch name generated deterministically
BRANCH="agent/task-$(date +%s)"
WORKTREE_DIR="/tmp/cli-agent/worktrees/$(uuidgen)"

git worktree add -b "${BRANCH}" "${WORKTREE_DIR}" HEAD
```

### 3.2. Execution Guarantees
- The agent's working directory is pinned to `WORKTREE_DIR`.
- The user's IDE and active files remain completely untouched during execution.
- If the agent crashes or enters an infinite loop, the user's repository state is 100% pristine.

### 3.3. Atomic Merge vs Teardown
```bash
# On User Approval: Fast-forward or merge back to working directory
git merge --ff-only "${BRANCH}"

# Teardown:
git worktree remove --force "${WORKTREE_DIR}"
git branch -D "${BRANCH}"
```

---

## 4. Rich Unified Diff Review UI

Before applying any merged changes:
- Renders GitHub-style side-by-side or syntax-highlighted unified diffs using Rich.
- Displays test execution summary (e.g. `24 passed in 1.1s in sandbox`).
- Asks developer: `Apply these changes to your active workspace? [y/N/inspect]`.
