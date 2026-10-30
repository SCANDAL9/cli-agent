# Agent Capabilities & Action Tools Specification

## 1. Context & Motivation

A true autonomous developer agent cannot merely inspect files; it must modify code, run test suites, execute linters, and inspect git states.

Currently, [`tools.py`](../../src/cli_agent/tools.py) provides only passive read operations:
- `read_file`
- `list_directory`
- `search_code`

This specification defines the addition of active mutation tools alongside a strict Human-in-the-Loop (HITL) safety and permission framework.

---

## 2. Tool Architecture & Security Matrix

```
                    ┌────────────────────────────┐
                    │    Model Tool Invocations  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Security Policy Evaluator │
                    └─────────────┬──────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
        [Passive / Safe]                  [Active / Mutating]
        - read_file                       - write_file
        - list_directory                  - edit_file
        - search_code                     - run_command
        - search_memory                   - git_commit
                │                                   │
                │                                   ▼
                │                         ┌───────────────────┐
                │                         │ Interactive User  │
                │                         │ Permission Check  │
                │                         └─────────┬─────────┘
                │                                   │ Approved
                ▼                                   ▼
        ┌───────────────────────────────────────────────┐
        │            Local Tool Execution Engine        │
        └───────────────────────────────────────────────┘
```

### Classification Matrix

| Tool | Risk Level | Permission Policy |
| :--- | :--- | :--- |
| `read_file`, `list_directory`, `search_code` | Low | Auto-approved (Silent) |
| `search_memory`, `save_memory` | Low | Auto-approved (Silent) |
| `write_file`, `edit_file` | Medium | User approval required if overwriting existing files |
| `run_command` (safe: `git status`, `pytest`, `npm test`) | Medium | Auto-approved or single session grant |
| `run_command` (destructive: `rm -rf`, `git reset --hard`, `drop`) | Critical | Explicit confirmation required with command highlighting |

---

## 3. Tool Specifications & Implementations

### 3.1. `write_file`
Creates or overwrites a file at the given relative or absolute path.

```python
def write_file(path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    """Writes full content to a file, creating parent directories if necessary."""
    try:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists() and not overwrite:
            return {"error": f"File '{path}' already exists. Pass overwrite=True to replace it."}

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "path": str(resolved),
            "bytes_written": len(content.encode("utf-8")),
            "lines": len(content.splitlines()),
        }
    except Exception as e:
        return {"error": f"Failed writing to '{path}': {str(e)}"}
```

### 3.2. `edit_file` (Targeted Substring Replacement)
Allows precision surgical modifications to existing files without rewriting entire multi-thousand line files.

```python
def edit_file(path: str, target: str, replacement: str) -> Dict[str, Any]:
    """Replaces an exact occurrence of target string with replacement."""
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            return {"error": f"Path '{path}' is not an existing file."}

        content = resolved.read_text(encoding="utf-8")
        occurrences = content.count(target)
        if occurrences == 0:
            return {"error": "Target substring was not found in the file."}
        if occurrences > 1:
            return {
                "error": f"Target substring occurs {occurrences} times. Must be uniquely identifiable."
            }

        new_content = content.replace(target, replacement, 1)
        resolved.write_text(new_content, encoding="utf-8")
        return {
            "status": "success",
            "path": str(resolved),
            "replacements_made": 1,
        }
    except Exception as e:
        return {"error": f"Failed editing '{path}': {str(e)}"}
```

### 3.3. `run_command` (Safe Shell Execution)
Executes terminal commands with timeouts, output truncation, and working directory encapsulation.

```python
import subprocess


def run_command(
    command: str, cwd: Optional[str] = None, timeout_seconds: int = 60
) -> Dict[str, Any]:
    """Executes a bash shell command with output capture and timeout protections."""
    try:
        work_dir = Path(cwd or ".").expanduser().resolve()
        proc = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout[-15000:] if len(proc.stdout) > 15000 else proc.stdout
        stderr = proc.stderr[-15000:] if len(proc.stderr) > 15000 else proc.stderr

        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": len(proc.stdout) > 15000 or len(proc.stderr) > 15000,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout_seconds} seconds."}
    except Exception as e:
        return {"error": f"Command execution failed: {str(e)}"}
```

---

## 4. Human-in-the-Loop Confirmation Gate

Before dispatching `write_file`, `edit_file`, or `run_command`, the orchestrator evaluates:

1. **Auto-approve mode:** Enabled only if `--auto-approve` flag is passed at CLI launch.
2. **Interactive prompt:**
   ```
   [bold red]Action Required:[/bold red] The agent requests to execute:
   ⚙ [bold yellow]run_command[/bold yellow]: pytest tests/unit/
   Allow execution? [y/N/always]:
   ```
3. **Session Allowlist:** If the user selects `always`, that specific tool pattern remains allowed for the rest of the active REPL session.
