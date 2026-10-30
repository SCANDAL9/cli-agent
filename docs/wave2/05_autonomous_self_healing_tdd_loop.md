# Specification: Autonomous Self-Healing TDD Repair Loop

## 1. Context: Moving Beyond Trial-and-Error Coding

Typical LLM coding assistants guess solutions and ask the user to test them.
Wave 2 introduces the **Autonomous TDD Self-Healing Engine**, adhering to strict Test-Driven Development (TDD) principles:

```
[User Bug Report / Feature Request]
                │
                ▼
      ┌──────────────────┐
      │ Phase 1: RED     │──► Synthesize reproducing unit test in sandbox
      └─────────┬────────┘    Run pytest -> Must FAIL (Red confirmed)
                │
                ▼
      ┌──────────────────┐
      │ Phase 2: PATCH   │──► Formulate hypothesis & apply surgical code edit
      └─────────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │ Phase 3: GREEN   │──► Re-run reproducing test
      └─────────┬────────┘    - If fails: Loop back to Phase 2 (max 5 iterations)
                │             - If passes: Proceed to Phase 4
                ▼
      ┌──────────────────┐
      │ Phase 4: NO REG  │──► Run entire existing repository test suite
      └─────────┬────────┘    - If regression detected: Roll back and adjust patch
                │             - If 100% green: SUCCESS
                ▼
  [Produce Verified Atomic Diff]
```

---

## 2. Iteration Loop Protocol & Safety Gates

1. **Max Repair Iterations:** Strictly capped at `5` cycles. If a fix cannot be verified within 5 cycles, the agent halts, rolls back the sandbox, and generates a root-cause explanation to the user.
2. **Reproduction Validation:** The agent is **forbidden** from writing production code until it has confirmed that a new test successfully reproduces the failure in the sandbox.
3. **Regression Budget:** If a proposed patch fixes the bug but breaks 2 existing tests, the patch is rejected immediately.
