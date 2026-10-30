# Wave 3: Enterprise Pre-Flight, IAM & Guided Onboarding Engine
## 00. Executive Architecture & System Topology

### 1. Executive Summary

Enterprise developer CLI tools must not fail with cryptic error messages such as `403 PERMISSION_DENIED`, `SERVICE_DISABLED`, or `PROJECT_NOT_FOUND`. A first-time developer cloning `cli-agent` or running it in a new environment should experience zero friction.

Wave 3 establishes the **Enterprise Pre-Flight & Guided Onboarding Engine**: an automated, interactive bootstrap pipeline that inspects, prompts, auto-provisions, and diagnoses the local environment before entering the agent event loop.

```
+-----------------------------------------------------------------------------------+
|                           Pre-Flight Inspection Pipeline                          |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ 1. GCloud Authentication & Session Guard     │
                 │    - Active account check                     │
                 │    - Interactive login trigger               │
                 └──────────────────────┬───────────────────────┘
                                         │ Pass
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ 2. Project Discovery & Interactive Picker    │
                 │    - Scan accessible GCP projects            │
                 │    - Interactive selection & config save     │
                 └──────────────────────┬───────────────────────┘
                                         │ Pass
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ 3. Vertex AI Service API Enablement          │
                 │    - Check `aiplatform.googleapis.com`       │
                 │    - One-click auto-enablement gate          │
                 └──────────────────────┬───────────────────────┘
                                         │ Pass
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ 4. IAM Role & Permission Matrix Validator    │
                 │    - Validate `roles/aiplatform.user`        │
                 │    - Pre-flight prediction endpoint probe    │
                 └──────────────────────┬───────────────────────┘
                                         │ Pass
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │ 5. Billing Link & Quota Health Guard         │
                 │    - Billing account association check       │
                 │    - Quota exhaustion & 429 pre-warning      │
                 └──────────────────────┬───────────────────────┘
                                         │ Pass
                                         ▼
               [Enter Agent REPL / Execute Requested Operation]
```

---

### 2. Design Principles

1. **Self-Healing & Actionable Remediation**: Every check failure must provide either an automated one-click fix or the exact `gcloud` copy-paste command required to resolve it.
2. **Graceful Non-Interactive Fallback**: If running in a headless CI/CD environment (detected via `not sys.stdin.isatty()` or `CI=true`), interactive prompts are skipped and actionable JSON error diagnostics are emitted.
3. **Zero Configuration Leakage**: Discovered project IDs, organization numbers, and account emails remain local to `~/.config/cli-agent/config.toml` and are never committed or logged in plaintext.
4. **Idempotence & Caching**: Pre-flight status is cached with a 15-minute TTL to avoid redundant CLI queries on repeated executions.

---

### 3. Module Index

| Specification | Focus Area | Core Responsibilities |
| :--- | :--- | :--- |
| **[01_gcloud_session_and_interactive_auth.md](01_gcloud_session_and_interactive_auth.md)** | Auth & Sessions | Active account discovery, OAuth token validation, browser login trigger. |
| **[02_project_picker_and_workspace_config.md](02_project_picker_and_workspace_config.md)** | Project Discovery | Dynamic project enumeration, interactive terminal selector, config save. |
| **[03_service_api_auto_enablement.md](03_service_api_auto_enablement.md)** | Service APIs | Service Usage verification, one-click `aiplatform` API activation. |
| **[04_iam_permissions_and_billing_guards.md](04_iam_permissions_and_billing_guards.md)** | IAM & Billing | IAM role verification, billing account verification, quota diagnostics. |
| **[05_guided_setup_and_doctor_repair.md](05_guided_setup_and_doctor_repair.md)** | UX & Repair | `/setup` wizard, `/doctor --fix` auto-repair, headless CI mode. |
