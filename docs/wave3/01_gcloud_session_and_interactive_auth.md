# Wave 3: Enterprise Pre-Flight Engine
## 01. GCloud Session Detection & Interactive Authentication

### 1. Motivation & Context

Without an active Google Cloud authentication session, any request to Vertex AI immediately aborts with an OAuth error. Rather than expecting the developer to decipher `gcloud auth print-access-token` failures, `cli-agent` detects session status and guides the user through authentication interactively.

---

### 2. Authentication Resolution Hierarchy

The engine verifies credentials using the following priority order:

1. **Explicit Token Override (`GCLOUD_ACCESS_TOKEN`)**:
   - Useful in ephemeral CI/CD environments or containerized workers.
   - Verified via a lightweight probe to the Google tokeninfo endpoint.
2. **Application Default Credentials (ADC)**:
   - Located at `~/.config/gcloud/application_default_credentials.json` or `GOOGLE_APPLICATION_CREDENTIALS`.
3. **Active `gcloud` CLI User Session**:
   - Inspected via `gcloud auth list --filter=status:ACTIVE --format="json"`.
   - Token retrieved asynchronously via `gcloud auth print-access-token`.

---

### 3. Interactive Login Trigger

If no active session or credentials are found:

1. **Interactive Terminal Mode (`sys.stdin.isatty()`)**:
   - The user is prompted:
     ```
     [bold red]No active Google Cloud session detected.[/bold red]
     Would you like to log in now via gcloud? [Y/n]:
     ```
   - Upon confirmation, launches:
     `gcloud auth login --update-adc`
   - Monitors subprocess completion, checks exit code, and immediately refreshes the in-memory token provider.
2. **Headless / Non-Interactive Mode**:
   - Emits structured diagnostic guidance:
     ```
     Error: GCloud session not authenticated.
     Run: 'gcloud auth login --update-adc' or export GCLOUD_ACCESS_TOKEN.
     ```
