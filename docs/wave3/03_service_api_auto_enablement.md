# Wave 3: Enterprise Pre-Flight Engine
## 03. Service API Verification & Auto-Enablement Gate

### 1. The Service Disabled Problem

When a Google Cloud project is newly created or lacks Vertex AI workloads, attempting any API call generates:
`403 Forbidden: Vertex AI API has not been used in project X before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project=X`

The agent should intercept this condition **before** making user model requests, and offer one-click enablement.

---

### 2. Service Verification Protocol

1. **Pre-Flight Query**:
   Query enabled services on the resolved project:
   `gcloud services list --enabled --filter="config.name:aiplatform.googleapis.com" --project=<PROJECT_ID> --format="json"`
2. **If Disabled**:
   - Check if current user has `serviceusage.services.enable` permission.
   - If interactive, prompt the user:
     ```
     [bold yellow]⚠ Vertex AI API (aiplatform.googleapis.com) is not enabled on project 'my-project'.[/bold yellow]
     Would you like cli-agent to enable it now? [Y/n]:
     ```
   - On confirmation, execute:
     `gcloud services enable aiplatform.googleapis.com --project=<PROJECT_ID>`
   - Display a spinner: `[bold cyan]Enabling Vertex AI API (this may take up to 30 seconds)...[/bold cyan]`
   - Verify enablement and continue.
3. **If Insufficient Permissions to Enable**:
   - Provide exact copy-paste command for the organization admin:
     `gcloud services enable aiplatform.googleapis.com --project=<PROJECT_ID>`
