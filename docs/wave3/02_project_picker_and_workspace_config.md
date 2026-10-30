# Wave 3: Enterprise Pre-Flight Engine
## 02. Dynamic Project Picker & Workspace Configuration

### 1. The Multi-Project Problem

Enterprise Google Cloud developers frequently work across dozens of client, staging, and production projects. Forcing manual editing of `config.toml` leads to typos and wrong billing allocations.

---

### 2. Resolution Protocol

When `cli-agent` boots:

```
[Start CLI] ──► Check GCP_PROJECT env var
                     │
                     ├─► Found ──► Validate format and proceed
                     │
                     ▼ Unset
             Check local ~/.config/cli-agent/config.toml
                     │
                     ├─► Found ──► Validate and proceed
                     │
                     ▼ Unset
             Query gcloud config get-value project
                     │
                     ├─► Valid non-empty ──► Use active project
                     │
                     ▼ Unset or "(unset)"
             Enumerate accessible projects (gcloud projects list)
                     │
                     ├─► Interactive TUI Picker (Rich / PromptSession)
                     ▼
             Persist selection to ~/.config/cli-agent/config.toml
```

---

### 3. Interactive Picker Specification

1. Query available projects:
   `gcloud projects list --filter="lifecycleState:ACTIVE" --format="json(projectId,name,projectNumber)"`
2. If multiple projects exist, present a numbered Rich table with fuzzy filtering:
   ```
   Vertex AI Project Selector:
   [1] client-alpha-dev       (ID: client-alpha-dev, #102938475)
   [2] internal-ml-sandbox    (ID: internal-ml-sandbox, #584930219)
   [3] corporate-prod         (ID: corporate-prod, #948302847)
   Select project [1-3] or type project ID:
   ```
3. Upon selection, save setting to `~/.config/cli-agent/config.toml`:
   ```toml
   [defaults]
   project = "client-alpha-dev"
   ```
