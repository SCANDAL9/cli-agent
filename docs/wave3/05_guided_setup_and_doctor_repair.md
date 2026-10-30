# Wave 3: Enterprise Pre-Flight Engine
## 05. Guided Setup Wizard & Self-Healing Doctor

### 1. Interactive `/setup` Wizard

Allows developers to reconfigure environment variables, switch active Google Cloud projects, log into another account, or customize preferences at any time.

#### Workflow:
1. Show current configuration:
   - Account email
   - Active GCP Project
   - Default Vertex AI Model & Region
2. Interactive steps:
   - Option to re-authenticate or switch Google account.
   - Option to choose another project from the active account.
   - Option to switch default model (Gemini 3.8 Flash, Gemini 3.1 Pro, Claude Sonnet 5, etc.).
3. Writes updated configuration directly to `~/.config/cli-agent/config.toml`.

---

### 2. Upgraded `/doctor --fix` Self-Healing Engine

Enhances the existing `/doctor` diagnostics command with an automated `--fix` flag.

```
$ cli-agent --doctor --fix
System & Workspace Diagnostics:
 - OAuth2 Token: [green]Valid[/green] (account: dev@corp.com)
 - Active Project: [yellow]Not Set[/yellow] -> [bold cyan]Auto-discovered 'client-dev' and saved[/bold cyan]
 - Vertex AI API: [red]Disabled[/red] -> [bold cyan]Enabling aiplatform.googleapis.com... Done ✔[/bold cyan]
 - Billing Status: [green]Linked[/green]
 - SQLite Memory & Code Graph: [green]Healthy[/green]
All diagnostics verified and repaired!
```
