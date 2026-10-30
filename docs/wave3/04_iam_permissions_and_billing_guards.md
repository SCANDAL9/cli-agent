# Wave 3: Enterprise Pre-Flight Engine
## 04. IAM Role Validation, Billing Association & Quota Guards

### 1. IAM Permission Verification

To execute Vertex AI predictions, a caller requires at minimum:
- `roles/aiplatform.user` or individual permission `aiplatform.endpoints.predict`
- `serviceusage.services.use` (included in `roles/editor` or `roles/viewer`)

#### Verification Check:
1. Probe permissions using Google Cloud Resource Manager `testIamPermissions`:
   API endpoint: `https://cloudresourcemanager.googleapis.com/v1/projects/{projectId}:testIamPermissions`
   Permissions: `["aiplatform.endpoints.predict"]`
2. Alternatively, perform a lightweight zero-token probe or verify `gcloud projects get-iam-policy`.
3. If permission is missing, emit:
   ```
   [bold red]✖ Missing IAM Role:[/bold red] Your account lacks 'roles/aiplatform.user' on project 'my-project'.
   To grant access, have an administrator run:
     gcloud projects add-iam-policy-binding my-project \
       --member="user:your-email@example.com" \
       --role="roles/aiplatform.user"
   ```

---

### 2. Billing Account Association Guard

Vertex AI calls fail if the project has no active billing account linked (`billingAccountName` is empty).

#### Verification:
1. Query billing status:
   `gcloud billing projects describe <PROJECT_ID> --format="json"`
2. If `billingEnabled: false`:
   ```
   [bold red]✖ Billing Disabled:[/bold red] Project 'my-project' does not have an active billing account linked.
   Link a billing account at: https://console.cloud.google.com/billing/linkedaccount?project=my-project
   ```

---

### 3. Quota & Credit Exhaustion Diagnostics

1. **Detection of 429 ResourceExhausted**:
   When Vertex AI returns HTTP 429:
   - Identify whether it is rate-limit throttling (RPM/TPM) or total quota depletion.
   - Display a human-readable diagnosis with links to the Vertex AI Quotas console.
