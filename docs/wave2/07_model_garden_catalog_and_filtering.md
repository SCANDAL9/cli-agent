# Specification: Vertex AI Model Garden Dynamic Discovery & Smart Filtering

## 1. Problem Statement: Static Lists & Model Deprecation

Cloud AI platforms evolve rapidly:
- Google Cloud Vertex AI continuously introduces new generations and deprecates older foundation models.
- Vertex AI Model Garden is not limited to Google Gemini; it hosts **Anthropic (Claude Sonnet 5, Claude Opus 5)**, **xAI (Grok 4.7, Grok 4.20 Reasoning)**, **Mistral (Codestral 2)**, and **OpenAI (MaaS)** under the same Google Cloud project authentication.
- Static lists include retired or obsolete models, while missing newly released reasoning models.
- Many models in Model Garden are fine-tuning checkpoints (`CAN_DEPLOY=Yes, CAN_PREDICT=No`), embedding models, or vision/TTS models unsuited for autonomous coding.

---

## 2. Dynamic Discovery Architecture

```
                    cli-agent Interactive REPL (/model)
                                     │
                                     ▼
                Dynamic Discovery Engine (`src/cli_agent/models.py`)
                                     │
                                     ▼
      Execute `gcloud ai model-garden models list --billing-project=<project>`
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────────┐
         │              Smart Capability & Quality Filter             │
         │  1. `CAN_PREDICT == True` (Instant serverless execution)  │
         │  2. Exclude non-coding models (TTS, Vision, Embedding)    │
         │  3. Exclude legacy/obsolete checkpoints (Gemma 1.1, etc.) │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
                       Multi-Publisher Model Catalog
         ┌───────────────────┬───────────────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼
    Google Gemini      Anthropic Claude       xAI Grok        Mistral Codestral
   - gemini-3.8-flash  - claude-sonnet-5   - grok-4.7         - codestral-2
   - gemini-4-argon    - claude-opus-5     - grok-4.20-reas.  - mistral-med-3
         │                   │                   │                   │
         ▼                   ▼                   ▼                   ▼
   Vertex Gemini       Vertex Claude       Vertex MaaS        Vertex MaaS
   :generateContent    :rawPredict         ChatCompletions    ChatCompletions
```

---

## 3. Publisher Protocol Adapters on Vertex AI

All models authenticate with the active Google Cloud OAuth2 token (`TokenProvider`) and project quota (`<gcp-project-id>`), but route through publisher-specific endpoints:

| Publisher Family | Endpoint Pattern | Payload Protocol |
| :--- | :--- | :--- |
| **Google (`google/*`)** | `.../publishers/google/models/{id}:streamGenerateContent` | Native Gemini `contents` + `tools` |
| **Anthropic (`anthropic/*`)** | `.../publishers/anthropic/models/{id}:streamRawPredict` | Anthropic Messages API (`messages`, `system`, `tools`) |
| **xAI (`xai/*`)** | `.../publishers/xai/models/{id}:streamRawPredict` | OpenAI-compatible Chat Completions |
| **Mistral (`mistralai/*`)** | `.../publishers/mistralai/models/{id}:streamRawPredict` | OpenAI-compatible Chat Completions |

---

## 4. Smart Filtering Logic

```python
def is_viable_coding_model(model_id: str, can_predict: bool) -> bool:
    """Filters out models unsuited for autonomous code development."""
    if not can_predict:
        return False

    mid = model_id.lower()

    # Exclude non-text/code modalities
    non_coding_keywords = [
        "embedding",
        "tts",
        "transcribe",
        "image2video",
        "video",
        "vision",
        "audio",
        "voice",
        "nano",
        "try-on",
    ]
    if any(k in mid for k in non_coding_keywords):
        return False

    # Exclude obsolete / legacy checkpoints
    legacy_prefixes = ["google/gemma@gemma-1", "openlm-research", "openai/clip"]
    if any(mid.startswith(p) for p in legacy_prefixes):
        return False

    return True
```

---

## 5. Interactive Terminal Experience (`/model`)

Running `/model` renders a structured, categorized terminal selector:

```
[bold magenta]Vertex AI Live Model Garden (Billing: <gcp-project-id>)[/bold magenta]

── [bold cyan]Google Gemini[/bold cyan] ──────────────────────────────────────────
  [1] google/gemini-3.8-flash            [Active]  Fast, lightweight, 1M context
  [2] google/gemini-3.7-flash                      Balanced reasoning
  [3] google/gemini-4-argon                        Next-gen architecture

── [bold yellow]Anthropic Claude[/bold yellow] ────────────────────────────────────
  [4] anthropic/claude-sonnet-5                    Elite coding & refactoring
  [5] anthropic/claude-opus-5                      Deep multi-step reasoning

── [bold red]xAI Grok[/bold red] ──────────────────────────────────────────────
  [6] xai/grok-4.7                                 Fast reasoning
  [7] xai/grok-4.20-reasoning                      High-depth code synthesis

── [bold green]Mistral AI[/bold green] ──────────────────────────────────────────
  [8] mistralai/codestral-2                        Specialized code model

Select model [1-8] or type custom ID (/model refresh to re-query):
```

Selecting a model updates the active provider adapter without resetting session memory or workspace state.
