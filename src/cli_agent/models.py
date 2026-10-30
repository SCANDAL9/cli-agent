"""
Dynamic Model Garden discovery and catalog management for Google Cloud Vertex AI.
Queries current live models across Google Gemini, Anthropic Claude, xAI Grok, and Mistral.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_CACHE = Path.home() / ".config" / "cli-agent" / "model_catalog.json"
CACHE_TTL_SECONDS = 86400  # 24 hours

NON_CODING_KEYWORDS = {
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
    "clip",
    "deeplabv3",
    "sentiment",
    "syntax",
    "classify-text",
    "occupancy",
    "vehicle",
    "aligner",
    "scribe",
    "stt",
    "lyria",
    "veo",
}

LEGACY_CHECKPOINTS = {
    "gemma@gemma-1",
    "openllama",
    "palmyra",
}


@dataclass
class ModelCatalogEntry:
    model_id: str
    publisher: str
    short_name: str
    launch_stage: str = "GA"
    version_id: str = "default"
    can_predict: bool = True


# Curated high-performance fallback baseline in case gcloud is unreachable
BASELINE_CATALOG: list[ModelCatalogEntry] = [
    ModelCatalogEntry(
        model_id="publishers/google/models/gemini-3.8-flash",
        publisher="google",
        short_name="gemini-3.8-flash",
        launch_stage="GA",
    ),
    ModelCatalogEntry(
        model_id="publishers/google/models/gemini-3.7-flash",
        publisher="google",
        short_name="gemini-3.7-flash",
        launch_stage="GA",
    ),
    ModelCatalogEntry(
        model_id="publishers/google/models/gemini-4-argon",
        publisher="google",
        short_name="gemini-4-argon",
        launch_stage="GA",
    ),
    ModelCatalogEntry(
        model_id="publishers/anthropic/models/claude-sonnet-5",
        publisher="anthropic",
        short_name="claude-sonnet-5",
        launch_stage="GA",
    ),
    ModelCatalogEntry(
        model_id="publishers/anthropic/models/claude-opus-5",
        publisher="anthropic",
        short_name="claude-opus-5",
        launch_stage="GA",
    ),
    ModelCatalogEntry(
        model_id="publishers/xai/models/grok-4.7",
        publisher="xai",
        short_name="grok-4.7",
        launch_stage="PUBLIC_PREVIEW",
    ),
    ModelCatalogEntry(
        model_id="publishers/xai/models/grok-4.20-reasoning",
        publisher="xai",
        short_name="grok-4.20-reasoning",
        launch_stage="PUBLIC_PREVIEW",
    ),
    ModelCatalogEntry(
        model_id="publishers/mistralai/models/codestral-2",
        publisher="mistralai",
        short_name="codestral-2",
        launch_stage="GA",
    ),
]


def is_viable_coding_model(model_id: str) -> bool:
    """Filters out non-text/code modalities and obsolete checkpoints."""
    mid = model_id.lower()

    if any(kw in mid for kw in NON_CODING_KEYWORDS):
        return False

    if any(leg in mid for leg in LEGACY_CHECKPOINTS):
        return False

    # Must belong to an AI generative foundation publisher
    recognized_publishers = {
        "google",
        "anthropic",
        "xai",
        "mistralai",
        "mistral-ai",
        "openai",
        "deepseek-ai",
    }
    parts = mid.split("/")
    if len(parts) >= 2 and parts[0] == "publishers":
        pub = parts[1]
        return pub in recognized_publishers

    return False


def parse_model_entry(raw_item: dict[str, Any]) -> ModelCatalogEntry | None:
    """Constructs a ModelCatalogEntry from gcloud Model Garden JSON output."""
    raw_name = raw_item.get("name", "")
    if not is_viable_coding_model(raw_name):
        return None

    parts = raw_name.split("/")
    publisher = parts[1] if len(parts) > 1 else "google"
    short_name = parts[-1] if parts else raw_name

    return ModelCatalogEntry(
        model_id=raw_name,
        publisher=publisher,
        short_name=short_name,
        launch_stage=raw_item.get("launchStage", "GA"),
        version_id=raw_item.get("versionId", "default"),
        can_predict=True,
    )


def detect_publisher(model_name_or_id: str) -> str:
    """Infers the publisher namespace from a model identifier or name."""
    mid = model_name_or_id.lower()
    if "anthropic" in mid or "claude" in mid:
        return "anthropic"
    if "xai" in mid or "grok" in mid:
        return "xai"
    if "mistral" in mid or "codestral" in mid:
        return "mistralai"
    if "openai" in mid:
        return "openai"
    return "google"


async def fetch_model_garden_catalog(
    billing_project: str,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> list[ModelCatalogEntry]:
    """
    Fetches real-time Model Garden models from Vertex AI with local disk caching.
    """
    cache_file = cache_path or DEFAULT_CATALOG_CACHE

    if not force_refresh and cache_file.is_file():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached_payload = json.load(f)
            cached_time = cached_payload.get("timestamp", 0)
            if time.time() - cached_time < CACHE_TTL_SECONDS:
                entries_raw = cached_payload.get("models", [])
                return [ModelCatalogEntry(**e) for e in entries_raw]
        except Exception:
            pass

    # Query gcloud model garden asynchronously
    try:
        proc = await asyncio.create_subprocess_exec(
            "gcloud",
            "ai",
            "model-garden",
            "models",
            "list",
            f"--billing-project={billing_project}",
            "--format=json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        raw_text = stdout.decode("utf-8", errors="replace")

        # Extract JSON array
        start_idx = raw_text.find("[")
        end_idx = raw_text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_text[start_idx : end_idx + 1]
            raw_models = json.loads(json_str)
            entries: list[ModelCatalogEntry] = []
            seen_shorts: set[str] = set()

            for item in raw_models:
                parsed = parse_model_entry(item)
                if parsed and parsed.short_name not in seen_shorts:
                    seen_shorts.add(parsed.short_name)
                    entries.append(parsed)

            if entries:
                # Save to cache
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "timestamp": time.time(),
                                "models": [asdict(e) for e in entries],
                            },
                            f,
                            indent=2,
                        )
                except Exception:
                    pass
                return entries
    except Exception:
        pass

    return BASELINE_CATALOG
