"""
Context window management and conversation history compaction for cli-agent.
Prevents context token bloat by summarizing and pruning older turns.
"""

import copy
import json
from typing import Any


def _prune_string(text: str, max_chars: int = 4000) -> str:
    """Truncates a string exceeding max_chars to keep head (500) and tail (1500)."""
    if len(text) <= max_chars:
        return text
    head_len = 500
    tail_len = 1500
    truncated_count = max(0, len(text) - (head_len + tail_len))
    return (
        text[:head_len]
        + f"\n[... truncated {truncated_count} chars of prior tool response ...]\n"
        + text[-tail_len:]
    )


def _prune_response_payload(payload: Any, max_chars: int = 4000) -> Any:
    """Prunes string values or serialized payloads inside a functionResponse object."""
    if isinstance(payload, str):
        return _prune_string(payload, max_chars)
    if isinstance(payload, dict):
        pruned_dict: dict[str, Any] = {}
        any_pruned = False
        for k, v in payload.items():
            if isinstance(v, str) and len(v) > max_chars:
                pruned_dict[k] = _prune_string(v, max_chars)
                any_pruned = True
            elif isinstance(v, (dict, list)):
                new_v = _prune_response_payload(v, max_chars)
                if new_v != v:
                    any_pruned = True
                pruned_dict[k] = new_v
            else:
                pruned_dict[k] = v
        if any_pruned:
            return pruned_dict
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) > max_chars:
            return {"result": _prune_string(serialized, max_chars)}
        return pruned_dict
    if isinstance(payload, list):
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) > max_chars:
            return _prune_string(serialized, max_chars)
        return [_prune_response_payload(item, max_chars) for item in payload]
    return payload


def prune_tool_responses(
    history: list[dict[str, Any]],
    max_chars: int = 4000,
) -> list[dict[str, Any]]:
    """
    Prunes older turns containing `functionResponse` parts whose content exceeds `max_chars`.
    Preserves the first 500 chars and last 1500 chars separated by a truncation marker.
    """
    if not history:
        return []

    pruned_history: list[dict[str, Any]] = []
    # Prune all turns except the very latest turn if there are multiple turns,
    # or prune any turn containing oversized functionResponse parts.
    for turn in history:
        new_turn = copy.deepcopy(turn)
        parts = new_turn.get("parts", [])
        for part in parts:
            if "functionResponse" in part and isinstance(part["functionResponse"], dict):
                fn_resp = part["functionResponse"]
                resp_data = fn_resp.get("response")
                serialized = json.dumps(resp_data, ensure_ascii=False, default=str)
                if len(serialized) > max_chars:
                    fn_resp["response"] = _prune_response_payload(resp_data, max_chars)
        pruned_history.append(new_turn)

    return pruned_history


def estimate_context_tokens(history: list[dict[str, Any]]) -> int:
    """
    Estimates the token count of the conversation history using standard heuristics (~4 chars/token).
    """
    if not history:
        return 0
    total_chars = 0
    for turn in history:
        parts = turn.get("parts", [])
        for part in parts:
            if "text" in part and isinstance(part["text"], str):
                total_chars += len(part["text"])
            else:
                total_chars += len(json.dumps(part, ensure_ascii=False, default=str))
    return max(1, total_chars // 4) if total_chars > 0 else 0


def compact_history(
    history: list[dict[str, Any]],
    max_turns: int = 12,
    keep_recent: int = 6,
    max_tool_chars: int = 4000,
) -> list[dict[str, Any]]:
    """
    Compacts conversation history when turn count exceeds max_turns, and prunes
    oversized tool responses so multi-turn sessions never suffer context blowout.
    """
    pruned_history = prune_tool_responses(history, max_chars=max_tool_chars)

    if len(pruned_history) <= max_turns:
        return pruned_history

    older_turns = pruned_history[:-keep_recent]
    recent_turns = pruned_history[-keep_recent:]

    # Extract high-level topic snippets from older turns
    summary_snippets: list[str] = []
    for turn in older_turns:
        role = turn.get("role", "user")
        parts = turn.get("parts", [])
        for p in parts:
            if "text" in p and p["text"]:
                text = p["text"].strip().replace("\n", " ")
                preview = text[:120] + ("..." if len(text) > 120 else "")
                summary_snippets.append(f"- [{role}]: {preview}")
            elif "functionCall" in p:
                fn = p["functionCall"].get("name", "tool")
                summary_snippets.append(f"- [tool call]: {fn}")
            elif "functionResponse" in p:
                fn = p["functionResponse"].get("name", "tool")
                summary_snippets.append(f"- [tool response]: {fn}")

    summary_content = (
        "[CONTEXT COMPACTION SUMMARY - Older session turns summarized]:\n"
        + "\n".join(summary_snippets[:10])
    )

    compacted: list[dict[str, Any]] = [
        {"role": "user", "parts": [{"text": summary_content}]},
        {"role": "model", "parts": [{"text": "Understood. I have absorbed the prior context."}]},
    ]
    compacted.extend(recent_turns)
    return compacted


def extract_session_learnings(history: list[dict[str, Any]]) -> str:
    """
    Extracts key discussion points from a completed session for persistence.
    """
    topics: list[str] = []
    for turn in history:
        if turn.get("role") == "user":
            for p in turn.get("parts", []):
                if "text" in p:
                    text = p["text"].strip()
                    if text and not text.startswith("/"):
                        topics.append(text[:100])

    if not topics:
        return "Session completed with no specific user instructions recorded."

    return "Session accomplishments and topics discussed:\n" + "\n".join(
        f"- {t}" for t in topics[:8]
    )
