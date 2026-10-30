"""
Unit tests for conversation context window compaction and session summarization.
"""

import pytest

from cli_agent.context import (
    compact_history,
    estimate_context_tokens,
    extract_session_learnings,
    prune_tool_responses,
)


@pytest.mark.unit
class TestContextCompaction:
    def test_compact_history_under_threshold_returns_intact(self):
        history = [
            {"role": "user", "parts": [{"text": "turn 1"}]},
            {"role": "model", "parts": [{"text": "response 1"}]},
        ]
        compacted = compact_history(history, max_turns=6, keep_recent=2)
        assert len(compacted) == 2
        assert compacted == history

    def test_compact_history_over_threshold_summarizes_older_turns(self):
        history = [
            {"role": "user", "parts": [{"text": f"User question {i}"}]}
            if i % 2 == 0
            else {"role": "model", "parts": [{"text": f"Model answer {i}"}]}
            for i in range(10)
        ]

        # max_turns=6, keep_recent=4
        compacted = compact_history(history, max_turns=6, keep_recent=4)

        # Result should have 2 synthetic summary turns + 4 recent turns = 6 turns
        assert len(compacted) == 6
        assert "[CONTEXT COMPACTION SUMMARY" in compacted[0]["parts"][0]["text"]
        # Verify the most recent turns are preserved exactly
        assert compacted[-1]["parts"][0]["text"] == "Model answer 9"
        assert compacted[-2]["parts"][0]["text"] == "User question 8"

    def test_prune_tool_responses_truncates_oversized_payloads(self):
        head_str = "A" * 500
        middle_str = "M" * 4000
        tail_str = "Z" * 1500
        huge_output = head_str + middle_str + tail_str  # total 6000 chars

        history = [
            {"role": "user", "parts": [{"text": "Read large file"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "read_file", "args": {"path": "big.py"}}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read_file",
                            "response": {"result": {"content": huge_output, "lines": 500}},
                        }
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "Done reading."}]},
        ]

        pruned = prune_tool_responses(history, max_chars=4000)
        pruned_content = pruned[2]["parts"][0]["functionResponse"]["response"]["result"]["content"]

        assert pruned_content.startswith(head_str)
        assert pruned_content.endswith(tail_str)
        assert "\n[... truncated 4000 chars of prior tool response ...]\n" in pruned_content
        # Original history must not be mutated in place
        assert (
            len(history[2]["parts"][0]["functionResponse"]["response"]["result"]["content"]) == 6000
        )

    def test_prune_tool_responses_list_and_direct_string_payloads(self):
        assert prune_tool_responses([]) == []

        huge_str = "X" * 5000
        large_list = [f"item_{i}_" + ("Y" * 100) for i in range(60)]
        large_flat_dict = {f"key_{i}": i for i in range(600)}

        history = [
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "str_tool", "response": huge_str}},
                    {"functionResponse": {"name": "list_tool", "response": large_list}},
                    {"functionResponse": {"name": "dict_tool", "response": large_flat_dict}},
                    {"functionResponse": {"name": "small_list", "response": ["short"]}},
                    {"functionResponse": {"name": "int_tool", "response": 12345}},
                ],
            }
        ]

        pruned = prune_tool_responses(history, max_chars=4000)
        p_parts = pruned[0]["parts"]
        assert (
            "[... truncated 3000 chars of prior tool response ...]"
            in p_parts[0]["functionResponse"]["response"]
        )
        assert "[... truncated " in p_parts[1]["functionResponse"]["response"]
        assert "[... truncated " in p_parts[2]["functionResponse"]["response"]["result"]
        assert p_parts[3]["functionResponse"]["response"] == ["short"]

    def test_compact_history_integrates_prune_tool_responses(self):
        huge_text = "H" * 500 + "B" * 3500 + "T" * 1500
        history = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "run_command",
                            "response": {"result": {"stdout": huge_text}},
                        }
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "Analyzed output."}]},
        ]
        compacted = compact_history(history, max_turns=12, keep_recent=6)
        stdout_after = compacted[0]["parts"][0]["functionResponse"]["response"]["result"]["stdout"]
        assert "\n[... truncated 3500 chars of prior tool response ...]\n" in stdout_after

        # Also test compact_history when over max_turns with tool calls and responses in older turns
        long_history = [
            {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}}]},
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read_file",
                            "response": {"result": {"content": huge_text}},
                        }
                    }
                ],
            },
            {"role": "user", "parts": [{"text": "q1"}]},
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"text": "q2"}]},
            {"role": "model", "parts": [{"text": "a2"}]},
        ]
        compacted_long = compact_history(long_history, max_turns=4, keep_recent=2)
        assert len(compacted_long) == 4
        summary_txt = compacted_long[0]["parts"][0]["text"]
        assert "[tool call]: read_file" in summary_txt
        assert "[tool response]: read_file" in summary_txt

    def test_estimate_context_tokens(self):
        assert estimate_context_tokens([]) == 0
        history = [
            {"role": "user", "parts": [{"text": "A" * 400}]},
            {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {}}}]},
        ]
        tokens = estimate_context_tokens(history)
        assert tokens >= 100

    def test_extract_session_learnings(self):
        history = [
            {"role": "user", "parts": [{"text": "Configure PostgreSQL database"}]},
            {"role": "model", "parts": [{"text": "Configured."}]},
            {"role": "user", "parts": [{"text": "/exit"}]},  # Slash command should be excluded
        ]
        learnings = extract_session_learnings(history)
        assert "Configure PostgreSQL database" in learnings
        assert "/exit" not in learnings

    def test_extract_session_learnings_empty(self):
        empty_res = extract_session_learnings([])
        assert "no specific user instructions" in empty_res
