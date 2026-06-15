"""Tests for tools/stream_parser.py — including regression for the
tool-result-in-user-message bug fixed during Phase 8 P8-1c."""
import json

from code_capsules.runtime.stream_parser import parse_stream


def _make_stream(events: list[dict]) -> str:
    """Serialize a list of stream events as NDJSON (one event per line)."""
    return "\n".join(json.dumps(e) for e in events) + "\n"


class TestToolResultParsing:
    """
    Claude Code emits tool results as content blocks INSIDE user messages,
    not as top-level events. The original parser only looked at top-level
    {"type":"tool_result"}, leaving output_size = 0 on every tool call.
    """

    def test_tool_result_inside_user_message_populates_output_size(self):
        """The regression case: tool results live inside user.message.content."""
        events = [
            {"type": "system", "subtype": "init", "model": "test", "session_id": "s1"},
            {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": 100, "output_tokens": 20,
                              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read",
                         "input": {"file_path": "/tmp/foo.py"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1",
                         "content": "line1\nline2\nline3"},
                    ],
                },
            },
            {"type": "result", "is_error": False, "num_turns": 1,
             "total_cost_usd": 0.01, "duration_ms": 100},
        ]
        s = parse_stream(_make_stream(events))
        assert len(s.tool_calls) == 1
        tc = s.tool_calls[0]
        assert tc.name == "Read"
        assert tc.file_path == "/tmp/foo.py"
        assert tc.output_size == len("line1\nline2\nline3"), \
            f"output_size should be non-zero; got {tc.output_size}"
        assert "line1" in tc.output_text

    def test_tool_result_list_content_format(self):
        """Tool result `content` can also be a list of {type, text} blocks."""
        events = [
            {"type": "system", "subtype": "init", "model": "test", "session_id": "s1"},
            {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "content": [
                        {"type": "tool_use", "id": "tA", "name": "Bash",
                         "input": {"command": "ls"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tA",
                         "content": [{"type": "text", "text": "file1.txt"},
                                     {"type": "text", "text": "file2.txt"}]},
                    ],
                },
            },
            {"type": "result", "num_turns": 1, "total_cost_usd": 0.01, "duration_ms": 100},
        ]
        s = parse_stream(_make_stream(events))
        assert len(s.tool_calls) == 1
        # joined with space → "file1.txt file2.txt"
        assert "file1.txt" in s.tool_calls[0].output_text
        assert s.tool_calls[0].output_size > 0

    def test_unmatched_tool_result_ignored(self):
        """A tool_result whose tool_use_id wasn't seen in a prior assistant turn
        should not crash; ToolCallRecord is keyed on a known pending id."""
        events = [
            {"type": "system", "subtype": "init", "model": "test", "session_id": "s1"},
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "ghost_id",
                         "content": "orphan"},
                    ],
                },
            },
            {"type": "result", "num_turns": 0, "total_cost_usd": 0.0, "duration_ms": 10},
        ]
        s = parse_stream(_make_stream(events))
        assert s.tool_calls == []
        # And no exception

    def test_top_level_tool_result_still_works(self):
        """Backwards-compat: if an adapter emits top-level tool_result events,
        the parser still picks them up. (Unused in current Claude Code stream,
        but keep the branch wired for resilience.)"""
        events = [
            {"type": "system", "subtype": "init", "model": "test", "session_id": "s1"},
            {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "content": [
                        {"type": "tool_use", "id": "t2", "name": "Read",
                         "input": {"file_path": "/x.py"}},
                    ],
                },
            },
            {"type": "tool_result", "tool_use_id": "t2",
             "content": "top-level result format"},
            {"type": "result", "num_turns": 1, "total_cost_usd": 0.01, "duration_ms": 100},
        ]
        s = parse_stream(_make_stream(events))
        assert len(s.tool_calls) == 1
        assert s.tool_calls[0].output_size == len("top-level result format")


class TestEndToEndOnRealArchive:
    """If a captured Phase 8 archive exists, sanity-check it parses with non-zero output."""

    def test_real_archive_has_nonzero_output_sizes(self, tmp_path):
        # Skip if no archives present (e.g., fresh checkout, never ran Phase 8 batch)
        import os
        archives = []
        evals_dir = "evals/streams"  # per-instance stream archives are not distributed; skips if absent
        if os.path.isdir(evals_dir):
            for sub in os.listdir(evals_dir):
                full = os.path.join(evals_dir, sub)
                if os.path.isdir(full):
                    for f in os.listdir(full):
                        if f.endswith(".jsonl"):
                            archives.append(os.path.join(full, f))
                            break
                    if archives: break
        if not archives:
            import pytest
            pytest.skip("no real archive available — skip end-to-end check")
        text = open(archives[0]).read()
        s = parse_stream(text)
        assert len(s.tool_calls) > 0
        # At least one tool call should have non-zero output_size
        assert any(tc.output_size > 0 for tc in s.tool_calls), \
            "expected at least one tool call with non-zero output_size on a real archive"
