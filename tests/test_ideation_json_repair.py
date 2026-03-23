"""Unit tests for JSON backslash repair and FinalizeIdea retry logic.

Tests cover:
- repair_json_backslashes() correctly escapes invalid sequences.
- Valid JSON escapes (\\n, \\t, \\uXXXX, etc.) are left untouched.
- parse_finalize_idea_args() succeeds when the LLM re-prompt returns valid JSON.
"""
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so imports work without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_scientist.perform_ideation_temp_free import (
    parse_finalize_idea_args,
    repair_json_backslashes,
)


# ---------------------------------------------------------------------------
# repair_json_backslashes
# ---------------------------------------------------------------------------

class TestRepairJsonBackslashes:
    def test_invalid_latex_backslash_is_escaped(self):
        """\\alpha (invalid JSON escape) should become \\\\alpha."""
        # Use \alpha and \delta – neither \a nor \d is a valid JSON escape char.
        raw = r'{"formula": "\alpha + \delta"}'
        repaired = repair_json_backslashes(raw)
        parsed = json.loads(repaired)
        assert parsed["formula"] == r"\alpha + \delta"

    def test_valid_newline_escape_unchanged(self):
        """\\n must NOT be double-escaped; it is a valid JSON escape."""
        raw = '{"text": "line1\\nline2"}'
        repaired = repair_json_backslashes(raw)
        assert repaired == raw, "\\n should be left untouched"
        parsed = json.loads(repaired)
        assert parsed["text"] == "line1\nline2"

    def test_valid_tab_escape_unchanged(self):
        raw = '{"text": "col1\\tcol2"}'
        repaired = repair_json_backslashes(raw)
        assert repaired == raw
        assert json.loads(repaired)["text"] == "col1\tcol2"

    def test_valid_unicode_escape_unchanged(self):
        raw = '{"char": "\\u0041"}'
        repaired = repair_json_backslashes(raw)
        assert repaired == raw
        assert json.loads(repaired)["char"] == "A"

    def test_valid_double_backslash_unchanged(self):
        """An already-escaped backslash (\\\\) must not be double-escaped again."""
        raw = '{"path": "C:\\\\Users\\\\foo"}'
        repaired = repair_json_backslashes(raw)
        assert repaired == raw
        assert json.loads(repaired)["path"] == "C:\\Users\\foo"

    def test_mixed_valid_and_invalid(self):
        """A string with both valid (\\n) and invalid (\\x) escapes."""
        raw = '{"msg": "hello\\nworld\\x21"}'
        repaired = repair_json_backslashes(raw)
        parsed = json.loads(repaired)
        # \\n stays as newline, \\x21 becomes literal \x21
        assert "hello" in parsed["msg"]
        assert "world" in parsed["msg"]

    def test_already_valid_json_unchanged(self):
        """Passing already-valid JSON must not corrupt it."""
        raw = '{"a": 1, "b": "hello"}'
        assert repair_json_backslashes(raw) == raw


# ---------------------------------------------------------------------------
# parse_finalize_idea_args
# ---------------------------------------------------------------------------

_VALID_IDEA = {"idea": {"Name": "test", "Title": "Test Idea"}}
_VALID_JSON_STR = json.dumps(_VALID_IDEA)

_DUMMY_CLIENT = MagicMock()
_DUMMY_MODEL = "claude-3-5-sonnet-20241022"


class TestParseFinalizeIdeaArgs:
    def test_valid_json_parsed_directly(self):
        result = parse_finalize_idea_args(_VALID_JSON_STR, _DUMMY_CLIENT, _DUMMY_MODEL)
        assert result == _VALID_IDEA

    def test_invalid_escape_repaired_automatically(self):
        """JSON containing \\alpha (invalid escape) is repaired without LLM call."""
        raw = r'{"idea": {"Name": "alpha_idea", "formula": "\alpha + \delta"}}'
        result = parse_finalize_idea_args(raw, _DUMMY_CLIENT, _DUMMY_MODEL)
        assert result["idea"]["Name"] == "alpha_idea"
        assert result["idea"]["formula"] == r"\alpha + \delta"

    def test_llm_reprompt_used_on_second_attempt(self):
        """If repair fails, the LLM is re-prompted and the corrected JSON is used."""
        # Completely broken JSON that repair_json_backslashes cannot fix
        broken_json = '{"idea": {NOTJSON'

        # LLM returns valid JSON on the first re-prompt call
        llm_response_text = _VALID_JSON_STR

        with patch(
            "ai_scientist.perform_ideation_temp_free.get_response_from_llm",
            return_value=(llm_response_text, []),
        ) as mock_llm:
            result = parse_finalize_idea_args(broken_json, _DUMMY_CLIENT, _DUMMY_MODEL)

        assert result == _VALID_IDEA
        # The LLM must have been called at least once
        mock_llm.assert_called()

    def test_raises_after_max_attempts(self):
        """After max_attempts all fail, a ValueError is raised with a preview."""
        broken_json = '{"idea": {NOTJSON'

        # LLM always returns broken JSON
        with patch(
            "ai_scientist.perform_ideation_temp_free.get_response_from_llm",
            return_value=(broken_json, []),
        ):
            with pytest.raises(ValueError, match="Failed to parse FinalizeIdea"):
                parse_finalize_idea_args(
                    broken_json, _DUMMY_CLIENT, _DUMMY_MODEL, max_attempts=3
                )

    def test_error_message_contains_preview_and_hint(self):
        """The raised ValueError must contain a text preview and backslash hint."""
        broken_json = '{"idea": {NOTJSON_SENTINEL'

        with patch(
            "ai_scientist.perform_ideation_temp_free.get_response_from_llm",
            return_value=(broken_json, []),
        ):
            with pytest.raises(ValueError) as exc_info:
                parse_finalize_idea_args(
                    broken_json, _DUMMY_CLIENT, _DUMMY_MODEL, max_attempts=2
                )

        msg = str(exc_info.value)
        assert "SENTINEL" in msg, "Error should include a preview of the bad text"
        assert "backslash" in msg.lower(), "Error should mention backslash escaping"

    def test_llm_response_with_markdown_fences_stripped(self):
        """If LLM wraps JSON in ```json fences, they should be stripped."""
        broken_json = '{"idea": {NOTJSON'
        fenced_response = f"```json\n{_VALID_JSON_STR}\n```"

        with patch(
            "ai_scientist.perform_ideation_temp_free.get_response_from_llm",
            return_value=(fenced_response, []),
        ):
            result = parse_finalize_idea_args(broken_json, _DUMMY_CLIENT, _DUMMY_MODEL)

        assert result == _VALID_IDEA
