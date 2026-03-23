"""Unit tests for the JSON repair helpers in ai_scientist/utils/json_utils.py."""

import json
import sys
import os

import pytest

# Make the ai_scientist package importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_scientist.utils.json_utils import (  # noqa: E402
    parse_arguments_json,
    repair_json_escapes,
)


# ---------------------------------------------------------------------------
# repair_json_escapes
# ---------------------------------------------------------------------------


class TestRepairJsonEscapes:
    """Tests for the low-level repair_json_escapes helper."""

    def test_latex_alpha(self):
        """Bare \\alpha in a JSON string value is escaped to \\\\alpha."""
        raw = '{"formula": "\\alpha"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"formula": "\\alpha"}

    def test_latex_beta_b_is_valid_escape(self):
        r"""\\b is a valid JSON escape (backspace); \\beta is NOT fully repaired.

        This documents a known limitation: the leading ``\\b`` in ``\\beta``
        is a valid JSON backspace escape and is left untouched.  Callers should
        use LaTeX sequences that start with letters outside ``bfnrt`` if they
        need repair.  We test that ``\\gamma`` (which starts with ``g``) IS
        correctly escaped.
        """
        # \gamma starts with 'g' which is not a valid JSON escape char → fixed
        raw = '{"formula": "\\gamma + 1"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"formula": "\\gamma + 1"}

    def test_latex_mathbb(self):
        r"""Bare \mathbb{R} in a JSON string value is escaped."""
        raw = '{"domain": "\\mathbb{R}"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"domain": "\\mathbb{R}"}

    def test_multiple_latex_sequences(self):
        r"""Multiple bare backslash sequences are all escaped."""
        raw = '{"eq": "\\alpha \\in \\mathbb{R}"}'
        repaired = repair_json_escapes(raw)
        result = json.loads(repaired)
        assert result == {"eq": "\\alpha \\in \\mathbb{R}"}

    def test_valid_escape_newline_unchanged(self):
        r"""\\n (valid JSON escape) is left untouched."""
        raw = '{"text": "line1\\nline2"}'
        repaired = repair_json_escapes(raw)
        # json.loads should see a real newline
        assert json.loads(repaired) == {"text": "line1\nline2"}

    def test_valid_escape_tab_unchanged(self):
        r"""\\t (valid JSON escape) is left untouched."""
        raw = '{"text": "col1\\tcol2"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"text": "col1\tcol2"}

    def test_valid_escape_quote_unchanged(self):
        r"""\\\" (valid JSON escape) is left untouched."""
        raw = '{"text": "say \\"hi\\""}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"text": 'say "hi"'}

    def test_windows_path_already_escaped(self):
        r"""C:\\\\Users\\\\name (already escaped backslashes) is not double-escaped."""
        raw = '{"path": "C:\\\\Users\\\\name"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"path": "C:\\Users\\name"}

    def test_valid_json_unchanged(self):
        """Completely valid JSON passes through repair_json_escapes unchanged."""
        original = '{"key": "value", "num": 42, "flag": true}'
        assert repair_json_escapes(original) == original

    def test_unicode_escape_unchanged(self):
        r"""\\uXXXX is a valid JSON escape and must not be modified."""
        raw = '{"char": "\\u0041"}'
        repaired = repair_json_escapes(raw)
        assert json.loads(repaired) == {"char": "A"}


# ---------------------------------------------------------------------------
# parse_arguments_json
# ---------------------------------------------------------------------------


class TestParseArgumentsJson:
    """Tests for the high-level parse_arguments_json helper."""

    def test_valid_json_parsed_directly(self):
        """Valid JSON is returned without any modification."""
        text = '{"idea": {"Name": "test", "Title": "A title"}}'
        result = parse_arguments_json(text)
        assert result == {"idea": {"Name": "test", "Title": "A title"}}

    def test_latex_in_field_is_repaired_and_parsed(self):
        r"""JSON containing \\alpha is repaired and successfully parsed."""
        # Simulate LLM output: the backslash before 'alpha' is not escaped.
        text = '{"idea": {"Abstract": "We study \\alpha-divergence."}}'
        result = parse_arguments_json(text)
        assert result["idea"]["Abstract"] == "We study \\alpha-divergence."

    def test_latex_mathbb_in_field(self):
        r"""JSON containing \\mathbb{R} is repaired and successfully parsed."""
        text = '{"idea": {"Experiments": "Test on \\mathbb{R}^n."}}'
        result = parse_arguments_json(text)
        assert result["idea"]["Experiments"] == "Test on \\mathbb{R}^n."

    def test_valid_newline_escape_preserved(self):
        r"""A \\n in valid JSON is parsed as an actual newline character."""
        text = '{"idea": {"Title": "Line1\\nLine2"}}'
        result = parse_arguments_json(text)
        assert result["idea"]["Title"] == "Line1\nLine2"

    def test_windows_path_in_value(self):
        """Windows paths stored with doubled backslashes parse correctly."""
        text = '{"path": "C:\\\\Users\\\\researcher"}'
        result = parse_arguments_json(text)
        assert result["path"] == "C:\\Users\\researcher"

    def test_raises_value_error_on_unfixable_json(self):
        """Raises ValueError with a helpful message when JSON cannot be repaired."""
        # Deliberately broken JSON that repair cannot fix
        bad_text = '{"unclosed": '
        with pytest.raises(ValueError, match="Failed to parse arguments JSON"):
            parse_arguments_json(bad_text)

    def test_error_message_contains_preview(self):
        """The ValueError message includes a preview of the offending text."""
        bad_text = '{"unclosed": '
        with pytest.raises(ValueError) as exc_info:
            parse_arguments_json(bad_text)
        assert "unclosed" in str(exc_info.value)
