"""Lightweight JSON repair utilities with no heavy dependencies.

These helpers are used by perform_ideation_temp_free.py to robustly parse
JSON produced by LLMs that may contain bare (unescaped) backslashes, e.g.
from LaTeX notation like ``\\alpha`` or ``\\mathbb{R}``.
"""

import json
import re

# Match either a valid JSON escape sequence (consumed atomically so the second
# character of ``\\`` is never mistakenly treated as a new escape start), or a
# bare backslash followed by any other character.
#
# Valid single-char JSON escapes: " \ / b f n r t
# Valid multi-char JSON escape:   u followed by exactly 4 hex digits
#
# The named group ``bad`` captures the character following a *bare* (invalid)
# backslash; that match is then doubled by _fix_escape_match().
_ESCAPE_FIXUP_RE = re.compile(
    r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}|(?P<bad>.))',
    re.DOTALL,
)


def _fix_escape_match(m: re.Match) -> str:
    """Return the corrected replacement for a backslash sequence in JSON text."""
    if m.group("bad") is not None:
        # Not a valid JSON escape — double the leading backslash.
        return "\\\\" + m.group("bad")
    # Valid escape sequence — leave it completely unchanged.
    return m.group(0)


def repair_json_escapes(text: str) -> str:
    """Replace invalid backslash escapes in *text* so it can be parsed as JSON.

    LLM outputs containing LaTeX (e.g. ``\\alpha``, ``\\mathbb{R}``) embed bare
    backslashes inside JSON string values.  Those are not valid JSON escape
    sequences and cause ``json.loads`` to raise ``JSONDecodeError``.

    This function escapes every ``\\`` that is *not* the start of a recognised
    JSON escape sequence (``\\"``, ``\\\\``, ``\\/``, ``\\b``, ``\\f``,
    ``\\n``, ``\\r``, ``\\t``, ``\\uXXXX``), turning it into ``\\\\``.

    Valid escape sequences and already-doubled backslashes are consumed
    *atomically* (as a 2-character unit), so normal JSON such as
    ``C:\\\\Users\\\\name`` is not corrupted.

    Note: ``\\b`` is a valid JSON escape (backspace U+0008).  A LaTeX sequence
    such as ``\\beta`` starts with ``\\b``, which this function leaves untouched
    because it cannot distinguish the two usages without full context.  Use
    LaTeX sequences that begin with letters other than ``b``, ``f``, ``n``,
    ``r``, or ``t`` to avoid this ambiguity.
    """
    return _ESCAPE_FIXUP_RE.sub(_fix_escape_match, text)


def parse_arguments_json(arguments_text: str) -> dict:
    """Parse *arguments_text* as JSON with a repair fallback for bare backslashes.

    Strategy:
    1. Try ``json.loads`` directly (fast path; valid JSON passes through unchanged).
    2. On ``JSONDecodeError``, apply :func:`repair_json_escapes` and retry.
    3. If still failing, raise a ``ValueError`` with a truncated preview and a
       suggestion to improve the prompt so the model emits valid JSON.
    """
    try:
        return json.loads(arguments_text)
    except json.JSONDecodeError:
        pass  # fall through to repair attempt

    repaired = repair_json_escapes(arguments_text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        preview = arguments_text[:200].replace("\n", " ")
        raise ValueError(
            f"Failed to parse arguments JSON even after escape repair.\n"
            f"JSONDecodeError: {exc}\n"
            f"Text preview (first 200 chars): {preview!r}\n"
            "Consider adjusting the prompt to ask the model for strictly valid JSON."
        ) from exc
