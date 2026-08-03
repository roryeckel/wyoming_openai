"""Unit tests for the e2e helper utilities (run in the default suite)."""

import pytest
from e2e_helpers import fuzzy_match, fuzzy_text_match


def test_fuzzy_text_match_accepts_punctuation_and_case_differences():
    assert fuzzy_text_match("turn on the kitchen lights", "Turn on the kitchen lights.")


def test_fuzzy_text_match_rejects_short_substring_match():
    assert not fuzzy_text_match("lights", "Turn on the kitchen lights.")


def test_fuzzy_match_reports_score_and_coverage():
    result = fuzzy_match("turn on the kitchen lights", "Turn on the kitchen lights.")
    assert result.matched
    assert result.score == 100.0
    assert result.coverage == pytest.approx(1.0)
    assert result.normalized_expected == "turn on the kitchen lights"
    assert result.normalized_transcript == "turn on the kitchen lights"


def test_fuzzy_match_coverage_rejection_reports_details():
    result = fuzzy_match("lights", "Turn on the kitchen lights.")
    assert not result.matched
    assert result.coverage < 0.75
    # partial_ratio itself is perfect - coverage is what rejects it.
    assert result.score == 100.0


def test_fuzzy_match_tolerates_minor_mishearing():
    assert fuzzy_text_match(
        "the quick brown fox jumped over the lazy dog", "The quick brown fox jumps over the lazy dog."
    )


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("", "Turn on the kitchen lights."),
        ("turn on the kitchen lights", ""),
        ("", ""),
        ("...", "Turn on the kitchen lights."),  # normalizes to empty
    ],
)
def test_fuzzy_match_empty_inputs_never_match(transcript: str, expected: str):
    result = fuzzy_match(transcript, expected)
    assert not result.matched
    assert result.score == 0.0
    assert result.coverage == 0.0
