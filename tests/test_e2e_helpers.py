from e2e_helpers import fuzzy_text_match


def test_fuzzy_text_match_accepts_punctuation_and_case_differences():
    assert fuzzy_text_match("turn on the kitchen lights", "Turn on the kitchen lights.")


def test_fuzzy_text_match_rejects_short_substring_match():
    assert not fuzzy_text_match("lights", "Turn on the kitchen lights.")
