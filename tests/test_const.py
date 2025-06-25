import wyoming_openai.const as const


def test_version_is_string():
    assert isinstance(const.__version__, str)

def test_attribution_names_are_strings():
    assert isinstance(const.ATTRIBUTION_NAME_MODEL, str)
    assert isinstance(const.ATTRIBUTION_NAME_PROGRAM, str)
    assert isinstance(const.ATTRIBUTION_NAME_PROGRAM_STREAMING, str)
    assert isinstance(const.ATTRIBUTION_URL, str)
