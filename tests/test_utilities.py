from io import BytesIO

from wyoming_openai.utilities import NamedBytesIO


def test_named_bytes_io_name_property():
    buf = NamedBytesIO(b"abc", name="test.wav")
    assert buf.name == "test.wav"
    assert buf.read() == b"abc"

def test_named_bytes_io_default_name():
    buf = NamedBytesIO()
    assert buf.name == "audio.wav"

def test_named_bytes_io_inherits_bytesio():
    buf = NamedBytesIO(b"xyz", name="foo.wav")
    assert isinstance(buf, BytesIO)
    assert buf.read() == b"xyz"
