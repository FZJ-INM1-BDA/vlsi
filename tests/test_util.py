import pytest
from vlsi.util import encode_data, encode_datum, decode_data


# ============================================================================
# encode_data Tests
# ============================================================================


@pytest.mark.parametrize(
    "bufs,expected",
    [
        ([b"abc", b"def", b"ghi"], b"abc\x00def\x00ghi\x00"),
        ([b"hello"], b"hello\x00"),
        ([], b""),
        ([b"", b"abc", b""], b"\x00abc\x00\x00"),
        ([b"\x01\x02\x03", b"\xff\xfe"], b"\x01\x02\x03\x00\xff\xfe\x00"),
    ],
    ids=[
        "multiple_buffers",
        "single_buffer",
        "empty_list",
        "empty_buffers",
        "binary_data",
    ],
)
def test_encode_data(bufs, expected):
    result = encode_data(bufs)
    assert result == expected


# ============================================================================
# encode_datum Tests
# ============================================================================


@pytest.mark.parametrize(
    "buf,expected",
    [
        (b"hello", b"hello\x00"),
        (b"", b"\x00"),
    ],
    ids=["simple_buffer", "empty_buffer"],
)
def test_encode_datum(buf, expected):
    result = encode_datum(buf)
    assert result == expected


@pytest.mark.parametrize("buf", [b"\x00\x01\x02"], ids=["contain_illegal_char"])
def test_expect_fail(buf):
    with pytest.raises(AssertionError):
        encode_datum(buf)


# ============================================================================
# decode_data Tests
# ============================================================================


@pytest.mark.parametrize(
    "buf,expected",
    [
        (b"abc\x00def\x00ghi\x00", [b"abc", b"def", b"ghi"]),
        (b"hello\x00", [b"hello"]),
        (b"\x00abc\x00\x00", [b"", b"abc", b""]),
        (b"\x00", [b""]),
        (b"\x01\x02\x03\x00\xff\xfe\x00", [b"\x01\x02\x03", b"\xff\xfe"]),
    ],
    ids=[
        "multiple_segments",
        "single_segment",
        "empty_segments",
        "empty_data",
        "binary_data",
    ],
)
def test_decode_data(buf, expected):
    result = decode_data(buf)
    assert result == expected


# ============================================================================
# Round-Trip Tests
# ============================================================================


@pytest.mark.parametrize(
    "data",
    [
        [b"abc", b"def", b"ghi"],
        [b"", b"", b""],
        [b"hello", b"world"],
        [b""],
    ],
    ids=[
        "multiple_buffers",
        "empty_buffers",
        "text_data",
        "single_empty",
    ],
)
def test_encode_decode_round_trip(data):
    encoded = encode_data(data)
    decoded = decode_data(encoded)
    assert decoded == data


@pytest.mark.parametrize(
    "data",
    [
        [b"\x00\x01\x02", b"\xff\xfe\xfd"],
    ],
    ids=["illegal_characters"],
)
def test_encode_decode_round_trip_fail(data):
    with pytest.raises(AssertionError):
        encode_data(data)


@pytest.mark.parametrize(
    "datum",
    [
        b"hello",
        b"",
        b"\xff\xfe",
    ],
    ids=["text", "empty", "high_bytes"],
)
def test_encode_datum_decode_round_trip(datum):
    encoded = encode_datum(datum)
    decoded = decode_data(encoded)
    assert decoded == [datum]


@pytest.mark.parametrize(
    "datum",
    [
        b"\x00\x01\x02",
    ],
    ids=["illegal_char"],
)
def test_encode_datum_decode_round_trip_fail(datum):
    with pytest.raises(AssertionError):
        encode_datum(datum)
