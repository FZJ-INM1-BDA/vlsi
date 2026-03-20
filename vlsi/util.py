import logging

SEP = b"\x00"


def encode_data(bufs: list[bytes]):
    """
    Encode a list of byte buffers into a single byte string.

    Parameters
    ----------
    bufs : list[bytes]
        List of byte buffers to encode.

    Returns
    -------
    bytes
        Encoded byte string with SEP separators between buffers and a trailing SEP.
    """
    return b"".join(encode_datum(b) for b in bufs)


def encode_datum(buf: bytes):
    assert SEP not in buf, "Cannot encode \\x00, as it is used as a separator"
    return buf + SEP


def decode_data(buf: bytes):
    """
    Decode byte buffer into list of data segments.

    Parameters
    ----------
    buf : bytes
        Byte buffer to decode, must end with SEP character.

    Returns
    -------
    list
        List of byte segments (b'...' objects) from the buffer.

    Raises
    ------
    AssertionError
        If buffer fails specification

    Examples
    --------
    >>> data = b'\\x00abc\\x00def\\x00'
    >>> decode_data(data)
    [b'', b'abc', b'def']
    """
    buf = buf.removesuffix(SEP)
    return buf.split(SEP)


logger = logging.getLogger(__name__)
