import hashlib
from typing import Any


def digest_string(obj: Any, length: int = 8) -> str:
    """
    Compute a short digest of a string.

    Parameters
    ----------
    obj : Any
        Input to digest. Will be converted to string.
    length : int, optional
        Length of the returned digest (default is 8).

    Returns
    -------
    digest : str
        Hexadecimal string digest of the input.

    Examples
    --------
    >>> digest_string("hello world")
    'b94d27b9'
    """
    s = str(obj)
    return hashlib.sha256(s.encode()).hexdigest()[:length]
