from collections import defaultdict
from typing import Any

import numpy as np


def assert_almost_equal_dict(
    dict_actual: Any,
    dict_expected: Any,
    decimal: int = 7,
    verbose: bool = True,
    path: str = "root",
) -> None:
    """Helper function to assert nested dict-like structures are (almost) equal."""
    if isinstance(dict_actual, defaultdict):
        dict_actual = dict(dict_actual)
    if isinstance(dict_expected, defaultdict):
        dict_expected = dict(dict_expected)

    if isinstance(dict_actual, dict) and isinstance(dict_expected, dict):
        assert dict_actual.keys() == dict_expected.keys(), (
            f"Key mismatch at {path}: {dict_actual.keys()} != {dict_expected.keys()}"
        )
        for k in dict_actual:
            assert_almost_equal_dict(
                dict_actual[k], dict_expected[k], decimal, verbose, path + f"->{k}"
            )

    elif isinstance(dict_actual, (list, tuple)) and isinstance(
        dict_expected, (list, tuple)
    ):
        assert len(dict_actual) == len(dict_expected), (
            f"Length mismatch at {path}: {len(dict_actual)} != {len(dict_expected)}"
        )
        for i, (a, e) in enumerate(zip(dict_actual, dict_expected)):
            assert_almost_equal_dict(a, e, decimal, verbose, path + f"[{i}]")

    elif isinstance(dict_actual, (float, int, np.ndarray, np.generic)) and isinstance(
        dict_expected, (float, int, np.ndarray, np.generic)
    ):
        try:
            np.testing.assert_almost_equal(dict_actual, dict_expected, decimal=decimal)
        except AssertionError as e:
            if verbose:
                raise AssertionError(f"{path}: {str(e)}") from e
            else:
                raise

    else:
        assert dict_actual == dict_expected, (
            f"Value mismatch at {path}: {dict_actual} != {dict_expected}"
        )
