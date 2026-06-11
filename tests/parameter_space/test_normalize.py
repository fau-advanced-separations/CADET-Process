import numpy as np
import pytest
from CADETProcess.parameter_space.normalize import (
    AutoNormalizer,
    LinearNormalizer,
    LogNormalizer,
    NullNormalizer,
)


def test_null_normalizer_identity():
    norm = NullNormalizer(0, 100)
    assert norm.lb == 0
    assert norm.ub == 100
    assert norm.normalize(0) == 0
    assert norm.denormalize(0) == 0


@pytest.mark.parametrize("value", [-10, 1000])
def test_linear_normalize_out_of_bounds_raises(value):
    norm = LinearNormalizer(0, 100)
    with pytest.raises(ValueError):
        norm.normalize(value)


@pytest.mark.parametrize("value", [-1, 2])
def test_linear_denormalize_out_of_bounds_raises(value):
    norm = LinearNormalizer(0, 100)
    with pytest.raises(ValueError):
        norm.denormalize(value)


@pytest.mark.parametrize(
    "x, expected",
    [(0, 0.0), (10, 0.1), (100, 1.0)],
)
def test_linear_normalize(x, expected):
    norm = LinearNormalizer(0, 100)
    assert np.isclose(norm.normalize(x), expected)


@pytest.mark.parametrize(
    "x, expected",
    [(0.0, 0), (0.1, 10), (1.0, 100)],
)
def test_linear_denormalize(x, expected):
    norm = LinearNormalizer(0, 100)
    assert np.isclose(norm.denormalize(x), expected)


@pytest.mark.parametrize(
    "x, expected",
    [(1, 0.0), (10, 1 / 3), (100, 2 / 3), (1000, 1.0)],
)
def test_log_normalize(x, expected):
    norm = LogNormalizer(1, 1000)
    assert np.isclose(norm.normalize(x), expected)


@pytest.mark.parametrize(
    "x, expected",
    [(0.0, 1), (1 / 3, 10), (2 / 3, 100), (1.0, 1000)],
)
def test_log_denormalize(x, expected):
    norm = LogNormalizer(1, 1000)
    assert np.isclose(norm.denormalize(x), expected)


def test_log_normalizer_nonpositive_lb():
    norm = LogNormalizer(-5, 95)
    expected = np.log(11) / np.log(101)
    assert np.isclose(norm.normalize(5), expected)


@pytest.mark.parametrize(
    "lb, ub, threshold, expect_linear",
    [
        (1, 100, 1000, True),
        (1, 1001, 1000, False),
        (-5, 95, 1000, True),
        (-5, 999, 1000, False),
    ],
)
def test_auto_normalizer_selects_method(lb, ub, threshold, expect_linear):
    norm = AutoNormalizer(lb, ub, threshold=threshold)
    assert norm.use_linear == expect_linear
    assert norm.use_log != expect_linear


def test_linear_normalizer_roundtrip():
    norm = LinearNormalizer(2.0, 8.0)
    for x in [2.0, 4.0, 6.0, 8.0]:
        assert np.isclose(norm.denormalize(norm.normalize(x)), x)


def test_log_normalizer_roundtrip():
    norm = LogNormalizer(1.0, 1000.0)
    for x in [1.0, 10.0, 100.0, 1000.0]:
        assert np.isclose(norm.denormalize(norm.normalize(x)), x)
