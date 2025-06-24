import numpy as np
import pytest
from CADETProcess.normalize import (
    AutoNormalizer,
    LinearNormalizer,
    LogNormalizer,
    NullNormalizer,
)


@pytest.mark.parametrize("value", [-10, 1000])
def test_linear_normalization_input_range_raises(value):
    norm = LinearNormalizer(0, 100)
    with pytest.raises(ValueError):
        norm.normalize(value)


@pytest.mark.parametrize("value", [-1, 2])
def test_linear_denormalization_output_range_raises(value):
    norm = LinearNormalizer(0, 100)
    with pytest.raises(ValueError):
        norm.denormalize(value)


def test_null_normalizer_identity_behavior():
    norm = NullNormalizer(0, 100)
    assert norm.lb == 0
    assert norm.ub == 100
    assert norm.normalize(0) == 0
    assert norm.denormalize(0) == 0


@pytest.mark.parametrize(
    "input_value, expected_output", [
        (0, 0.0),
        (10, 0.1),
        (100, 1.0),
    ]
)
def test_linear_normalization(input_value, expected_output):
    norm = LinearNormalizer(0, 100)
    out = norm.normalize(input_value)
    assert np.isclose(out, expected_output)


@pytest.mark.parametrize(
    "norm_value, expected_input", [
        (0.0, 0),
        (0.1, 10),
        (1.0, 100),
    ]
)
def test_linear_denormalization(norm_value, expected_input):
    norm = LinearNormalizer(0, 100)
    out = norm.denormalize(norm_value)
    assert np.isclose(out, expected_input)


@pytest.mark.parametrize(
    "input_value, expected_output", [
        (1, 0.0),
        (10, 1 / 3),
        (100, 2 / 3),
        (1000, 1.0),
    ]
)
def test_log_normalization(input_value, expected_output):
    norm = LogNormalizer(1, 1000)
    out = norm.normalize(input_value)
    assert np.isclose(out, expected_output)


@pytest.mark.parametrize(
    "norm_value, expected_input", [
        (0.0, 1),
        (1 / 3, 10),
        (2 / 3, 100),
        (1.0, 1000),
    ]
)
def test_log_denormalization(norm_value, expected_input):
    norm = LogNormalizer(1, 1000)
    out = norm.denormalize(norm_value)
    assert np.isclose(out, expected_input)


def test_log_normalizer_handles_lb_input_leq_zero():
    norm = LogNormalizer(-5, 95)
    x = 5
    out = norm.normalize(x)
    expected = np.log(11) / np.log(101)
    assert np.isclose(out, expected)


@pytest.mark.parametrize(
    "lb, ub, threshold, expected_linear",
    [
        (1, 100, 1000, True),
        (1, 1001, 1000, False),
        (-5, 95, 1000, True),  # shifted range is < threshold
        (-5, 999, 1000, False),  # shifted range is > threshold
    ]
)
def test_auto_normalizer_behavior(lb, ub, threshold, expected_linear):
    norm = AutoNormalizer(lb, ub, threshold=threshold)
    assert norm.use_linear == expected_linear
    assert norm.use_log != expected_linear
