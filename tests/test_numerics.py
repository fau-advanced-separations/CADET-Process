import numpy as np
import pytest
from CADETProcess.numerics import round_to_significant_digits


def test_basic_functionality():
    values = np.array([123.456, 0.001234, 98765])
    expected = np.array([123.0, 0.00123, 98800.0])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_handling_zeros():
    values = np.array([0, 0.0, -0.0])
    expected = np.array([0.0, 0.0, -0.0])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_array_equal(result, expected)


def test_negative_numbers():
    values = np.array([-123.456, -0.001234, -98765])
    expected = np.array([-123.0, -0.00123, -98800.0])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_large_numbers():
    values = np.array([1.23e10, 9.87e15])
    expected = np.array([1.23e10, 9.87e15])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_small_numbers():
    values = np.array([1.23e-10, 9.87e-15])
    expected = np.array([1.23e-10, 9.87e-15])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_mixed_values():
    values = np.array([123.456, 0, -0.001234, 9.8765e-5])
    expected = np.array([123.0, 0.0, -0.00123, 9.88e-5])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_invalid_digits():
    with pytest.raises(
        ValueError, match="Number of significant digits must be greater than 0."
    ):
        round_to_significant_digits(np.array([123.456]), 0)


def test_empty_array():
    values = np.array([])
    expected = np.array([])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_array_equal(result, expected)


def test_non_array_input():
    values = [123.456, 0.001234, 98765]  # List input
    expected = np.array([123.0, 0.00123, 98800.0])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_none_digits():
    values = [123.456, 0.001234, 98765]  # List input
    expected = np.array([123.456, 0.001234, 98765])
    result = round_to_significant_digits(values, None)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_nan_digits():
    values = np.array([123.456, np.nan, 98765])
    expected = np.array([123.0, np.nan, 98800.0])
    result = round_to_significant_digits(values, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
