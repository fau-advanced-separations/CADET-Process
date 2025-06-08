import numpy as np


def round_to_significant_digits(
    values: np.ndarray | list[float],
    digits: int,
) -> np.ndarray:
    """
    Round an array of numbers to the specified number of significant digits.

    Parameters
    ----------
    values : np.ndarray | list[float]
        Input array of floats to be rounded. Can be a NumPy array or a list of floats.
    digits : int
        Number of significant digits to retain. Must be greater than 0.

    Returns
    -------
    np.ndarray
        Array of values rounded to the specified significant digits.
        The shape matches the input.

    Notes
    -----
    - The function handles zero values by returning 0 directly, avoiding issues
      with logarithms.
    - Input arrays are automatically converted to `ndarray` if not already.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([123.456, 0.001234, 98765, 0, -45.6789])
    >>> round_to_significant_digits(values, 3)
    array([ 123.   ,    0.00123, 98700.   ,    0.     ,  -45.7   ])

    >>> values = [1.2345e-5, 6.789e3, 0.0]
    >>> round_to_significant_digits(values, 2)
    array([ 1.2e-05,  6.8e+03,  0.0e+00])
    """
    if digits is None:
        return values

    input_type = type(values)

    values = np.asarray(values)  # Ensure input is a NumPy array

    if digits <= 0:
        raise ValueError("Number of significant digits must be greater than 0.")

    # Mask for non-zero values
    nonzero_mask = values != 0
    nan_mask = ~np.isnan(values)
    combined_mask = np.logical_and(nonzero_mask, nan_mask)
    result = np.zeros_like(values)  # Initialize result array

    # For non-zero elements, calculate the scaling and apply rounding
    if np.any(combined_mask):  # Check if there are any non-zero values
        nonzero_values = values[combined_mask]
        scales = digits - np.floor(np.log10(np.abs(nonzero_values))).astype(int) - 1

        # Round each non-zero value individually
        rounded_nonzero = [
            round(v, int(scale)) for v, scale in zip(nonzero_values, scales)
        ]

        result[combined_mask] = rounded_nonzero  # Assign the rounded values back

    result[~nan_mask] = np.nan
    if input_type is not np.ndarray:
        result = input_type(result)

    return result
