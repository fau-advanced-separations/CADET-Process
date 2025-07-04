import warnings
from typing import Optional

import numpy as np
import scipy
import scipy.optimize as optimize
from scipy.interpolate import PchipInterpolator


def pearson(
    time: np.ndarray,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
    offset: Optional[float] = 0,
) -> float:
    """
    Calculate Pearson correlation.

    Optionally, an offset can be specified that shifts the signal in time. This can be
    a helpful metric in parameter estimation.

    Parameters
    ----------
    time: np.ndarray
        The time array
    reference_spline: PchipInterpolator
        The reference data.
    simulation_spline: PchipInterpolator
        The simulated data.
    offset: float, optional
        A time offset to be applied to the simulation spline.

    Returns
    -------
    float
        The pearson correlation given the current offset.
    """
    shifted_time = time - offset

    # restrict to the valid domain of reference_spline
    t_min, t_max = time[0], time[-1]
    valid_mask = (shifted_time >= t_min) & (shifted_time <= t_max)

    time_eval = time[valid_mask]
    shifted_eval = shifted_time[valid_mask]

    ref_vals = reference_spline(time_eval)
    sim_vals = simulation_spline(shifted_eval)

    try:
        pear = scipy.stats.pearsonr(ref_vals, sim_vals)[0]
    except ValueError:
        warnings.warn(
            f"Pearson correlation failed due to NaN or Inf in array reference: "
            f"{reference_spline.x}, simulation: {simulation_spline.x}"
        )
        pear = -1

    return pear


def flip_pearson(pear: float) -> float:
    """Flip value s.t. 0 is best and 1 is worst."""
    return 0.5 * (1 - pear)


def shape(
    time: np.ndarray,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
    offset: Optional[float] = 0,
    flip: Optional[bool] = True,
) -> float:
    """
    Calculate shape metric.

    Parameters
    ----------
    time: np.ndarray
        The time array
    reference_spline: PchipInterpolator
        The reference data.
    simulation_spline: PchipInterpolator
        The simulated data.
    offset: float, optional
        A time offset to be applied to the simulation spline.
    flip: bool, optional
        If True, flip value s.t. 0 is best and 1 is worst.
        The default is True.

    Returns
    -------
    float
        The pearson correlation given the current offset.
    """
    pear = pearson(time, reference_spline, simulation_spline, offset)
    if flip:
        pear = flip_pearson(pear)

    return pear


def determine_optimal_offset(
    time: np.ndarray,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
) -> tuple[float, float]:
    """
    Determine the optimal time offset s.t. Pearson correlation is maximixed.

    Uses scipy.optimize.minimize to find the optimal time offset.

    Parameters
    ----------
    time: np.ndarray
        The time array
    reference_spline: PchipInterpolator
        The reference data.
    simulation_spline: PchipInterpolator
        The simulated data.

    Returns
    -------
    tuple[float, float]
        The maximum pearson correlation and the corresponding time offset.
    """
    x0 = 0
    window = 0.05 * (time[-1] - time[0])
    bounds = (-time[-1] + window, time[-1] - window)  # Limit offset to avoid issues

    # Brute force for screening
    _, _, offsets, scores = optimize.brute(
        lambda x: shape(time, reference_spline, simulation_spline, x, True),
        [bounds],
        Ns=101,
        full_output=True,
        finish=None,
    )
    ind = np.nanargmin(scores)
    f = scores[ind]
    x = offsets[ind]

    # Update for refinement
    x0 = x
    window = 0.01 * (time[-1] - time[0])
    bounds = (max(bounds[0], x0 - window), min(bounds[1], x0 + window))

    # Refinement
    result = optimize.minimize(
        lambda x: shape(time, reference_spline, simulation_spline, x, True),
        x0=x0,
        bounds=[bounds],
        method="Powell",
        options={"xtol": 1e-12},
    )

    f = result.fun
    x = result.x[0]

    return f, x
