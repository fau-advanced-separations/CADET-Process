import warnings

import numba
import numpy as np
import numpy.typing as npt
import scipy
from scipy.interpolate import PchipInterpolator

from CADETProcess import CADETProcessError


def linear_coeff(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return paramters that fit y = a*x+b."""
    a = (y2 - y1) / (x2 - x1)
    b = y2 - a * x2
    return a, b


def exponential_coeff(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Return paramaters that fit y = b*exp(m*x)."""
    b = (np.log(y2) - np.log(y1)) / (x2 - x1)
    a = y1 * np.exp(-b * x1)
    return a, b


def exponential(x: float, a: float, b: float) -> float:
    """Evaluate exponential function."""
    return a * np.exp(b * x)


def linear(x: float, a: float, b: float) -> float:
    """Evaluate linear function."""
    return a * x + b


def find_opt_poly(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find optimal overlap between x (offset) and y (pearson).

    Given a curve, find highest point.
    """
    if index == 0:
        indices = np.array([index, index + 1, index + 2])
    elif index == (len(x) - 1):
        indices = np.array([index - 2, index - 1, index])
    else:
        indices = np.array([index - 1, index, index + 1])

    x = x[indices]
    y = y[indices]

    if x[0] > x[-1]:  # need to invert order
        x = x[::-1]
        y = y[::-1]

    try:
        poly, res = np.polynomial.Polynomial.fit(x, y, 2, full=True)
    except np.linalg.LinAlgError:
        raise CADETProcessError(f"Polyfit failed for {x}, {y}")
    try:
        root = poly.deriv().roots()[0]
    except IndexError:
        # If all y values are equal, use center value.
        root = x[0]
    root = np.clip(root, x[0], x[1])
    return root, x, y


def pear_corr(cr: float) -> float:
    """Flip pearson correlation s.t. 0 is best and 1 is worst."""
    # handle the case where a nan is returned
    if np.isnan(cr):
        return 1.0
    if cr < 0.0:
        return 1.0
    else:
        return 1 - cr


@numba.njit(fastmath=True)
def pearsonr_mat(x: np.ndarray, Y: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    High performance implementation of the pearson correlation.

    This is to simultaneously evaluate the pearson correlation between a vector and a
    matrix. Scipy can only evaluate vector/vector.
    """
    r = np.zeros(Y.shape[0])
    xm = x - x.mean()

    r_x_den = np.linalg.norm(xm)

    for i in range(Y.shape[0]):
        ym = Y[i] - np.mean(Y[i])

        r_num = np.dot(xm, ym)
        r_y_den = np.linalg.norm(ym)

        denominator = r_x_den * r_y_den

        if denominator == 0.0:
            r[i] = -1.0
        else:
            min_fun = 0
            for j in range(x.shape[0]):
                min_fun += min(x[j], Y[i, j])

            r[i] = min(max(r_num / denominator, -1.0), 1.0) * min_fun
    return r


def pearson_offset(
    time: npt.ArrayLike,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
    offset: float,
) -> float:
    """Calculate single pearson correlation at offset."""
    simulation_data_offset = simulation_spline(time - offset)
    reference_data = reference_spline(time)
    try:
        pear = scipy.stats.pearsonr(reference_data, simulation_data_offset)[0]
    except ValueError:
        warnings.warn(
            f"Pearson correlation failed to do NaN or InF in array exp_array: "
            f"{list(reference_data)}, sim_array: {list(simulation_data_offset)}"
        )
        pear = 0
    score_local = pear_corr(pear)

    return score_local


def eval_offsets(
    time: float,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
    offsets: np.ndarray,
) -> np.ndarray:
    """Calculate pearson correlation for each offset."""
    rol_mat = np.zeros([len(offsets), len(time)])

    for idx, offset in enumerate(offsets):
        rol_mat[idx, :] = simulation_spline(time - offset)

    reference_data = reference_spline(time)
    scores = pearsonr_mat(reference_data, rol_mat, time)
    return scores


def pearson(
    time: np.ndarray,
    reference_spline: PchipInterpolator,
    simulation_spline: PchipInterpolator,
    size: int = 20,
    nest: int = 50,
    bounds: int = 2,
    tol: float = 1e-13,
) -> tuple[float, float]:
    """
    Find highest correlation between reference and simulation.

    The two signals are shifted in time to find the time offset which
    corresponds to highest correlation. To ensure deterministic results, a
    a bisection scheme is used to refine the offset instead of an optimizer.

    Parameters
    ----------
    time: np.ndarray
        Time points.
    reference_spline : PchipInterpolator
        Reference data points.
    simulation_spline : PchipInterpolator
        Simulation Data points.
    size : int, optional
        Number of points to be generated between upper and lower bounds.
        The default is 20.
    nest : int, optional
        Maximum number of recursions. The default is 50.
    bounds : int, optional
        Number of indices allowed for shifting signal outside bounds.
        The default is 2.
    tol : float, optional
        Tolerance for difference between lb and ub. The default is 1e-13.

    Returns
    -------
    score_local : float
        pearson correlation (inverted).
    dt : float
        time offset.
    """
    for i in range(nest + 1):
        if i == 0:
            lb = -time[-1]
            ub = time[-1]
            local_size = min(100 + 1, int((ub - lb) * 2 + 1))
        else:
            idx_max = np.argmax(pearson)

            try:
                lb = offsets[idx_max - bounds]  # noqa: F821
            except IndexError:
                lb = offsets[0]  # noqa: F821

            try:
                ub = offsets[idx_max + bounds]  # noqa: F821
            except IndexError:
                ub = offsets[-1]  # noqa: F821
            local_size = size

        if ub - lb < tol:
            break

        offsets = np.linspace(lb, ub, local_size)

        pearson = eval_offsets(time, reference_spline, simulation_spline, offsets)

        idx_max = np.argmax(pearson)

        expand_lb = max(bounds - idx_max, 0)
        expand_ub = max(bounds - (len(pearson) - 1 - idx_max), 0)

        if expand_lb or expand_ub:
            expand_lb = expand_lb * 2
            expand_ub = expand_ub * 2
            dt = offsets[1] - offsets[0]
            if expand_lb:
                local_offsets = np.linspace(offsets[0] - expand_lb * dt, offsets[0] - dt, expand_lb)
                local_pearson = eval_offsets(
                    time, reference_spline, simulation_spline, local_offsets
                )

                offsets = np.concatenate([local_offsets, offsets])
                pearson = np.concatenate([local_pearson, pearson])

            if expand_ub:
                local_offsets = np.linspace(
                    offsets[-1] + dt, offsets[-1] + expand_ub * dt, expand_ub
                )

                local_pearson = eval_offsets(
                    time, reference_spline, simulation_spline, local_offsets
                )

                offsets = np.concatenate([offsets, local_offsets])
                pearson = np.concatenate([pearson, local_pearson])

    idx = np.argmax(pearson)

    dt, time_found, goal_found = find_opt_poly(offsets, pearson, idx)

    # calculate pearson correlation at the new time
    score_local = pearson_offset(time, reference_spline, simulation_spline, dt)

    return score_local, dt
