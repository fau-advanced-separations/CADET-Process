import warnings

import numpy as np

from CADETProcess.reference import ReferenceIO


def crop(
    reference: ReferenceIO,
    start: float,
    end: float,
) -> ReferenceIO:
    """
    Extract a time window from a ReferenceIO and re-zero the time axis.

    Use this to align an experimental run with a simulation: the experiment
    may include pre-injection startup or post-run wash steps that are not
    part of the simulated process.
    Time is shifted so that ``start`` maps to $t = 0$, regardless of whether
    a sample falls exactly on ``start``.

    Parameters
    ----------
    reference : ReferenceIO
        Input signal covering the full experimental run.
    start : float
        Start of the window in seconds (inclusive).
    end : float
        End of the window in seconds (inclusive).

    Returns
    -------
    ReferenceIO
        New ReferenceIO containing only the points in ``[start, end]``,
        with the time axis shifted so that ``start`` maps to $t = 0$.
    """
    time = reference.time
    mask = (time >= start) & (time <= end)
    cropped_time = time[mask] - start
    cropped_solution = reference.solution[mask]
    return ReferenceIO(
        reference.name + "_cropped",
        cropped_time,
        cropped_solution,
        reference.flow_rate,
    )


def fit_baseline(
    time: np.ndarray,
    signal: np.ndarray,
    start: float | None = None,
    end: float | None = None,
    threshold: float = 0.025,
) -> np.ndarray:
    """
    Estimate a linear baseline from the lowest-intensity points in a time window.

    Fits a line to all points whose normalized intensity falls below `threshold`,
    then evaluates that line over the full `time` array.

    Parameters
    ----------
    time : ndarray
        1D array of time points.
    signal : ndarray
        1D array of signal values.
    start : float, optional
        Start of the window used for baseline fitting. Defaults to ``time[0]``.
    end : float, optional
        End of the window. Defaults to ``time[-1]``.
    threshold : float
        Normalized intensity cutoff (0–1) for selecting baseline points. Default 0.025.

    Returns
    -------
    ndarray
        Linear baseline evaluated over the entire `time` array.

    Raises
    ------
    ValueError
        If the window contains no points or fewer than two baseline points.
    """
    if start is None:
        start = time[0]
    if end is None:
        end = time[-1]

    mask = (time >= start) & (time <= end)
    t_win = time[mask]
    s_win = signal[mask]

    if len(t_win) == 0:
        raise ValueError("No points in the baseline fitting window.")

    s_min = np.min(s_win)
    s_max = np.max(s_win)

    if s_max == s_min:
        return np.full_like(signal, s_min, dtype=float)

    normalized = (s_win - s_min) / (s_max - s_min)
    baseline_idx = np.where(normalized < threshold)[0]

    if len(baseline_idx) < 2:
        raise ValueError("Fewer than two baseline points found under threshold.")

    coeffs = np.polyfit(t_win[baseline_idx], s_win[baseline_idx], 1)
    return np.polyval(coeffs, time)


def correct_baseline(
    reference: ReferenceIO,
    start: float | None = None,
    end: float | None = None,
    threshold: float = 0.025,
) -> ReferenceIO:
    """
    Subtract a linear baseline from a ReferenceIO signal.

    The baseline is estimated from the lowest-intensity points within the
    optional fitting window ``[start, end]``.
    The correction is applied to the full signal; no points are zeroed or
    removed.
    To restrict the signal to a simulation window, apply :func:`crop` first.

    Small negative values may appear after subtraction due to noise or baseline
    uncertainty; these are not clipped automatically.

    Parameters
    ----------
    reference : ReferenceIO
        Input signal.
    start : float, optional
        Start of the baseline fitting window in seconds. Defaults to the first time point.
    end : float, optional
        End of the baseline fitting window in seconds. Defaults to the last time point.
    threshold : float
        Normalized intensity cutoff for baseline point selection. Default 0.025.

    Returns
    -------
    ReferenceIO
        New ReferenceIO with the baseline subtracted across the full time axis.
    """
    time = reference.time
    if start is None:
        start = time[0]
    if end is None:
        end = time[-1]

    corrected = np.empty_like(reference.solution, dtype=float)
    for i in range(reference.solution.shape[1]):
        col = reference.solution[:, i]
        baseline = fit_baseline(time, col, start, end, threshold)
        corrected[:, i] = col - baseline

    return ReferenceIO(
        reference.name + "_baseline_corrected",
        time,
        corrected,
        reference.flow_rate,
    )


def normalize_area(
    reference: ReferenceIO,
    target_area: float,
    start: float | None = None,
    end: float | None = None,
) -> ReferenceIO:
    """
    Rescale a ReferenceIO signal so its flow-weighted time integral matches `target_area`.

    Parameters
    ----------
    reference : ReferenceIO
        Input signal.
    target_area : float
        Known injected amount (e.g. mol or kg), matched by the flow-weighted
        integral of the rescaled signal.
    start : float, optional
        Start of the integration range in seconds. Defaults to the first time point.
    end : float, optional
        End of the integration range in seconds. Defaults to the last time point.

    Returns
    -------
    ReferenceIO
        New ReferenceIO with rescaled signal.

    Raises
    ------
    ValueError
        If the integrated area over ``[start, end]`` is zero.
    """
    current_area = reference.fraction_mass(start, end)
    if np.isclose(current_area, 0.0):
        raise ValueError(
            "Integrated area is zero or near-zero; cannot normalize. "
            "Check that the integration window contains the peak."
        )
    scale = target_area / current_area
    return ReferenceIO(
        reference.name + "_normalized",
        reference.time,
        reference.solution * scale,
        reference.flow_rate,
    )


def correct_baseline_and_normalize(
    reference: ReferenceIO,
    target_area: float,
    start_baseline: float | None = None,
    end_baseline: float | None = None,
    threshold: float = 0.025,
    start_normalization: float | None = None,
    end_normalization: float | None = None,
) -> ReferenceIO:
    """
    Correct the baseline then normalize the peak area.

    Convenience wrapper combining :func:`correct_baseline` and
    :func:`normalize_area`.

    Parameters
    ----------
    reference : ReferenceIO
        Input signal.
    target_area : float
        Known injected amount matched by the flow-weighted integral after normalization.
    start_baseline : float, optional
        Start of the baseline fitting window.
    end_baseline : float, optional
        End of the baseline fitting window.
    threshold : float
        Normalized intensity cutoff for baseline point selection. Default 0.025.
    start_normalization : float, optional
        Start of the integration range for normalization.
    end_normalization : float, optional
        End of the integration range for normalization.

    Returns
    -------
    ReferenceIO
        New ReferenceIO with baseline corrected and area normalized.
    """
    reference = correct_baseline(reference, start_baseline, end_baseline, threshold)
    return normalize_area(reference, target_area, start_normalization, end_normalization)


def fit_polynomial(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
) -> tuple[list[float], float]:
    """
    Fit a polynomial calibration curve to (signal, concentration) data.

    Parameters
    ----------
    x_data : array-like
        Independent variable (e.g. raw signal values).
    y_data : array-like
        Dependent variable (e.g. known concentrations).
    degree : int
        Degree of the polynomial.

    Returns
    -------
    coefficients : list[float]
        Fitted polynomial coefficients in descending degree order.
    r_squared : float
        Coefficient of determination.
    """
    coefficients = np.polyfit(x_data, y_data, degree)
    y_fit = np.polyval(coefficients, x_data)
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return coefficients.tolist(), r_squared


def apply_polynomial_calibration(
    reference: ReferenceIO,
    coefficients: list[float] | np.ndarray,
) -> ReferenceIO:
    """
    Apply a polynomial calibration curve to a ReferenceIO signal.

    Parameters
    ----------
    reference : ReferenceIO
        Input signal in raw units.
    coefficients : list[float] or ndarray
        Polynomial coefficients in descending degree order, as returned by
        :func:`fit_polynomial`.

    Returns
    -------
    ReferenceIO
        New ReferenceIO with signal converted to calibrated units.
    """
    return ReferenceIO(
        reference.name + "_calibrated",
        reference.time,
        np.polyval(coefficients, reference.solution),
        reference.flow_rate,
    )


def apply_beer_lambert(
    reference: ReferenceIO,
    extinction_coefficient: float,
    path_length: float,
) -> ReferenceIO:
    r"""
    Convert an absorbance signal to concentration via Beer-Lambert law.

    Applies $c = A / (\\varepsilon \\cdot l)$ pointwise, where $\\varepsilon$ is
    the molar extinction coefficient and $l$ is the optical path length.

    Small negative concentrations may appear due to noise or baseline
    uncertainty and are not clipped automatically.

    Parameters
    ----------
    reference : ReferenceIO
        Single-component absorbance signal (e.g. mAU from a UV detector).
    extinction_coefficient : float
        Molar extinction coefficient $\\varepsilon$ in L/(mol·cm).
    path_length : float
        Optical path length $l$ in cm.

    Returns
    -------
    ReferenceIO
        New ReferenceIO with signal in mol/L.
    """
    return ReferenceIO(
        reference.name + "_concentration",
        reference.time,
        reference.solution / (extinction_coefficient * path_length),
        reference.flow_rate,
    )


def deconvolve_extinction(
    references: list[ReferenceIO],
    extinction_matrix: np.ndarray,
    path_length: float,
    component_names: list[str] | None = None,
) -> list[ReferenceIO]:
    r"""
    Recover per-component concentration profiles from multi-wavelength absorbance data.

    Solves $\\mathbf{A}(t) = l \\cdot E \\, \\mathbf{c}(t)$ in the least-squares sense,
    where $E \\in \\mathbb{R}^{n_\\lambda \\times n_\\text{comp}}$ is the extinction
    coefficient matrix.
    The system is overdetermined when $n_\\lambda > n_\\text{comp}$ and exactly
    determined when $n_\\lambda = n_\\text{comp}$.

    Small negative concentrations may appear due to noise or baseline uncertainty
    and are not clipped automatically.

    Parameters
    ----------
    references : list[ReferenceIO]
        One single-channel ReferenceIO per wavelength, all sharing the same time axis.
    extinction_matrix : ndarray, shape (n_wavelengths, n_components)
        Molar extinction coefficients $\\varepsilon_{i,j}$ in L/(mol·cm), where
        rows correspond to wavelengths and columns to components.
    path_length : float
        Optical path length in cm, applied uniformly to all channels.
    component_names : list[str], optional
        Names for the returned concentration references.
        Defaults to ``"component_0"``, ``"component_1"``, …

    Returns
    -------
    list[ReferenceIO]
        One ReferenceIO per component, in mol/L, in the same order as the columns
        of `extinction_matrix`.

    Raises
    ------
    ValueError
        If the number of references does not match the number of rows in
        `extinction_matrix`, if any reference contains more than one signal channel,
        if `component_names` length does not match the number of components,
        if the references do not share the same time axis, or if the extinction
        matrix is rank-deficient.
    """
    E = np.asarray(extinction_matrix, dtype=float)
    n_wavelengths, n_components = E.shape

    if len(references) != n_wavelengths:
        raise ValueError(
            f"Got {len(references)} references but extinction_matrix has "
            f"{n_wavelengths} rows."
        )

    for ref in references:
        if ref.solution.shape[1] != 1:
            raise ValueError(
                "Each reference must contain exactly one signal channel; "
                f"got shape {ref.solution.shape} for '{ref.name}'."
            )

    time = references[0].time
    for ref in references[1:]:
        if not np.array_equal(ref.time, time):
            raise ValueError("All references must share the same time axis.")

    if component_names is not None and len(component_names) != n_components:
        raise ValueError(
            f"component_names has {len(component_names)} entries but "
            f"extinction_matrix has {n_components} columns."
        )

    rank = np.linalg.matrix_rank(E)
    if rank < n_components:
        raise ValueError(
            f"Extinction matrix is rank deficient (rank {rank} < {n_components}); "
            "components cannot be uniquely resolved."
        )

    cond = np.linalg.cond(E)
    if cond > 1e10:
        warnings.warn(
            f"Extinction matrix is poorly conditioned (condition number {cond:.2e}); "
            "recovered concentrations may be inaccurate.",
            stacklevel=2,
        )

    # A_mat: shape (n_time, n_wavelengths)
    A_mat = np.hstack([ref.solution for ref in references])

    # Solve E @ c(t) = A(t) / path_length for all t simultaneously
    C, _, _, _ = np.linalg.lstsq(E * path_length, A_mat.T, rcond=None)
    # C: shape (n_components, n_time) → transpose to (n_time, n_components)
    C = C.T

    if component_names is None:
        component_names = [f"component_{i}" for i in range(n_components)]

    flow_rate = references[0].flow_rate
    return [
        ReferenceIO(name, time, C[:, i : i + 1], flow_rate)
        for i, name in enumerate(component_names)
    ]
