import copy

import numpy as np
import scipy.signal

from CADETProcess.solution import SolutionIO


def find_peaks(
    solution: SolutionIO,
    normalize: bool = True,
    prominence: float = 0.5,
    find_minima: bool = False,
) -> list[list[tuple[float, float]]]:
    """
    Find peaks in solution.

    Parameters
    ----------
    solution : SolutionIO
        Solution object.
    normalize : bool, optional
        If true, normalize data to maximum value (for each component).
        The default is True.
    prominence : float, optional
        Required prominence to  detekt peak. The default is 0.5.
    find_minima : bool, optional
        Find negative peaks/minima of solution. The default is False.

    Returns
    -------
    peaks : list
        List with list of (time, height) for each peak for every component.
        Regardless of normalization, the actual peak height is returned.
    """
    solution_original = solution
    solution = copy.deepcopy(solution)

    if normalize:
        solution = solution.normalize()

    peaks = []
    for i in range(solution.component_system.n_comp):
        sol = solution.solution[:, i].copy()

        if find_minima:
            sol *= -1

        peak_indices, _ = scipy.signal.find_peaks(sol, prominence=prominence)
        if len(peak_indices) == 0:
            peak_indices = [np.argmax(sol)]
        time = solution.time[peak_indices]
        peak_heights = solution_original.solution[peak_indices, i]

        peaks.append([(t, h) for t, h in zip(time, peak_heights)])

    return peaks


def find_breakthroughs(
    solution: SolutionIO,
    normalize: bool = True,
    threshold: float = 0.95,
) -> list[(float, float)]:
    """
    Find breakthroughs in solution.

    Parameters
    ----------
    solution : SolutionIO
        Solution object.
    normalize : bool, optional
        If true, normalize data to maximum value (for each component).
        The default is True.
    threshold : float, optional
        Percentage of maximum concentration that needs to be reached to be
        considered as breakthrough. The default is 0.95.

    Returns
    -------
    breakthrough : list
        List with (time, height) for breakthrough of every component.
        Regardless of normalization, the actual breakthroug height is returned.
    """
    solution_original = solution
    solution = copy.deepcopy(solution)

    if normalize:
        solution = solution.normalize()

    breakthrough = []
    for i in range(solution.component_system.n_comp):
        sol = solution.solution[:, i].copy()

        breakthrough_indices = np.where(sol > threshold * np.max(sol))[0][0]
        if len(breakthrough_indices) == 0:
            breakthrough_indices = [np.argmax(sol)]
        time = solution.time[breakthrough_indices]
        breakthrough_height = solution_original.solution[breakthrough_indices, i]

        breakthrough.append((time, breakthrough_height))

    return breakthrough
