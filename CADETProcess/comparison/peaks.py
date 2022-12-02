import numpy as np
import scipy.signal


def find_peaks(solution, normalize=True, prominence=0.5, find_minima=False):
    """Find peaks in solution.

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
    peaks = []
    if normalize and not solution.is_normalized:
        normalized = True
        solution.normalize()
    else:
        normalized = False

    for i in range(solution.component_system.n_comp):
        sol = solution.solution[:, i].copy()

        if find_minima:
            sol *= -1

        peak_indices, _ = scipy.signal.find_peaks(sol, prominence=prominence)
        if len(peak_indices) == 0:
            peak_indices = [np.argmax(sol)]
        time = solution.time[peak_indices]
        peak_heights = solution.solution[peak_indices, i]

        if normalized:
            peak_heights = solution.transform.untransform(peak_heights)

        peaks.append([(t, h) for t, h in zip(time, peak_heights)])

    if normalized:
        solution.denormalize()

    return peaks


def find_breakthroughs(solution, normalize=True, threshold=0.95):
    """Find breakthroughs in solution.

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
    breakthrough = []
    if normalize and not solution.is_normalized:
        normalized = True
        solution.normalize()
    else:
        normalized = False

    for i in range(solution.component_system.n_comp):
        sol = solution.solution[:, i].copy()

        breakthrough_indices = np.where(sol > threshold*np.max(sol))[0][0]
        if len(breakthrough_indices) == 0:
            breakthrough_indices = [np.argmax(sol)]
        time = solution.time[breakthrough_indices]
        breakthrough_height = solution.solution[breakthrough_indices, i]

        if solution.is_normalized:
            breakthrough_height = solution.transform.untransform(breakthrough_height)

        breakthrough.append((time, breakthrough_height))

    if normalized:
        solution.denormalize()

    return breakthrough
