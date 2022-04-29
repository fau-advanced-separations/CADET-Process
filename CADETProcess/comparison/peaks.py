import numpy as np
import scipy.signal


def find_peaks(solution, normalize=True, height=0.01, find_minima=False):
    """Find peaks in solution.

    Parameters
    ----------
    solution : SolutionIO
        Solution object.
    normalize : bool, optional
        If true, normalize data to maximum value (for each component).
        The default is True.
    height : float, optional
        Required height of peaks. The default is 0.01.
    find_minima : bool, optional
        Invert solution and find minima. The default is False.

    Returns
    -------
    peaks : list
        List with list of (time, height) for each peak for every component.

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

        peak_indices, _ = scipy.signal.find_peaks(sol, height=height)
        if len(peak_indices) == 0:
            peak_indices = [np.argmax(sol)]
        time = solution.time[peak_indices]
        peak_heights = solution.solution[peak_indices, i]

        if solution.is_normalized:
            peak_heights = solution.transform.untransform(peak_heights)

        peaks.append([(t, h) for t, h in zip(time, peak_heights)])

    if normalized:
        solution.denormalize()

    return peaks


def find_breakthroughs(solution, threshold=0.95):
    """Find breakthroughs in solution.

    Parameters
    ----------
    solution : SolutionIO
        Solution object.
    threshold : float, optional
        Percentage of maximum concentration that needs to be reached to be
        considered as breakthrough. The default is 0.95.

    Returns
    -------
    breakthrough : list
        List with (time, height) for breakthrough of every component.

    """
    breakthrough = []

    for i in range(solution.component_system.n_comp):
        sol = solution.solution[:, i].copy()

        breakthrough_indices = np.where(sol > threshold*np.max(sol))[0][0]
        time = solution.time[breakthrough_indices]
        breakthrough_height = solution.solution[breakthrough_indices, i]
        breakthrough.append((time, breakthrough_height))

    return breakthrough
