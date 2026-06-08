"""Synthetic SolutionIO builders for use across the test suite."""

import numpy as np
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionIO


def rectangle_chromatogram(component_system: ComponentSystem, windows, name="outlet"):
    """Rectangular pulse chromatogram.

    Parameters
    ----------
    component_system : ComponentSystem
    windows : list of list of (start, end), one inner list per component
        e.g. [[(2, 4), (7, 9)]] for one component with two peaks,
        or [[(2, 4)], [(7, 9)]] for two separated components.
    name : str, optional
        Name of the SolutionIO object. Default is "outlet".
    """
    n_time = 1001
    time = np.linspace(0, 10, n_time)
    solution = np.zeros((n_time, component_system.n_comp))
    for i, comp_windows in enumerate(windows):
        for t0, t1 in comp_windows:
            solution[(time >= t0) & (time < t1), i] = 1.0
    return SolutionIO(name, component_system, time, solution, np.ones(n_time))


def gaussian_chromatogram(component_system: ComponentSystem, peaks, name="outlet"):
    """Gaussian pulse chromatogram.

    Parameters
    ----------
    component_system : ComponentSystem
    peaks : list of (mu, sigma) per component
    name : str, optional
        Name of the SolutionIO object. Default is "outlet".
    """
    n_time = 1001
    time = np.linspace(0, 10, n_time)
    solution = np.zeros((n_time, component_system.n_comp))
    for i, (mu, sigma) in enumerate(peaks):
        solution[:, i] = np.exp(-0.5 * ((time - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return SolutionIO(name, component_system, time, solution, np.ones(n_time))
