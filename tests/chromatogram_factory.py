"""Synthetic SolutionIO builders for use across the test suite."""

import numpy as np
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionIO


def rectangle_chromatogram(
    component_system: ComponentSystem,
    windows,
    heights=None,
    t_start=0.0,
    t_end=10.0,
    n_time=1001,
    name="outlet",
):
    """Rectangular pulse chromatogram.

    Parameters
    ----------
    component_system : ComponentSystem
    windows : list of list of (start, end), one inner list per component
        e.g. [[(2, 4), (7, 9)]] for one component with two peaks,
        or [[(2, 4)], [(7, 9)]] for two separated components.
    heights : list of float, optional
        Peak height per component. Defaults to 1.0 for all components.
    t_start : float, optional
        Start of the time axis. Default is 0.0.
    t_end : float, optional
        End of the time axis. Default is 10.0.
    n_time : int, optional
        Number of time points. Default is 1001.
    name : str, optional
        Name of the SolutionIO object. Default is "outlet".
    """
    if heights is None:
        heights = [1.0] * component_system.n_comp
    time = np.linspace(t_start, t_end, n_time)
    solution = np.zeros((n_time, component_system.n_comp))
    for i, (comp_windows, h) in enumerate(zip(windows, heights)):
        for t0, t1 in comp_windows:
            solution[(time >= t0) & (time < t1), i] = h
    return SolutionIO(name, component_system, time, solution, np.ones(n_time))


def gaussian_chromatogram(
    component_system: ComponentSystem,
    peaks,
    t_start=0.0,
    t_end=10.0,
    n_time=1001,
    name="outlet",
):
    """Gaussian pulse chromatogram.

    Parameters
    ----------
    component_system : ComponentSystem
    peaks : list of (mu, sigma) per component
    t_start : float, optional
        Start of the time axis. Default is 0.0.
    t_end : float, optional
        End of the time axis. Default is 10.0.
    n_time : int, optional
        Number of time points. Default is 1001.
    name : str, optional
        Name of the SolutionIO object. Default is "outlet".
    """
    time = np.linspace(t_start, t_end, n_time)
    solution = np.zeros((n_time, component_system.n_comp))
    for i, (mu, sigma) in enumerate(peaks):
        solution[:, i] = np.exp(-0.5 * ((time - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return SolutionIO(name, component_system, time, solution, np.ones(n_time))
