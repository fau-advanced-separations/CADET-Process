"""
=========================================
Reference (:mod:`CADETProcess.reference`)
=========================================

.. currentmodule:: CADETProcess.reference

This module provides functionality for setting up reference solutions used for
comparison with ``SimulationResults``

.. autosummary::
    :toctree: generated/

    ReferenceBase
    ReferenceIO

"""

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionBase, SolutionIO


__all__ = ['ReferenceBase', 'ReferenceIO']


class ReferenceBase(SolutionBase):
    """Class representing references to be compared with SimulationResults.

    See Also
    --------
    CADETProcess.solution.SolutionBase

    """

    pass


class ReferenceIO(SolutionIO):
    """A class representing reference data of of inlet or outlet unitoperations.

    Attributes
    ----------
    name : str
        The name of the reference.
    component_system : ComponentSystem
        The reference component system.
    time : np.ndarray
        The time points for the reference.
    solution : np.ndarray
        The reference solution values.
    flow_rate : np.ndarray
        The flow rates for the reference.

    See Also
    --------
    CADETProcess.reference.ReferenceBase
    CADETProcess.solution.SolutionIO

    """

    def __init__(
            self, name, time, solution,
            flow_rate=None, component_system=None):
        """Initialize a ReferenceIO object.

        Parameters
        ----------
        name : str
            The name of the reference.
        time : array-like
            The time points for the reference.
        solution : array-like
            The reference solution values.
        flow_rate : array-like or float, optional
            The flow rates for the reference.
            If not provided, flow rate of 1 is assumed.
        component_system : ComponentSystem, optional
            The reference component system.
            If not provided, a ComponentSystem with the same number of components as the
            solution is created.

        Raises
        ------
        TypeError
            If the provided time, solution, or flow rate are not array-like.
        ValueError
            If the time and solution arrays are not the same length.
            If the flow rate array and time array are not the same length.

        """
        time = np.array(time, dtype=np.float64).reshape(-1)
        solution = np.array(solution, ndmin=2, dtype=np.float64).reshape(len(time), -1)

        if component_system is None:
            n_comp = solution.shape[1]
            component_system = ComponentSystem(n_comp)

        if flow_rate is None:
            flow_rate = 1
        if isinstance(flow_rate, (int,  float)):
            flow_rate = flow_rate * np.ones(time.shape)

        super().__init__(name, component_system, time, solution, flow_rate)
