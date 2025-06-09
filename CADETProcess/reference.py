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

"""  # noqa

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionBase, SolutionIO

__all__ = ["ReferenceBase", "ReferenceIO", "FractionationReference"]


class ReferenceBase(SolutionBase):
    """
    Class representing references to be compared with SimulationResults.

    See Also
    --------
    CADETProcess.solution.SolutionBase
    """

    pass


class ReferenceIO(ReferenceBase, SolutionIO):
    """
    A class representing reference data of inlet or outlet concentration profiles.

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
        self,
        name: str,
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
        flow_rate: Optional[float | npt.ArrayLike] = None,
        component_system: Optional[ComponentSystem] = None,
    ) -> None:
        """
        Initialize a ReferenceIO object.

        Parameters
        ----------
        name : str
            The name of the reference.
        time : array-like
            The time points for the reference.
        solution : array-like
            The reference solution values with shape = (n_time, n_comp).
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

        if solution.shape[0] != len(time):
            raise ValueError(
                "Solution had the wrong shape. Solution needs the shape (time, n_comp)."
            )

        solution = np.array(solution, ndmin=2, dtype=np.float64).reshape(len(time), -1)

        if component_system is None:
            n_comp = solution.shape[1]
            component_system = ComponentSystem(n_comp)

        if flow_rate is None:
            flow_rate = 1
        if isinstance(flow_rate, (int, float)):
            flow_rate = flow_rate * np.ones(time.shape)

        super().__init__(name, component_system, time, solution, flow_rate)


class FractionationReference(ReferenceBase):
    """
    A class representing reference data of fractionation data.

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

    See Also
    --------
    CADETProcess.reference.ReferenceBase
    CADETProcess.fractionation.Fraction
    """

    dimensions = SolutionBase.dimensions + ["component_coordinates"]

    def __init__(
        self,
        name: str,
        fractions: list,
        component_system: Optional[ComponentSystem] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize FractionationReference object."""
        from CADETProcess.fractionation import Fraction

        for frac in fractions:
            if not isinstance(frac, Fraction):
                raise TypeError("Expected Fraction.")
            if frac.start is None or frac.end is None:
                raise CADETProcessError("Fractionation times must be provided.")
            if not frac.end > frac.start:
                raise CADETProcessError("Fraction end time must be greater than start.")

        self.fractions = fractions

        time = np.array([(frac.start + frac.end) / 2 for frac in self.fractions])
        solution = np.array([frac.mass / frac.volume for frac in self.fractions])

        if component_system is None:
            n_comp = solution.shape[1]
            component_system = ComponentSystem(n_comp)

        super().__init__(name, component_system, time, solution)
