"""
=============================================
Performance (:mod:`CADETProcess.performance`)
=============================================

.. currentmodule:: CADETProcess.performance


Performance data
================

Classes for storing the performance parameters after fractionation.

.. autosummary::
    :toctree: generated/

    Performance
    RankedPerformance

Performance Indicators
======================

Individual performance indicators (extracted from Performance).
Mostly convenince method.

.. autosummary::
    :toctree: generated/

    PerformanceIndicator
    Mass
    Recovery
    Productivity
    EluentConsumption
    Purity
    Concentration
    PerformanceProduct
    MassBalanceDifference

Notes
-----
Performance Indicators might be deprecated in future since with new evaluation chains
it's no longer required for setting up as optimization problem.

"""  # noqa

from typing import Any, Optional

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import SizedNdArray, Structure
from CADETProcess.metric import MetricBase
from CADETProcess.processModel import ComponentSystem


class Performance(Structure):
    """
    Class for storing the performance parameters after fractionation.

    Attributes
    ----------
    mass : np.ndarray
        Mass of each component in the system after fractionation.
        Size depends on number of components.
    concentration : np.ndarray
        Concentration of each component in the system after fractionation.
        Size depends on number of components.
    purity : np.ndarray
        Purity of each component in the system after fractionation.
        Size depends on number of components.
    recovery : np.ndarray
        Recovery of each component in the system after fractionation.
        Size depends on number of components.
    productivity : np.ndarray
        Productivity of each component in the system after fractionation.
        Size depends on number of components.
    eluent_consumption : np.ndarray
        Eluent consumption of each component in the system after fractionation.
        Size depends on number of components.
    mass_balance_difference : np.ndarray
        Mass balance difference of each component.
        Size depends on number of components.
    component_system : ComponentSystem
        The component system used for fractionation.
        If not provided, a default component system is used.

    See Also
    --------
    CADETProcess.fractionation
    RankedPerformance
    """

    _performance_keys = [
        "mass",
        "concentration",
        "purity",
        "recovery",
        "productivity",
        "eluent_consumption",
        "mass_balance_difference",
    ]

    mass = SizedNdArray(size=("n_comp"))
    concentration = SizedNdArray(size=("n_comp"))
    purity = SizedNdArray(size=("n_comp"))
    recovery = SizedNdArray(size=("n_comp"))
    productivity = SizedNdArray(size=("n_comp"))
    eluent_consumption = SizedNdArray(size=("n_comp"))
    mass_balance_difference = SizedNdArray(size=("n_comp"))

    def __init__(
        self,
        mass: np.ndarray,
        concentration: np.ndarray,
        purity: np.ndarray,
        recovery: np.ndarray,
        productivity: np.ndarray,
        eluent_consumption: np.ndarray,
        mass_balance_difference: np.ndarray,
        component_system: Optional[ComponentSystem] = None,
    ) -> None:
        """
        Initialize Performance.

        Parameters
        ----------
        mass : ndarray
            The mass of each component.
        concentration : ndarray
            The concentration of each component.
        purity : ndarray
            The purity of each component.
        recovery : ndarray
            The recovery of each component.
        productivity : ndarray
            The productivity of each component.
        eluent_consumption : ndarray
            The eluent consumption of each component.
        mass_balance_difference : ndarray
            The difference in mass balance of each component.
        component_system : ComponentSystem
            The ComponentSystem object that describes the system's components.
        """
        if component_system is None:
            component_system = ComponentSystem(mass.shape[0])

        self.component_system = component_system
        self.mass = mass
        self.concentration = concentration
        self.purity = purity
        self.recovery = recovery
        self.productivity = productivity
        self.eluent_consumption = eluent_consumption
        self.mass_balance_difference = mass_balance_difference

    @property
    def n_comp(self) -> int:
        """int: Number of components in the system."""
        return self.component_system.n_comp

    def to_dict(self) -> dict:
        """Return dictionary representation of the object."""
        return {key: getattr(self, key).tolist() for key in self._performance_keys}

    def __getitem__(self, item: str) -> Any:
        """Get an attribute of the object by its name."""
        if item not in self._performance_keys:
            raise AttributeError("Not a valid performance parameter")

        return getattr(self, item)

    def __repr__(self) -> str:
        """str: String representation of the object."""
        return (
            f"{self.__class__.__name__}(mass={np.array_repr(self.mass)}, "
            f"concentration={np.array_repr(self.concentration)}, "
            f"purity={np.array_repr(self.purity)}, "
            f"recovery={np.array_repr(self.recovery)}, "
            f"productivity={np.array_repr(self.productivity)}, "
            f"eluent_consumption={np.array_repr(self.eluent_consumption)} "
            f"mass_balance_difference={np.array_repr(self.mass_balance_difference)})"
        )


class RankedPerformance:
    """
    Class for calculating a weighted average of the Performance.

    See Also
    --------
    Performance
    ranked_objective_decorator
    """

    _performance_keys = Performance._performance_keys

    def __init__(
        self,
        performance: Performance,
        ranking: Optional[float] = 1.0,
    ) -> None:
        """Initialize RankedPerformance."""
        if not isinstance(performance, Performance):
            raise TypeError("Expected Performance")

        self._performance = performance

        self.ranking = ranking

    @property
    def performance(self) -> Performance:
        """Performance: Performance object."""
        return self._performance

    @property
    def ranking(self) -> float:
        """list[float]: Relative weighting factors for multi component evaluation."""
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: tuple[float | int] | np.ndarray) -> None:
        if isinstance(ranking, (float, int)):
            ranking = self.performance.n_comp * [ranking]
        elif len(ranking) != self.performance.n_comp:
            raise CADETProcessError("Number of components does not match.")

        self._ranking = ranking

    def to_dict(self) -> dict:
        """Return dictionary representation of the object."""
        return {key: float(getattr(self, key)) for key in self._performance_keys}

    def __getattr__(self, item: str) -> float:
        """Retrieve a performance attribute by its name, weighted by ranking."""
        if item not in self._performance_keys:
            raise AttributeError
        return sum(self._performance[item] * self.ranking) / sum(self.ranking)

    def __getitem__(self, item: str) -> Any:
        """Retrieve an attribute of the object by its name using indexing syntax."""
        if item not in self._performance_keys:
            raise AttributeError("Not a valid performance parameter")
        return getattr(self, item)

    def __repr__(self) -> str:
        """str: Sting representation of the object."""
        return (
            f"{self.__class__.__name__}(mass={np.array_repr(self.mass)}, "
            f"concentration={np.array_repr(self.concentration)}, "
            f"purity={np.array_repr(self.purity)}, "
            f"recovery={np.array_repr(self.recovery)}, "
            f"productivity={np.array_repr(self.productivity)}, "
            f"eluent_consumption={np.array_repr(self.eluent_consumption)} "
            f"mass_balance_difference={np.array_repr(self.mass_balance_difference)})"
        )


class PerformanceIndicator(MetricBase):
    """
    Base class for performance indicators used in optimization and fractionation.

    See Also
    --------
    RankedPerformance
    """

    def __init__(self, ranking: Optional[list[float]] = None) -> None:
        """
        Initialize PerformanceIndicator.

        Parameters
        ----------
        ranking : list of floats, optional
            Weights to rank individual compoments. If None, all compoments are ranke
        """
        self.ranking = ranking

    @property
    def ranking(self) -> float:
        """list[float]: Relative weighting factors for multi component evaluation."""
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: list[float]) -> None:
        self._ranking = ranking

    @property
    def bad_metrics(self) -> int:
        """int: Bad metrics to use when evaluation fails."""
        return 0

    def evaluate(self, performance: Performance) -> list:
        """
        Evaluate the performance indicator for the given performance data.

        Parameters
        ----------
        performance : Performance
            Object containing performance data.

        Returns
        -------
        list
            List of performance indicator values.
        """
        try:
            performance = performance.performance
        except AttributeError:
            pass

        if self.ranking is not None:
            performance = RankedPerformance(performance, self.ranking)

        value = self._evaluate(performance).tolist()

        if self.ranking is not None:
            metric = [value]
        else:
            metric = []
            for i, comp in enumerate(performance.component_system):
                metric.append(value[i])

        return metric

    __call__ = evaluate

    def __str__(self) -> str:
        """str: String representation of the class."""
        return self.__class__.__name__


class Mass(PerformanceIndicator):
    """
    Performance indicator based on the mass of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.mass


class Recovery(PerformanceIndicator):
    """
    Performance indicator based on the recovery of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.recovery


class Productivity(PerformanceIndicator):
    """
    Performance indicator based on the productivity of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.productivity


class EluentConsumption(PerformanceIndicator):
    """
    Performance indicator based on the specific eluent consumption of each component.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.eluent_consumption


class Purity(PerformanceIndicator):
    """
    Performance indicator based on the purity of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.purity


class Concentration(PerformanceIndicator):
    """
    Performance indicator based on the concentration of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return performance.concentration


class PerformanceProduct(PerformanceIndicator):
    """
    Performance indicator based on the product of several performance indicators.

    See Also
    --------
    Productivity
    Recovery
    EluentConsumption
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return (
            performance.productivity
            * performance.recovery
            * performance.eluent_consumption
        )


class MassBalanceDifference(PerformanceIndicator):
    """
    Performance indicator based on the mass balance of each component in the system.

    See Also
    --------
    PerformanceIndicator
    """

    def _evaluate(self, performance: Performance) -> np.ndarray:
        return np.abs(performance.mass_balance_difference)
