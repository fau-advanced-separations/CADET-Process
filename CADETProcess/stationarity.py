"""
======================================================
Cyclic Stationarity (:mod:`CADETProcess.stationarity`)
======================================================

.. currentmodule:: CADETProcess.stationarity

Module to evaluate cyclic stationarity of succeeding cycles.

.. autosummary::
    :toctree: generated/

    RelativeArea
    NRMSE
    StationarityEvaluator

"""  # noqa

from typing import Any, Optional

import numpy as np
from addict import Dict

from CADETProcess import SimulationResults, log
from CADETProcess.comparison import Comparator
from CADETProcess.dataStructure import Structure, UnsignedFloat
from CADETProcess.processModel import Inlet

__all__ = ["RelativeArea", "NRMSE", "StationarityEvaluator"]


class CriterionBase(Structure):
    threshold = UnsignedFloat(default=1e-3)

    def __str__(self) -> str:
        return self.__class__.__name__


class RelativeArea(CriterionBase):
    """Class to evaluate difference in relative area as stationarity critereon."""

    pass


class NRMSE(CriterionBase):
    """Class to evaluate NRMSE as stationarity critereon."""

    pass


class StationarityEvaluator(Comparator):
    """Class for checking two succeding chromatograms for stationarity."""

    valid_criteria = ["RelativeArea", "NRMSE"]

    def __init__(
        self,
        criteria: Optional[list[CriterionBase]] = None,
        log_level: str = "WARNING",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the stationarity evaluator.

        Parameters
        ----------
        criteria : List[CriterionBase], optional
            List of criteria for stationarity evaluation, by default None
        log_level : str, optional
            The logging level, by default 'WARNING'
        args : list
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.
        """
        # TODO: Check why cirteria are not stored.
        super().__init__(*args, **kwargs)

        self.logger = log.get_logger("StationarityEvaluator", level=log_level)

        self._criteria = []

    @property
    def criteria(self) -> list:
        """list: List of criteria."""
        return self._criteria

    def add_criterion(self, criterion: CriterionBase) -> None:
        """
        Add a criterion to the list of criteria.

        Parameters
        ----------
        criterion : CriterionBase
            Criterion to add to the list of criteria.
        """
        if not isinstance(criterion, CriterionBase):
            raise TypeError("Expected CriterionBase.")

        self._criteria.append(criterion)

    def assert_stationarity(self, simulation_results: SimulationResults) -> bool:
        """
        Check stationarity of two succeeding cycles.

        Parameters
        ----------
        simulation_results : SimulationResults
            Results of current cycle.

        Returns
        -------
        bool
            True if stationarity is reached. False otherwise.

        Raises
        ------
        TypeError
            If simulation_results is not a SimulationResults object.
        """
        self._metrics = []
        criteria = Dict()
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError("Expcected SimulationResults")

        stationarity = True
        for unit, solution in simulation_results.solution_cycles.items():
            if isinstance(simulation_results.process.flow_sheet[unit], Inlet):
                continue
            solution_previous = solution.outlet[-2]
            solution_this = solution.outlet[-1]
            self.add_reference(solution_previous, update=True, smooth=False)

            for c in self.criteria:
                metric = self.add_difference_metric(
                    str(c), unit, f"{unit}.outlet", smooth=False
                )
                criteria[unit][str(c)]["threshold"] = c.threshold
                diff = metric.evaluate(solution_this)
                criteria[unit][str(c)]["metric"] = diff
                if not np.all(diff <= c.threshold):
                    s = False
                    stationarity = s
                else:
                    s = True
                criteria[unit][str(c)]["stationarity"] = s

        self.logger.debug(f"Stationrity criteria: {criteria}")

        return stationarity
