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

from CADETProcess import log
from CADETProcess.comparison import Comparator
from CADETProcess.dataStructure import Structure, UnsignedFloat
from CADETProcess.processModel import Inlet
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import SolutionIO

__all__ = ["MassBalance", "NRMSE", "RelativeArea", "StationarityEvaluator"]


class CriterionBase(Structure):
    threshold = UnsignedFloat(default=1e-3)

    def __str__(self) -> str:
        return self.__class__.__name__


class MassBalance(CriterionBase):
    """Class to evaluate mass balance as stationarity critereon."""

    pass


class NRMSE(CriterionBase):
    """Class to evaluate NRMSE as stationarity critereon."""

    pass


class RelativeArea(CriterionBase):
    """Class to evaluate difference in relative area as stationarity critereon."""

    pass


class StationarityEvaluator(Comparator):
    """Class for checking two succeding chromatograms for stationarity."""

    valid_criteria = ["MassBalance", "NRMSE", "RelativeArea"]

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
        super().__init__(*args, **kwargs)
        self.logger = log.get_logger("StationarityEvaluator", level=log_level)
        self._criteria = []

    @property
    def criteria(self) -> list[CriterionBase]:
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

        process = simulation_results.process
        flow_sheet = process.flow_sheet

        stationarity = True

        # System Mass Balance
        for c in self.criteria:
            if not isinstance(c, MassBalance):
                continue

            m_feed = process.m_feed
            results_outlets = [
                simulation_results.solution_cycles[unit.name].outlet[-1]
                for unit in flow_sheet.outlets
            ]
            m_out = np.sum([
                outlet_solution.create_fraction().mass
                for outlet_solution in results_outlets
            ], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = abs(m_feed - m_out) / m_feed
                diff = np.where(np.isfinite(diff), diff, 0.0)

            if not np.all(diff <= c.threshold):
                s = False
                stationarity = s
            else:
                s = True

            criteria[str(c)]["threshold"] = c.threshold
            criteria[str(c)]["stationarity"] = s

        # Per unit comparison
        def _assert_unit_io(
            solution_previous: SolutionIO,
            solution_this: SolutionIO,
            unit: str,
            side: str,
            criteria: dict,
            c: object,
        ) -> None:
            """Assert stationarity for a given inlet/outlet of a unit."""
            solution_previous.name = f"{unit}.{side}"
            self.add_reference(
                solution_previous,
                update=True,
                smooth=False,
            )
            metric = self.add_difference_metric(
                str(c),
                f"{unit}.{side}",
                f"{unit}.{side}",
                smooth=False,
            )
            diff = metric.evaluate(solution_this)
            criteria[str(c)][unit][side]["metric"] = diff
            stationarity = np.all(diff <= c.threshold)
            criteria[str(c)][unit][side]["stationarity"] = stationarity

        for unit, solution in simulation_results.solution_cycles.items():
            if isinstance(flow_sheet[unit], Inlet):
                continue

            for c in self.criteria:
                if isinstance(c, MassBalance):
                    continue
                _assert_unit_io(
                    solution.inlet[-2],
                    solution.inlet[-1],
                    unit,
                    "inlet",
                    criteria,
                    c,
                )
                _assert_unit_io(
                    solution.outlet[-2],
                    solution.outlet[-1],
                    unit,
                    "outlet",
                    criteria,
                    c,
                )

        self.logger.debug(f"Stationrity criteria: {criteria}")

        return stationarity
