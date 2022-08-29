from addict import Dict
import numpy as np

from CADETProcess import log
from CADETProcess.dataStructure import StructMeta, UnsignedFloat
from CADETProcess import SimulationResults
from CADETProcess.comparison import Comparator
from CADETProcess.processModel import Source


class CriterionBase(metaclass=StructMeta):
    threshold = UnsignedFloat(default=1e-3)

    def __str__(self):
        return self.__class__.__name__


class RelativeArea(CriterionBase):
    pass


class SSE(CriterionBase):
    pass


class StationarityEvaluator(Comparator):
    """Class for checking two succeding chromatograms for stationarity"""
    valid_criteria = ['RelativeArea', 'SSE']

    def __init__(
            self,
            criteria=None,
            log_level='WARNING',
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = log.get_logger('StationarityEvaluator', level=log_level)

        self._criteria = []

    @property
    def criteria(self):
        return self._criteria

    def add_criterion(self, criterion):
        if not isinstance(criterion, CriterionBase):
            raise TypeError("Expected CriterionBase.")

        self._criteria.append(criterion)

    def assert_stationarity(self, simulation_results):
        """Check stationarity of two succeeding cycles.

        Parameters
        ----------
        simulation_results : SimulationResults
            Results of current cycle.

        Returns
        -------
        bool
            True if stationarity is reached. False otherwise

        """
        self._metrics = []
        criteria = Dict()
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError('Expcected SimulationResults')

        stationarity = True
        for unit, solution in simulation_results.solution_cycles.items():
            if isinstance(simulation_results.process.flow_sheet[unit], Source):
                continue
            solution_previous = solution.outlet[-2]
            solution_this = solution.outlet[-1]
            self.add_reference(solution_previous, update=True, smooth=False)

            for c in self.criteria:
                metric = self.add_difference_metric(
                    str(c), unit, f'{unit}.outlet', smooth=False
                )
                criteria[unit][str(c)]['threshold'] = c.threshold
                diff = metric.evaluate(solution_this)
                criteria[unit][str(c)]['metric'] = diff
                if not np.all(diff <= c.threshold):
                    s = False
                    stationarity = s
                else:
                    s = True
                criteria[unit][str(c)]['stationarity'] = s

        self.logger.debug(f'Stationrity criteria: {criteria}')

        return stationarity
