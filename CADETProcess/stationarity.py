import numpy as np

from CADETProcess import log
from CADETProcess.dataStructure import StructMeta, UnsignedFloat
from CADETProcess import SimulationResults
from CADETProcess.comparison import Comparator


class CriterionBase(metaclass=StructMeta):
    threshold = UnsignedFloat(default=1e-3)

    def __str__(self):
        return self.__class__.__name__


class RelativeArea(CriterionBase):
    pass


class SSE(CriterionBase):
    pass


class StationarityEvaluator(Comparator):
    """Class for checking two succeding chromatograms for stationarity

    Attributes
    ----------

    """
    valid_criteria = ['RelativeArea', 'SSE']

    def __init__(
            self,
            criteria=None,
            log_level='WARNING', save_log=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = log.get_logger(
            'StationarityEvaluator', level=log_level, save_log=save_log
        )

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

        if not isinstance(simulation_results, SimulationResults):
            raise TypeError('Expcected SimulationResults')

        for chrom in simulation_results.chromatograms:
            chrom_previous = \
                simulation_results.solution_cycles[chrom.name].outlet[-2]
            self.add_reference(chrom_previous, update=True, smooth=False)

            for c in self.criteria:
                self.add_difference_metric(
                    str(c), chrom.name, f'{chrom.name}.outlet'
                )

        differences = self.evaluate(simulation_results, smooth=False)

        stationarity = True
        criteria = {}
        for crit, r in zip(self.criteria, differences):
            if not np.all(r <= crit.threshold):
                stationarity = False
            criteria[str(crit)] = r

        self.logger.debug(f'Stationrity criteria: {criteria}')

        return stationarity
