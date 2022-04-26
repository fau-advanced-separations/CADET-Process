import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import NdArray, DependentlySizedNdArray
from CADETProcess.metric import MetricBase

from CADETProcess.processModel import ComponentSystem


class Performance(Structure):
    """Class for storing the performance parameters after fractionation.

    See Also
    --------
    Fractionation
    ProcessMeta
    RankedPerformance

    """
    _performance_keys = [
        'mass', 'concentration', 'purity', 'recovery',
        'productivity', 'eluent_consumption'
    ]

    mass = DependentlySizedNdArray(dep=('n_comp'))
    concentration = DependentlySizedNdArray(dep=('n_comp'))
    purity = DependentlySizedNdArray(dep=('n_comp'))
    recovery = DependentlySizedNdArray(dep=('n_comp'))
    productivity = DependentlySizedNdArray(dep=('n_comp'))
    eluent_consumption = DependentlySizedNdArray(dep=('n_comp'))

    def __init__(
            self, mass, concentration, purity, recovery,
            productivity, eluent_consumption, component_system=None):

        if component_system is None:
            component_system = ComponentSystem(mass.shape[0])

        self.component_system = component_system
        self.mass = mass
        self.concentration = concentration
        self.purity = purity
        self.recovery = recovery
        self.productivity = productivity
        self.eluent_consumption = eluent_consumption

    @property
    def n_comp(self):
        return self.component_system.n_comp

    def to_dict(self):
        return {key: getattr(self, key).tolist()
                for key in self._performance_keys}

    def __getitem__(self, item):
        if item not in self._performance_keys:
            raise AttributeError('Not a valid performance parameter')

        return getattr(self, item)

    def __repr__(self):
        return \
            f'{self.__class__.__name__}(mass={np.array_repr(self.mass)}, '\
            f'concentration={np.array_repr(self.concentration)}, '\
            f'purity={np.array_repr(self.purity)}, '\
            f'recovery={np.array_repr(self.recovery)}, '\
            f'productivity={np.array_repr(self.productivity)}, '\
            f'eluent_consumption={np.array_repr(self.eluent_consumption)})'


class RankedPerformance():
    """Class for calculating a weighted average of the Performance

    See Also
    --------
    Performance
    ranked_objective_decorator

    """
    _performance_keys = Performance._performance_keys

    def __init__(self, performance, ranking=1.0):
        if not isinstance(performance, Performance):
            raise TypeError('Expected Performance')

        self._performance = performance

        if isinstance(ranking, (float, int)):
            ranking = performance.n_comp * [ranking]
        elif len(ranking) != performance.n_comp:
            raise CADETProcessError('Number of components does not match.')

        self._ranking = ranking

    @property
    def ranking(self):
        return self._ranking

    @ranking.setter
    def ranking(self, ranking):
        n_metrics = self.component_system.n_comp - self.n_exclude

        if isinstance(ranking, (float, int)):
            ranking = n_metrics * [ranking]

        if ranking is not None and len(ranking) != n_metrics:
            raise CADETProcessError(
                'Ranking does not match number of metrics'
            )

        self._ranking = ranking

    def to_dict(self):
        return {
            key: float(getattr(self, key)) for key in self._performance_keys
        }

    def __getattr__(self, item):
        if item not in self._performance_keys:
            raise AttributeError
        return sum(self._performance[item]*self._ranking)/sum(self._ranking)

    def __getitem__(self, item):
        if item not in self._performance_keys:
            raise AttributeError('Not a valid performance parameter')
        return getattr(self, item)

    def __repr__(self):
        return \
            f'{self.__class__.__name__}(mass={np.array_repr(self.mass)}, '\
            f'concentration={np.array_repr(self.concentration)}, '\
            f'purity={np.array_repr(self.purity)}, '\
            f'recovery={np.array_repr(self.recovery)}, '\
            f'productivity={np.array_repr(self.productivity)}, '\
            f'eluent_consumption={np.array_repr(self.eluent_consumption)})'


class PerformanceIndicator(MetricBase):
    def __init__(self, exclude=None, ranking=None):
        self.exclude = exclude
        self.ranking = ranking

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, exclude=None):
        if exclude is None:
            exclude = []

        self._exclude = exclude

    @property
    def ranking(self):
        return self._ranking

    @ranking.setter
    def ranking(self, ranking):
        self._ranking = ranking

    @property
    def bad_metrics(self):
        return 0

    def evaluate(self, performance):
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
                if comp.name in self.exclude:
                    continue
                metric.append(value[i])

        return metric

    __call__ = evaluate


class Mass(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.mass


class Recovery(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.recovery


class Productivity(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.productivity


class EluentConsumption(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.eluent_consumption


class Purity(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.purity


class Concentration(PerformanceIndicator):
    def _evaluate(self, performance):
        return - performance.concentration


class PerformanceProduct(PerformanceIndicator):
    def _evaluate(self, performance):
        return \
            - performance.productivity \
            * performance.recovery \
            * performance.eluent_consumption
