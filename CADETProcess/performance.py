import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import NdArray
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

    mass = NdArray()
    concentration = NdArray()
    purity = NdArray()
    recovery = NdArray()
    productivity = NdArray()
    eluent_consumption = NdArray()

    def __init__(
            self, mass, concentration, purity, recovery,
            productivity, eluent_consumption):
        self.mass = mass
        self.concentration = concentration
        self.purity = purity
        self.recovery = recovery
        self.productivity = productivity
        self.eluent_consumption = eluent_consumption

    @property
    def n_comp(self):
        return self.mass.shape[0]

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
            ranking = [ranking]*performance.n_comp
        elif len(ranking) != performance.n_comp:
            raise CADETProcessError('Number of components does not match.')

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
    def __init__(self, component_system, exclude=None, ranking=None):
        self.component_system = component_system
        self.exclude = exclude
        self.ranking = ranking

    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, exclude=None):
        if exclude is None:
            exclude = []

        component_names = [comp.name for comp in self.component_system]
        if not all([ex in component_names for ex in exclude]):
            raise CADETProcessError('Unknown component in exclude.')

        self._exclude = exclude

    @property
    def n_exclude(self):
        return len(self.exclude)

    @property
    def ranking(self):
        return self._ranking

    @ranking.setter
    def ranking(self, ranking):
        n_metrics = self.component_system.n_comp - self.n_exclude

        if isinstance(ranking, (float, int)):
            ranking = n_metrics * [ranking]

        if ranking is not None and len(ranking) != n_metrics:
            raise CADETProcessError('Ranking does not match number of metrics')

        self._ranking = ranking

    @property
    def n_metrics(self):
        if self.ranking is None:
            return self.component_system.n_comp - self.n_exclude
        else:
            return 1

    @property
    def bad_metrics(self):
        exclude = len(self.exclude)
        return np.zeros((self.n_metrics - exclude,)).tolist()

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
            for i, comp in enumerate(self.component_system):
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
