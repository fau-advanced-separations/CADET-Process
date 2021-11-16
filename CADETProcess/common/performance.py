import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import NdArray

class Performance(Structure):
    """Class for storing the performance parameters after fractionation.

    See also
    --------
    Fractionation
    ProcessMeta
    RankedPerformance
    """
    _performance_keys = ['mass', 'concentration', 'purity', 'recovery',
        'productivity', 'eluent_consumption']

    mass = NdArray()
    concentration = NdArray()
    purity = NdArray()
    recovery = NdArray()
    productivity = NdArray()
    eluent_consumption = NdArray()


    def __init__(self, mass, concentration, purity, recovery,
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
        return '{}(mass={}, concentration={}, purity={}, recovery={}, \
                    productivity={}, eluent_consumption={}'.format(
            self.__class__.__name__, np.array_repr(self.mass),
            np.array_repr(self.concentration), np.array_repr(self.purity),
            np.array_repr(self.recovery), np.array_repr(self.productivity),
            np.array_repr(self.eluent_consumption))


class RankedPerformance():
    """Class for calculating a weighted average of the Performance

    See also
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
        return {key: float(getattr(self, key)) for key in self._performance_keys}

    def __getattr__(self, item):
        if item not in self._performance_keys:
            raise AttributeError
        return sum(self._performance[item]*self._ranking)/sum(self._ranking)

    def __getitem__(self, item):
        if item not in self._performance_keys:
            raise AttributeError('Not a valid performance parameter')
        return getattr(self, item)

    def __repr__(self):
        return '{}(mass={}, concentration={}, purity={}, recovery={}, \
                    productivity={}, eluent_consumption={}'.format(
            self.__class__.__name__, np.array_repr(self.mass),
            np.array_repr(self.concentration), np.array_repr(self.purity),
            np.array_repr(self.recovery), np.array_repr(self.productivity),
            np.array_repr(self.eluent_consumption))

def get_bad_performance(n_comp):
    return Performance(
            mass = np.zeros((n_comp,)),
            concentration = np.zeros((n_comp,)),
            purity = np.zeros((n_comp,)),
            recovery = np.zeros((n_comp,)),
            productivity = np.zeros((n_comp,)),
            eluent_consumption = np.zeros((n_comp,)),
            )
