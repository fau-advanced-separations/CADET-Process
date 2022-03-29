from abc import ABC, abstractmethod

import numpy as np


class MetricBase(ABC):
    """Base Class for metrics used in Optimization, Fractionation, Comparison.

    See Also
    --------
    PerformanceIndicator
    DifferenceMetric

    """
    n_metrics = 1
    bad_metrics = np.inf

    @abstractmethod
    def evaluate(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def __str__(self):
        return self.__class__.__name__
