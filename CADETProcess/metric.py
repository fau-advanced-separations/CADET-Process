from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MetricBase(ABC):
    """
    Abstract base class for metrics used in optimization, fractionation, and comparison.

    Attributes
    ----------
    n_metrics : int
        Number of metrics, default is 1.
    bad_metrics : float
        Value representing a bad metric, default is infinity.

    """

    n_metrics = 1
    bad_metrics = np.inf

    @abstractmethod
    def evaluate(self) -> Any:
        """
        Evaluate the metric.

        This method should be implemented by subclasses to perform specific metric calculations.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke the evaluate method.

        Parameters
        ----------
        *args : tuple
            Variable length argument list passed to evaluate.
        **kwargs : dict
            Arbitrary keyword arguments passed to evaluate.

        Returns
        -------
        Any
            The result of the metric evaluation.
        """
        return self.evaluate(*args, **kwargs)

    def __str__(self) -> str:
        """
        Return the class name as the string representation of the instance.

        Returns
        -------
        str
            The name of the class.
        """
        return self.__class__.__name__
