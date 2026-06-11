"""
=========================================
 (:mod:`CADETProcess.normalize`)
=========================================

.. currentmodule:: CADETProcess.parameter_space.normalize

This module provides functionality for normalizing data.


.. autosummary::
    :toctree: generated/

    NormalizerBase
    NullNormalizer
    LinearNormalizer
    LogNormalizer
    AutoNormalizer

"""  # noqa

from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from CADETProcess import plotting
from CADETProcess.numerics import round_to_significant_digits


class NormalizerBase(ABC):
    """Base class for parameter normalization.

    Provides an interface for normalizing an input parameter space to an output
    parameter space.

    Attributes
    ----------
    lb_input : float or np.ndarray
        Lower bounds of the input parameter space.
    ub_input : float or np.ndarray
        Upper bounds of the input parameter space.
    lb : float or np.ndarray
        Lower bounds of the output parameter space.
    ub : float or np.ndarray
        Upper bounds of the output parameter space.
    allow_extended_input : bool
        If True, input values may exceed the declared bounds without raising.
    allow_extended_output : bool
        If True, output values may exceed the declared bounds without raising.
    """

    def __init__(
        self,
        lb_input: float | np.ndarray = -np.inf,
        ub_input: float | np.ndarray = np.inf,
        allow_extended_input: Optional[bool] = False,
        allow_extended_output: Optional[bool] = False,
    ) -> None:
        self.lb_input = lb_input
        self.ub_input = ub_input
        self.allow_extended_input = allow_extended_input
        self.allow_extended_output = allow_extended_output

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """Return whether the normalization is linear."""
        pass

    @property
    def lb_input(self) -> float | np.ndarray:
        """Lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input: float | np.ndarray) -> None:
        self._lb_input = lb_input

    @property
    def ub_input(self) -> float | np.ndarray:
        """Upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input: float | np.ndarray) -> None:
        self._ub_input = ub_input

    @property
    @abstractmethod
    def lb(self) -> float | np.ndarray:
        """Lower bounds of the output parameter space."""
        pass

    @property
    @abstractmethod
    def ub(self) -> float | np.ndarray:
        """Upper bounds of the output parameter space."""
        pass

    def normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """Normalize input values to the output space.

        Parameters
        ----------
        x : float or np.ndarray
            Input parameter values.

        Returns
        -------
        float or np.ndarray
            Normalized parameter values.

        Raises
        ------
        ValueError
            If x exceeds input or output bounds and the corresponding
            ``allow_extended_*`` flag is False.
        """
        if not self.allow_extended_input and not np.all(
            (self.lb_input <= x) & (x <= self.ub_input)
        ):
            raise ValueError("Value exceeds input bounds.")

        x = self._normalize(x)

        if not self.allow_extended_output and not np.all(
            (self.lb <= x) & (x <= self.ub)
        ):
            raise ValueError("Value exceeds output bounds.")

        return x

    @abstractmethod
    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        pass

    def denormalize(
        self,
        x: float | np.ndarray,
        significant_digits: Optional[int] = None,
    ) -> float | np.ndarray:
        """Denormalize output values back to the input space.

        Parameters
        ----------
        x : float or np.ndarray
            Output parameter values in the normalized space.
        significant_digits : int, optional
            Round to this many significant digits. If None, no rounding.

        Returns
        -------
        float or np.ndarray
            Denormalized parameter values.
        """
        x_ = round_to_significant_digits(x, digits=significant_digits)

        if not self.allow_extended_output and not np.all(
            (self.lb <= x_) & (x_ <= self.ub)
        ):
            raise ValueError("Value exceeds output bounds.")

        x_ = self._denormalize(x_)
        x_ = round_to_significant_digits(x_, digits=significant_digits)

        if not self.allow_extended_input and not np.all(
            (self.lb_input <= x_) & (x_ <= self.ub_input)
        ):
            raise ValueError("Value exceeds input bounds.")

        return x_

    @abstractmethod
    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        pass

    @plotting.figure_utils
    def plot(self, ax: plt.Axes, use_log_scale: bool = False) -> None:
        """Plot the normalized space against the input space."""
        allow_extended_input = self.allow_extended_input
        self.allow_extended_input = True

        y = np.linspace(self.lb, self.ub)
        x = self.denormalize(y)

        ax.plot(x, y)
        ax.set_xlabel("Input Space")
        ax.set_ylabel("Normalized Space")

        if use_log_scale:
            ax.set_xscale("log")

        self.allow_extended_input = allow_extended_input

    def __str__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__


class NullNormalizer(NormalizerBase):
    """Normalizer that returns input values unchanged."""

    @property
    def is_linear(self) -> bool:
        """Return True; identity is a linear map."""
        return True

    @property
    def lb(self) -> float | np.ndarray:
        """Lower bound of output space (same as input lower bound)."""
        return self.lb_input

    @property
    def ub(self) -> float | np.ndarray:
        """Upper bound of output space (same as input upper bound)."""
        return self.ub_input

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x


class LinearNormalizer(NormalizerBase):
    """Normalizer that scales values linearly to [0, 1]."""

    @property
    def is_linear(self) -> bool:
        """Return True."""
        return True

    @property
    def lb(self) -> float:
        """Lower bound of output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Upper bound of output space (1)."""
        return 1.0

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return (x - self.lb_input) / (self.ub_input - self.lb_input)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return (self.ub_input - self.lb_input) * x + self.lb_input


class LogNormalizer(NormalizerBase):
    """Normalizer that scales values logarithmically to [0, 1]."""

    @property
    def is_linear(self) -> bool:
        """Return False."""
        return False

    @property
    def lb(self) -> float:
        """Lower bound of output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Upper bound of output space (1)."""
        return 1.0

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        if self.lb_input <= 0:
            x_ = x + (abs(self.lb_input) + 1)
            ub = 1 + (self.ub_input - self.lb_input)
            return np.log(x_) / np.log(ub)
        else:
            return np.log(x / self.lb_input) / np.log(self.ub_input / self.lb_input)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        if self.lb_input <= 0:
            return (
                np.exp(x * np.log(self.ub_input - self.lb_input + 1))
                + self.lb_input
                - 1
            )
        else:
            return self.lb_input * np.exp(x * np.log(self.ub_input / self.lb_input))


class AutoNormalizer(NormalizerBase):
    """Normalizer that automatically selects between linear and logarithmic scaling.

    Attributes
    ----------
    threshold : int
        Ratio/range threshold above which logarithmic normalization is preferred.
    """

    def __init__(self, *args: Any, threshold: int = 100, **kwargs: Any) -> None:
        self.linear = LinearNormalizer()
        self.log = LogNormalizer()

        super().__init__(*args, **kwargs)
        self.threshold = threshold

        self.linear.allow_extended_input = self.allow_extended_input
        self.linear.allow_extended_output = self.allow_extended_output
        self.log.allow_extended_input = self.allow_extended_input
        self.log.allow_extended_output = self.allow_extended_output

    @property
    def is_linear(self) -> bool:
        """Return True when linear normalization is active."""
        return self.use_linear

    @property
    def use_linear(self) -> bool:
        """Return True when the ratio/range falls below the threshold."""
        if self.lb_input <= 0:
            return np.log10(self.ub_input - self.lb_input) < np.log10(self.threshold)
        return (self.ub_input / self.lb_input) < self.threshold

    @property
    def use_log(self) -> bool:
        """Return True when logarithmic normalization is active."""
        return not self.use_linear

    @property
    def lb(self) -> float:
        """Lower bound of output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Upper bound of output space (1)."""
        return 1.0

    @property
    def lb_input(self) -> float | np.ndarray:
        """Lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input: float | np.ndarray) -> None:
        """Set lower bounds and propagate to sub-normalizers."""
        self.linear.lb_input = lb_input
        self.log.lb_input = lb_input
        self._lb_input = lb_input

    @property
    def ub_input(self) -> float | np.ndarray:
        """Upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input: float | np.ndarray) -> None:
        """Set upper bounds and propagate to sub-normalizers."""
        self.linear.ub_input = ub_input
        self.log.ub_input = ub_input
        self._ub_input = ub_input

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.log._normalize(x) if self.use_log else self.linear._normalize(x)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.log._denormalize(x) if self.use_log else self.linear._denormalize(x)
