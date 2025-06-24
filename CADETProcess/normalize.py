

"""
=========================================
 (:mod:`CADETProcess.normalize`)
=========================================

.. currentmodule:: CADETProcess.normalize

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
    """
    Base class for parameter normalization.

    This class provides an interface for normalizing an input parameter space to some
    output parameter space.

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
        If True, the input value may exceed the lower/upper bounds.
        Else, an exception is thrown.
    allow_extended_output : bool
        If True, the output value may exceed the lower/upper bounds.
        Else, an exception is thrown.

    Raises
    ------
    ValueError
        If lb_input and ub_input have different shapes.

    Notes
    -----
    - This is an abstract base class and cannot be instantiated directly.
    - The `normalize` method must be implemented by a subclass.

    Examples
    --------
    >>> class MyNormalizer(NormalizerBase):
    ...     def normalize(self, x: float) -> float:
    ...         return x**2
    >>> t = MyNormalizer(lb_input=0, ub_input=10, lb=-100, ub=100)
    >>> t.normalize(3)
    9
    """

    def __init__(
        self,
        lb_input: float | np.ndarray = -np.inf,
        ub_input: float | np.ndarray = np.inf,
        allow_extended_input: Optional[bool] = False,
        allow_extended_output: Optional[bool] = False,
    ) -> None:
        """
        Initialize NormalizerBase.

        Parameters
        ----------
        lb_input : float or np.ndarray, optional
            Lower bounds of the input parameter space. Default is -inf.
        ub_input : float or np.ndarray, optional
            Upper bounds of the input parameter space. Default is inf.
        allow_extended_input : bool, optional
            If True, the input value may exceed the lower/upper bounds.
            Else, an exception is thrown. Default is False.
        allow_extended_output : bool, optional
            If True, the output value may exceed the lower/upper bounds.
            Else, an exception is thrown. Default is False.
        """
        self.lb_input = lb_input
        self.ub_input = ub_input
        self.allow_extended_input = allow_extended_input
        self.allow_extended_output = allow_extended_output

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """Return whether the normalizion is linear."""
        pass

    @property
    def lb_input(self) -> float | np.ndarray:
        """Return the lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input: float | np.ndarray) -> None:
        self._lb_input = lb_input

    @property
    def ub_input(self) -> float | np.ndarray:
        """Return the upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input: float | np.ndarray) -> None:
        self._ub_input = ub_input

    @property
    @abstractmethod
    def lb(self) -> float | np.ndarray:
        """Return the lower bounds of the output parameter space."""
        pass

    @property
    @abstractmethod
    def ub(self) -> float | np.ndarray:
        """Return the upper bounds of the output parameter space."""
        pass

    def normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize the input parameter space to the output parameter space.

        Applies the normalizion function `_normalize` to `x` after performing input
        bounds checking. If the normalized value exceeds the output bounds, an error
        is raised.

        Parameters
        ----------
        x : float or np.ndarray
            Input parameter values.

        Returns
        -------
        float or np.ndarray
            normalized parameter values.

        Raises
        ------
        ValueError
            If `x` exceeds input or output bounds and `allow_extended_*` is False.
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
        """
        Apply the normalizion from input to output parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : float or np.ndarray
            Input parameter values.

        Returns
        -------
        float or np.ndarray
            normalized parameter values.
        """
        pass

    def denormalize(
        self,
        x: float | np.ndarray,
        significant_digits: Optional[int] = None,
    ) -> float | np.ndarray:
        """
        Normalize the output parameter space back to the input parameter space.

        Parameters
        ----------
        x : float or np.ndarray
            Output parameter values.
        significant_digits : int, optional
            float | np.ndarray of significant figures to which variable can be rounded.
            If None, variable is not rounded.

        Returns
        -------
        float or np.ndarray
            normalized parameter values.
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
        """
        Apply the inverse normalizion from output to input parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : float or np.ndarray
            Output parameter values.

        Returns
        -------
        float or np.ndarray
            Normalized parameter values.
        """
        pass

    @plotting.create_and_save_figure
    def plot(self, ax: plt.Axes, use_log_scale: bool = False) -> None:
        """
        Plot the normalized space against the input space.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to plot on.
        use_log_scale : bool, optional
            If True, use a logarithmic scale for the x-axis.
        """
        allow_extended_input = self.allow_extended_input
        self.allow_extended_input = True

        y = np.linspace(self.lb, self.ub)
        x = self.denormalize(y)

        ax.plot(x, y)
        ax.set_xlabel("Input Space")
        ax.set_ylabel("normalized Space")

        if use_log_scale:
            ax.set_xscale("log")

        self.allow_extended_input = allow_extended_input

    def __str__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__


class NullNormalizer(NormalizerBase):
    """
    A normalizer that performs no normalizion.

    This class simply returns the input values as output without modification.

    See Also
    --------
    NormalizerBase : The base class for parameter normalizion.
    """

    @property
    def is_linear(self) -> bool:
        """Return True, as this is a linear normalizion."""
        return True

    @property
    def lb(self) -> float | np.ndarray:
        """Return the lower bound of the output space (same as input lower bound)."""
        return self.lb_input

    @property
    def ub(self) -> float | np.ndarray:
        """Return the upper bound of the output space (same as input upper bound)."""
        return self.ub_input

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Return the input value(s) as output without modification.

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be normalized.

        Returns
        -------
        float or np.ndarray
            The normalized output value(s) (same as input).
        """
        return x

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Return the output value(s) as input without modification.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) to be unnormalized.

        Returns
        -------
        float or np.ndarray
            The unnormalized input value(s) (same as output).
        """
        return x


class LinearNormalizer(NormalizerBase):
    """
    A normalizer that normalizes values linearly to the range [0, 1].

    This normalizion scales the input value between the given lower
    and upper bounds into a normalized range of [0,1].

    See Also
    --------
    NormalizerBase : The base class for parameter normalizion.
    """

    @property
    def is_linear(self) -> bool:
        """Return True, as this is a linear normalizion."""
        return True

    @property
    def lb(self) -> float:
        """Return the lower bound of the output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Return the upper bound of the output space (1)."""
        return 1.0

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize input values to the range [0,1].

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be normalized.

        Returns
        -------
        float or np.ndarray
            The normalized output value(s) in the range [0,1].
        """
        return (x - self.lb_input) / (self.ub_input - self.lb_input)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Denormalize output values back to the original range.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the normalized range [0,1].

        Returns
        -------
        float or np.ndarray
            The unnormalized input value(s) in the original range.
        """
        return (self.ub_input - self.lb_input) * x + self.lb_input


class LogNormalizer(NormalizerBase):
    """
    A normalizer that normalizes values logarithmically to the range [0, 1].

    This normalizion scales input values logarithmically between the given lower
    and upper bounds into a normalized range of [0,1].

    See Also
    --------
    NormalizerBase : The base class for parameter normalizion.
    """

    @property
    def is_linear(self) -> bool:
        """Return False, as this is a non-linear normalizion."""
        return False

    @property
    def lb(self) -> float:
        """Return the lower bound of the output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Return the upper bound of the output space (1)."""
        return 1.0

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize input values to the range [0,1] using a logarithmic normalizion.

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be normalized.

        Returns
        -------
        float or np.ndarray
            The normalized output value(s) in the range [0,1].

        Raises
        ------
        ValueError
            If `lb_input` is non-positive, the normalizion shifts all values accordingly.
        """
        if self.lb_input <= 0:
            x_ = x + (abs(self.lb_input) + 1)
            ub = 1 + (self.ub_input - self.lb_input)
            return np.log(x_) / np.log(ub)
        else:
            return np.log(x / self.lb_input) / np.log(self.ub_input / self.lb_input)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Denormalize output values back to the original range using logarithmic inverse.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the normalized range [0,1].

        Returns
        -------
        float or np.ndarray
            The unnormalized input value(s) in the original range.
        """
        if self.lb_input <= 0:
            return (
                np.exp(x * np.log(self.ub_input - self.lb_input + 1))
                + self.lb_input
                - 1
            )
        else:
            return self.lb_input * np.exp(x * np.log(self.ub_input / self.lb_input))


class AutoNormalizer(NormalizerBase):
    """
    A normalizer that automatically selects between linear and logarithmic normalizions.

    Normalizes the input value to the range [0, 1] using either
    the :class:`LinearNormalizer` or the :class:`LogNormalizer`
    based on the input parameter space.

    Attributes
    ----------
    linear : LinearNormalizer
        Instance of the linear normalization transform.
    log : LogNormalizater
        Instance of the logarithmic normalization transform.
    threshold : int
        The maximum threshold to switch from linear to logarithmic normalizion.

    See Also
    --------
    NormalizerBase
    LinearNormalizer
    LogNormalizer
    """

    def __init__(self, *args: Any, threshold: int = 100, **kwargs: Any) -> None:
        """
        Initialize an AutoNormalizer object.

        Parameters
        ----------
        *args : tuple
            Arguments for the :class:`NormalizerBase` class.
        threshold : int, optional
            The maximum threshold to switch from linear to logarithmic
            normalizion. The default is 100.
        **kwargs : dict
            Keyword arguments for the :class:`NormalizerBase` class.
        """
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
        """Return True if linear normalizion is used, otherwise False."""
        return self.use_linear

    @property
    def use_linear(self) -> bool:
        """Determine whether linear normalizion should be used."""
        if self.lb_input <= 0:
            return np.log10(self.ub_input - self.lb_input) < np.log10(self.threshold)
        return (self.ub_input / self.lb_input) < self.threshold

    @property
    def use_log(self) -> bool:
        """Return True if logarithmic normalizion is used, otherwise False."""
        return not self.use_linear

    @property
    def lb(self) -> float:
        """Return the lower bound of the output parameter space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Return the upper bound of the output parameter space (1)."""
        return 1.0

    @property
    def lb_input(self) -> float | np.ndarray:
        """Return the lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input: float | np.ndarray) -> None:
        """Set the lower bounds of the input parameter space."""
        self.linear.lb_input = lb_input
        self.log.lb_input = lb_input
        self._lb_input = lb_input

    @property
    def ub_input(self) -> float | np.ndarray:
        """Return the upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input: float | np.ndarray) -> None:
        """Set the upper bounds of the input parameter space."""
        self.linear.ub_input = ub_input
        self.log.ub_input = ub_input
        self._ub_input = ub_input

    def _normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize the input value to an output value in the range [0, 1].

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be normalized.

        Returns
        -------
        float or np.ndarray
            The normalized output value(s).
        """
        return self.log._normalize(x) if self.use_log else self.linear._normalize(x)

    def _denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Denormalize the output value back to the input parameter space.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the normalized range.

        Returns
        -------
        float or np.ndarray
            The denormalized input value(s).
        """
        return self.log._denormalize(x) if self.use_log else self.linear._denormalize(x)
