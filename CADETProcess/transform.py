"""
=========================================
Transform (:mod:`CADETProcess.transform`)
=========================================

.. currentmodule:: CADETProcess.transform

This module provides functionality for transforming data.


.. autosummary::
    :toctree: generated/

    TransformerBase
    NullTransformer
    NormLinearTransformer
    NormLogTransformer
    AutoTransformer

"""  # noqa

from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from CADETProcess import plotting
from CADETProcess.numerics import round_to_significant_digits


class TransformerBase(ABC):
    """
    Base class for parameter transformation.

    This class provides an interface for transforming an input parameter space to some
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
    - The `transform` method must be implemented by a subclass.

    Examples
    --------
    >>> class MyTransformer(TransformerBase):
    ...     def transform(self, x: float) -> float:
    ...         return x ** 2
    ...
    >>> t = MyTransformer(lb_input=0, ub_input=10, lb=-100, ub=100)
    >>> t.transform(3)
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
        Initialize TransformerBase.

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
        """Return whether the transformation is linear."""
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

    def transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Transform the input parameter space to the output parameter space.

        Applies the transformation function `_transform` to `x` after performing input
        bounds checking. If the transformed value exceeds the output bounds, an error
        is raised.

        Parameters
        ----------
        x : float or np.ndarray
            Input parameter values.

        Returns
        -------
        float or np.ndarray
            Transformed parameter values.

        Raises
        ------
        ValueError
            If `x` exceeds input or output bounds and `allow_extended_*` is False.
        """
        if not self.allow_extended_input and not np.all(
            (self.lb_input <= x) & (x <= self.ub_input)
        ):
            raise ValueError("Value exceeds input bounds.")

        x = self._transform(x)

        if not self.allow_extended_output and not np.all(
            (self.lb <= x) & (x <= self.ub)
        ):
            raise ValueError("Value exceeds output bounds.")

        return x

    @abstractmethod
    def _transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Apply the transformation from input to output parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : float or np.ndarray
            Input parameter values.

        Returns
        -------
        float or np.ndarray
            Transformed parameter values.
        """
        pass

    def untransform(
        self,
        x: float | np.ndarray,
        significant_digits: Optional[int] = None,
    ) -> float | np.ndarray:
        """
        Transform the output parameter space back to the input parameter space.

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
            Transformed parameter values.
        """
        x_ = round_to_significant_digits(x, digits=significant_digits)

        if (
            not self.allow_extended_output
            and
            not np.all((self.lb <= x_) & (x_ <= self.ub))
        ):
            raise ValueError("Value exceeds output bounds.")

        x_ = self._untransform(x_)
        x_ = round_to_significant_digits(x_, digits=significant_digits)

        if (
            not self.allow_extended_input
            and
            not np.all((self.lb_input <= x_) & (x_ <= self.ub_input))
        ):
            raise ValueError("Value exceeds input bounds.")

        return x_

    @abstractmethod
    def _untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Apply the inverse transformation from output to input parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : float or np.ndarray
            Output parameter values.

        Returns
        -------
        float or np.ndarray
            Transformed parameter values.
        """
        pass

    @plotting.create_and_save_figure
    def plot(self, ax: plt.Axes, use_log_scale: bool = False) -> None:
        """
        Plot the transformed space against the input space.

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
        x = self.untransform(y)

        ax.plot(x, y)
        ax.set_xlabel("Input Space")
        ax.set_ylabel("Transformed Space")

        if use_log_scale:
            ax.set_xscale("log")

        self.allow_extended_input = allow_extended_input

    def __str__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__


class NullTransformer(TransformerBase):
    """
    A transformer that performs no transformation.

    This class simply returns the input values as output without modification.

    See Also
    --------
    TransformerBase : The base class for parameter transformation.
    """

    @property
    def is_linear(self) -> bool:
        """Return True, as this is a linear transformation."""
        return True

    @property
    def lb(self) -> float | np.ndarray:
        """Return the lower bound of the output space (same as input lower bound)."""
        return self.lb_input

    @property
    def ub(self) -> float | np.ndarray:
        """Return the upper bound of the output space (same as input upper bound)."""
        return self.ub_input

    def _transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Return the input value(s) as output without modification.

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be transformed.

        Returns
        -------
        float or np.ndarray
            The transformed output value(s) (same as input).
        """
        return x

    def _untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Return the output value(s) as input without modification.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) to be untransformed.

        Returns
        -------
        float or np.ndarray
            The untransformed input value(s) (same as output).
        """
        return x


class NormLinearTransformer(TransformerBase):
    """
    A transformer that normalizes values linearly to the range [0, 1].

    This transformation scales the input value between the given lower
    and upper bounds into a normalized range of [0,1].

    See Also
    --------
    TransformerBase : The base class for parameter transformation.
    """

    @property
    def is_linear(self) -> bool:
        """Return True, as this is a linear transformation."""
        return True

    @property
    def lb(self) -> float:
        """Return the lower bound of the output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Return the upper bound of the output space (1)."""
        return 1.0

    def _transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize input values to the range [0,1].

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be transformed.

        Returns
        -------
        float or np.ndarray
            The transformed output value(s) in the range [0,1].
        """
        return (x - self.lb_input) / (self.ub_input - self.lb_input)

    def _untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Denormalize output values back to the original range.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the normalized range [0,1].

        Returns
        -------
        float or np.ndarray
            The untransformed input value(s) in the original range.
        """
        return (self.ub_input - self.lb_input) * x + self.lb_input


class NormLogTransformer(TransformerBase):
    """
    A transformer that normalizes values logarithmically to the range [0, 1].

    This transformation scales input values logarithmically between the given lower
    and upper bounds into a normalized range of [0,1].

    See Also
    --------
    TransformerBase : The base class for parameter transformation.
    """

    @property
    def is_linear(self) -> bool:
        """Return False, as this is a non-linear transformation."""
        return False

    @property
    def lb(self) -> float:
        """Return the lower bound of the output space (0)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Return the upper bound of the output space (1)."""
        return 1.0

    def _transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize input values to the range [0,1] using a logarithmic transformation.

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be transformed.

        Returns
        -------
        float or np.ndarray
            The transformed output value(s) in the range [0,1].

        Raises
        ------
        ValueError
            If `lb_input` is non-positive, the transformation shifts all values accordingly.
        """
        if self.lb_input <= 0:
            x_ = x + (abs(self.lb_input) + 1)
            ub = 1 + (self.ub_input - self.lb_input)
            return np.log(x_) / np.log(ub)
        else:
            return np.log(x / self.lb_input) / np.log(self.ub_input / self.lb_input)

    def _untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Denormalize output values back to the original range using logarithmic inverse.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the normalized range [0,1].

        Returns
        -------
        float or np.ndarray
            The untransformed input value(s) in the original range.
        """
        if self.lb_input <= 0:
            return (
                np.exp(x * np.log(self.ub_input - self.lb_input + 1))
                + self.lb_input
                - 1
            )
        else:
            return self.lb_input * np.exp(x * np.log(self.ub_input / self.lb_input))


class AutoTransformer(TransformerBase):
    """
    A transformer that automatically selects between linear and logarithmic transformations.

    Transforms the input value to the range [0, 1] using either
    the :class:`NormLinearTransformer` or the :class:`NormLogTransformer`
    based on the input parameter space.

    Attributes
    ----------
    linear : NormLinearTransformer
        Instance of the linear normalization transform.
    log : NormLogTransformer
        Instance of the logarithmic normalization transform.
    threshold : int
        The maximum threshold to switch from linear to logarithmic transformation.

    See Also
    --------
    TransformerBase
    NormLinearTransformer
    NormLogTransformer
    """

    def __init__(self, *args: Any, threshold: int = 100, **kwargs: Any) -> None:
        """
        Initialize an AutoTransformer object.

        Parameters
        ----------
        *args : tuple
            Arguments for the :class:`TransformerBase` class.
        threshold : int, optional
            The maximum threshold to switch from linear to logarithmic
            transformation. The default is 100.
        **kwargs : dict
            Keyword arguments for the :class:`TransformerBase` class.
        """
        self.linear = NormLinearTransformer()
        self.log = NormLogTransformer()

        super().__init__(*args, **kwargs)
        self.threshold = threshold

        self.linear.allow_extended_input = self.allow_extended_input
        self.linear.allow_extended_output = self.allow_extended_output
        self.log.allow_extended_input = self.allow_extended_input
        self.log.allow_extended_output = self.allow_extended_output

    @property
    def is_linear(self) -> bool:
        """Return True if linear transformation is used, otherwise False."""
        return self.use_linear

    @property
    def use_linear(self) -> bool:
        """Determine whether linear transformation should be used."""
        if self.lb_input <= 0:
            return np.log10(self.ub_input - self.lb_input) < np.log10(self.threshold)
        return (self.ub_input / self.lb_input) < self.threshold

    @property
    def use_log(self) -> bool:
        """Return True if logarithmic transformation is used, otherwise False."""
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

    def _transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Transform the input value to an output value in the range [0, 1].

        Parameters
        ----------
        x : float or np.ndarray
            The input value(s) to be transformed.

        Returns
        -------
        float or np.ndarray
            The transformed output value(s).
        """
        return self.log._transform(x) if self.use_log else self.linear._transform(x)

    def _untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Untransform the output value back to the input parameter space.

        Parameters
        ----------
        x : float or np.ndarray
            The output value(s) in the transformed range.

        Returns
        -------
        float or np.ndarray
            The untransformed input value(s).
        """
        return self.log._untransform(x) if self.use_log else self.linear._untransform(x)
