"""
=========================================
Transform (:mod:`CADETProcess.transform`)
=========================================

.. currentmodule:: CADETProcess.transform

This module provides functionality for transforming data.


.. autosummary::
    :toctree: generated/

    TransformBase
    NoTransform
    NormLinearTransform
    NormLogTransform
    AutoTransform

"""

from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class TransformBase(ABC):
    """
    Base class for parameter transformation.

    This class provides an interface for transforming an input parameter space to some
    output parameter space.

    Attributes
    ----------
    lb_input : {float, array-like}
        Lower bounds of the input parameter space.
    ub_input : {float, array-like}
        Upper bounds of the input parameter space.
    lb : {float, array-like}
        Lower bounds of the output parameter space.
    ub : {float, array-like}
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
    - The `transform` method is not implemented in this class and must be implemented by a subclass.

    Examples
    --------
    >>> class MyTransform(TransformBase):
    ...     def transform(self, x):
    ...         return x ** 2
    ...
    >>> t = MyTransform(lb_input=0, ub_input=10, lb=-100, ub=100)
    >>> t.transform(3)
    9

    """

    def __init__(
            self,
            lb_input=-np.inf, ub_input=np.inf,
            allow_extended_input=False, allow_extended_output=False):
        """Initialize TransformBase

        Parameters
        ----------
        lb_input : {float, array-like}, optional
            Lower bounds of the input parameter space. The default is -inf.
        ub_input : {float, array-like}, optional
            Upper bounds of the input parameter space. The default is inf.
        allow_extended_input : bool, optional
            If True, the input value may exceed the lower/upper bounds.
            Else, an exception is thrown.
            The default is False.
        allow_extended_output : bool, optional
            If True, the output value may exceed the lower/upper bounds.
            Else, an exception is thrown.
            The default is False.
        """
        self.lb_input = lb_input
        self.ub_input = ub_input
        self.allow_extended_input = allow_extended_input
        self.allow_extended_output = allow_extended_output

    @property
    def lb_input(self):
        """{float, array-like}: The lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input):
        self._lb_input = lb_input

    @property
    def ub_input(self):
        """{float, array-like}: The upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input):
        self._ub_input = ub_input

    @abstractproperty
    def lb(self):
        """{float, array-like}: The lower bounds of the output parameter space.

        Must be implemented in the child class.
        """
        pass

    @abstractproperty
    def ub(self):
        """{float, array-like}: The upper bounds of the output parameter space.

        Must be implemented in the child class.
        """
        pass

    def transform(self, x):
        """Transform the input parameter space to the output parameter space.

        Applies the transformation function _transform to x after performing input
        bounds checking. If the transformed value exceeds the output bounds, an error
        is raised.

        Parameters
        ----------
        x : {float, array}
            Input parameter values.

        Returns
        -------
        {float, array}
            Transformed parameter values.
        """
        if (
                not self.allow_extended_input and
                not np.all((self.lb_input <= x) * (x <= self.ub_input))):
            raise ValueError("Value exceeds input bounds.")
        x = self._transform(x)
        if (
                not self.allow_extended_output and
                not np.all((self.lb <= x) * (x <= self.ub))):
            raise ValueError("Value exceeds output bounds.")

        return x

    @abstractmethod
    def _transform(self, x):
        """Transform the input parameter space to the output parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : {float, array}
            Input parameter values.

        Returns
        -------
        {float, array}
            Transformed parameter values.
        """
        pass

    def untransform(self, x):
        """Transform the output parameter space to the input parameter space.

        Applies the transformation function _untransform to x after performing output
        bounds checking. If the transformed value exceeds the input bounds, an error
        is raised.

        Parameters
        ----------
        x : {float, array}
            Output parameter values.

        Returns
        -------
        {float, array}
            Transformed parameter values.
        """
        if (
                not self.allow_extended_output and
                not np.all((self.lb <= x) * (x <= self.ub))):
            raise ValueError("Value exceeds output bounds.")
        x = self._untransform(x)
        if (
                not self.allow_extended_input and
                not np.all((self.lb_input <= x) * (x <= self.ub_input))):
            raise ValueError("Value exceeds input bounds.")

        return x

    @abstractmethod
    def _untransform(self, x):
        """Transform the output parameter space to the input parameter space.

        Must be implemented in the child class.

        Parameters
        ----------
        x : {float, array}
            Output parameter values.

        Returns
        -------
        {float, array}
            Transformed parameter values.
        """
        pass

    def __str__(self):
        """Return the class name as a string."""
        return self.__class__.__name__


class NoTransform(TransformBase):
    """A class that implements no transformation.

    Returns the input values without any transformation.

    See Also
    --------
    TransformBase : The base class for parameter transformation.
    """

    @property
    def lb(self):
        """{float, array-like}: The lower bounds of the output parameter space."""
        return self.lb_input

    @property
    def ub(self):
        """{float, array-like}: The upper bounds of the output parameter space."""
        return self.ub_input

    def _transform(self, x):
        """Transform the input value to output value.

        Parameters
        ----------
        x : {float, array-like}
            The input value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The transformed output value(s).
        """
        return x

    def _untransform(self, x):
        """Untransform the output value to input value.

        Parameters
        ----------
        x : {float, array-like}
            The output value(s) to be untransformed.

        Returns
        -------
        {float, array-like}
            The untransformed input value(s).
        """
        return x


class NormLinearTransform(TransformBase):
    """A class that implements a normalized linear transformation.

    Transforms the input value to the range [0, 1] by normalizing it using
    the lower and upper bounds of the input parameter space.

    See Also
    --------
    TransformBase : The base class for parameter transformation.

    """

    @property
    def lb(self):
        """{float, array-like}: The lower bounds of the output parameter space."""
        return 0

    @property
    def ub(self):
        """{float, array-like}: The upper bounds of the output parameter space."""
        return 1

    def _transform(self, x):
        """Transform the input value to output value.

        Parameters
        ----------
        x : {float, array-like}
            The input value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The transformed output value(s).
        """
        return (x - self.lb_input) / (self.ub_input - self.lb_input)

    def _untransform(self, x):
        """Untransform the output value to input value.

        Parameters
        ----------
        x : {float, array-like}
            The output value(s) to be untransformed.

        Returns
        -------
        {float, array-like}
            The untransformed input value(s).
        """
        return (self.ub_input - self.lb_input) * x + self.lb_input


class NormLogTransform(TransformBase):
    """A class that implements a normalized logarithmic transformation.

    Transforms the input value to the range [0, 1] using a logarithmic
    transformation with the lower and upper bounds of the input parameter space.

    See Also
    --------
    TransformBase : The base class for parameter transformation.

    """

    @property
    def lb(self):
        """{float, array-like}: The lower bounds of the output parameter space."""
        return 0

    @property
    def ub(self):
        """{float, array-like}: The upper bounds of the output parameter space."""
        return 1

    def _transform(self, x):
        """Transform the input value to output value.

        Parameters
        ----------
        x : {float, array-like}
            The input value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The transformed output value(s).
        """
        if self.lb_input <= 0:
            x_ = x + (abs(self.lb_input) + 1)
            ub = 1 + (self.ub_input - self.lb_input)
            return np.log(x_) / (np.log(ub))

        else:
            return \
                np.log(x/self.lb_input) / np.log(self.ub_input/self.lb_input)

    def _untransform(self, x):
        """Transform the input value to output value.

        Parameters
        ----------
        x : {float, array-like}
            The input value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The transformed output value(s).
        """
        if self.lb_input < 0:
            return \
                np.exp(x * np.log(self._ub - self.lb_input + 1)) \
                + self.lb_input - 1
        else:
            return \
                self.lb_input * np.exp(x * np.log(self.ub_input/self.lb_input))


class AutoTransform(TransformBase):
    """A class that implements an automatic parameter transformation.

    Transforms the input value to the range [0, 1] using either
    the :class:`NormLinearTransform` or the :class:`NormLogTransform`
    based on the input parameter space.

    Attributes
    ----------
    linear : :class:`NormLinearTransform`
        Instance of the linear normalization transform.
    log : :class:`NormLogTransform`
        Instance of the logarithmic normalization transform.

    See Also
    --------
    TransformBase
    NormLinearTransform
    NormLogTransform

    """

    def __init__(self, *args, threshold=1000, **kwargs):
        """Initialize an AutoTransform object.

        Parameters
        ----------
        *args : tuple
            Arguments for the :class:`TransformBase` class.
        threshold : int, optional
            The maximum threshold to switch from linear to logarithmic
            transformation. The default is 1000.
        **kwargs : dict
            Keyword arguments for the :class:`TransformBase` class.
        """
        self.linear = NormLinearTransform()
        self.log = NormLogTransform()

        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.linear.allow_extended_input = self.allow_extended_input
        self.linear.allow_extended_output = self.allow_extended_input
        self.log.allow_extended_input = self.allow_extended_input
        self.log.allow_extended_output = self.allow_extended_input

    @property
    def use_linear(self):
        """bool: Indicates whether linear transformation is used."""
        if self.lb_input <= 0:
            return \
                np.log10(self.ub_input - self.lb_input) \
                <= np.log10(self.threshold)
        else:
            return self.ub_input/self.lb_input <= self.threshold

    @property
    def use_log(self):
        """bool: Indicates whether logarithmic transformation is used."""
        return not self.use_linear

    @property
    def lb(self):
        """{float, array-like}: The lower bounds of the output parameter space."""
        return 0

    @property
    def ub(self):
        """{float, array-like}: The upper bounds of the output parameter space."""
        return 1

    @property
    def lb_input(self):
        """{float, array-like}: The lower bounds of the input parameter space."""
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input):
        self.linear.lb_input = lb_input
        self.log.lb_input = lb_input
        self._lb_input = lb_input

    @property
    def ub_input(self):
        """{float, array-like}: The upper bounds of the input parameter space."""
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input):
        self.linear.ub_input = ub_input
        self.log.ub_input = ub_input
        self._ub_input = ub_input

    def _transform(self, x):
        """Transform the input value to output value.

        Parameters
        ----------
        x : {float, array-like}
            The input value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The transformed output value(s).
        """
        if self.use_log:
            return self.log.transform(x)
        else:
            return self.linear.transform(x)

    def _untransform(self, x):
        """Untransforms the output value to input value.

        Parameters
        ----------
        x : {float, array-like}
            The output value(s) to be transformed.

        Returns
        -------
        {float, array-like}
            The untransformed output value(s).
        """
        if self.use_log:
            return self.log.untransform(x)
        else:
            return self.linear.untransform(x)
