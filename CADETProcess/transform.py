from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class TransformBase(ABC):
    """Base Class for parameter transformation.

    Maps input parameter space to some output parameter space.

    Attributes
    ----------
    lb_input : {float, array}
        Lower bounds of input parameter space.
    ub_input : {float, array}
        Upper bounds of input parameter space.
    lb: {float, array}
        Lower bounds of output parameter space.
    ub : {float, array}
        Upper bounds of output parameter space.
    allow_extended_input : bool
        If True, the input value may exceed the lower/upper bounds.
        Else, an exception is thrown.
        The default is False.
    allow_extended_output : bool
        If True, the output value may exceed the lower/upper bounds.
        Else, an exception is thrown.
        The default is False.
    """

    def __init__(
            self,
            lb_input=-np.inf, ub_input=np.inf,
            allow_extended_input=False, allow_extended_output=False):
        self.lb_input = lb_input
        self.ub_input = ub_input
        self.allow_extended_input = allow_extended_input
        self.allow_extended_output = allow_extended_output

    @property
    def lb_input(self):
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input):
        self._lb_input = lb_input

    @property
    def ub_input(self):
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input):
        self._ub_input = ub_input

    @abstractproperty
    def lb(self):
        pass

    @abstractproperty
    def ub(self):
        pass

    def transform(self, x):
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
        pass

    def untransform(self, x):
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
        pass

    def __str__(self):
        return self.__class__.__name__


class NoTransform(TransformBase):
    @property
    def lb(self):
        return self.lb_input

    @property
    def ub(self):
        return self.ub_input

    def _transform(self, x):
        return x

    def _untransform(self, x):
        return x


class NormLinearTransform(TransformBase):
    @property
    def lb(self):
        return 0

    @property
    def ub(self):
        return 1

    def _transform(self, x):
        return (x - self.lb_input) / (self.ub_input - self.lb_input)

    def _untransform(self, x):
        return (self.ub_input - self.lb_input) * x + self.lb_input


class NormLogTransform(TransformBase):
    @property
    def lb(self):
        return 0

    @property
    def ub(self):
        return 1

    def _transform(self, x):
        if self.lb_input <= 0:
            x_ = x + (abs(self.lb_input) + 1)
            ub = 1 + (self.ub_input - self.lb_input)
            return np.log(x_) / (np.log(ub))

        else:
            return \
                np.log(x/self.lb_input) / np.log(self.ub_input/self.lb_input)

    def _untransform(self, x):
        if self.lb_input < 0:
            return \
                np.exp(x * np.log(self._ub - self.lb_input + 1)) \
                + self.lb_input - 1
        else:
            return \
                self.lb_input * np.exp(x * np.log(self.ub_input/self.lb_input))


class AutoTransform(TransformBase):
    def __init__(self, *args, threshold=1000, **kwargs):
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
        if self.lb_input <= 0:
            return \
                np.log10(self.ub_input - self.lb_input) \
                <= np.log10(self.threshold)
        else:
            return self.ub_input/self.lb_input <= self.threshold

    @property
    def use_log(self):
        return not self.use_linear

    @property
    def lb(self):
        return 0

    @property
    def ub(self):
        return 1

    @property
    def lb_input(self):
        return self._lb_input

    @lb_input.setter
    def lb_input(self, lb_input):
        self.linear.lb_input = lb_input
        self.log.lb_input = lb_input
        self._lb_input = lb_input

    @property
    def ub_input(self):
        return self._ub_input

    @ub_input.setter
    def ub_input(self, ub_input):
        self.linear.ub_input = ub_input
        self.log.ub_input = ub_input
        self._ub_input = ub_input

    def _transform(self, x):
        if self.use_log:
            return self.log.transform(x)
        else:
            return self.linear.transform(x)

    def _untransform(self, x):
        if self.use_log:
            return self.log.untransform(x)
        else:
            return self.linear.untransform(x)
