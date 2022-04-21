from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class TransformBase(ABC):
    def __init__(self, lb=-np.inf, ub=np.inf):
        self._lb = lb
        self._ub = ub

    @abstractproperty
    def lb(self):
        pass

    @abstractproperty
    def ub(self):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def untransform(self, x):
        pass

    def __str__(self):
        return self.__class__.__name__


class NoTransform(TransformBase):
    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, ub):
        self._ub = ub

    def transform(self, x):
        return x

    def untransform(self, x):
        return x


class AutoTransform(TransformBase):
    @property
    def lb(self):
        return 0

    @lb.setter
    def lb(self, lb):
        self.linear.lb = lb
        self.log.lb = lb
        self._lb = lb

    @property
    def ub(self):
        return 1

    @ub.setter
    def ub(self, ub):
        self.linear.ub = ub
        self.log.ub = ub
        self._ub = ub

    def __init__(self, lb=-np.inf, ub=np.inf, threshold=1000):
        self.linear = NormLinearTransform(lb, ub)
        self.log = NormLogTransform(lb, ub)
        self.max_factor = 1000
        self.lb = lb
        self.ub = ub

    @property
    def use_linear(self):
        if self._lb <= 0:
            return np.log10(self._ub - self._lb) <= np.log10(self.max_factor)
        else:
            return self._ub/self._lb <= self.max_factor

    @property
    def use_log(self):
        return not self.use_linear

    def transform(self, x):
        if self.use_log:
            return self.log.transform(x)
        else:
            return self.linear.transform(x)

    def untransform(self, x):
        if self.use_log:
            return self.log.untransform(x)
        else:
            return self.linear.untransform(x)


class NormLinearTransform(TransformBase):
    @property
    def lb(self):
        return 1

    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        return 0

    @ub.setter
    def ub(self, ub):
        self._ub = ub

    def transform(self, x):
        return (x - self._lb) / (self._ub - self._lb)

    def untransform(self, x):
        return (self._ub - self._lb) * x + self._lb


class NormLogTransform(TransformBase):
    @property
    def lb(self):
        return 1

    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        return 0

    @ub.setter
    def ub(self, ub):
        self._ub = ub

    def transform(self, x):
        if self.lb <= 0:
            x_ = x + (abs(self._lb) + 1)
            ub_ = 1 + (self._ub - self._lb)
            return np.log(x_) / (np.log(ub_))

        else:
            return np.log(x/self._lb) / np.log(self._ub/self._lb)

    def untransform(self, x):
        if self.lb < 0:
            return np.exp(x * np.log(self._ub - self._lb + 1)) + self._lb - 1
        else:
            return self._lb * np.exp(x * np.log(self._ub/self._lb))
