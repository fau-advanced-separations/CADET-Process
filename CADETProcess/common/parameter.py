import math
import operator

import numpy as np

from CADETProcess.common import Descriptor

class Parameter(Descriptor):
    """Parameter class for setting default values.

    This class initializes default values for given parameters, when an object
    is instantiated.

    Parameters
    ----------
    default : NoneType
        Default value of the objcet with None as default.
    unit : NoneType
        unit object with None as default.
    description : NoneType
        Description of the object with None as default.

    See also
    --------
    Descriptor
    StructMeta
    """
    def __init__(self, *args, default=None, unit=None, description=None, **kwargs):
        self.default = default
        self.unit = unit
        self.description = description
        super().__init__(*args, **kwargs)

    def __get__(self, instance, cls):
        try:
            return super().__get__(instance, cls)
        except KeyError:
            return self.default
        
    def __delete__(self, instance):
        if self.default is not None:
            instance.__dict__[self.name] = self.default
        else:
            del instance.__dict__[self.name]


class Typed(Parameter):
    ty = object

    def __set__(self, instance, value):
        if not isinstance(value, self.ty):
            raise TypeError("Expected {}".format(self.ty))
        super().__set__(instance, value)

class Bool(Typed):
    ty = bool

class Integer(Typed):
    ty = int
    
class Tuple(Typed):
    ty = tuple

class Float(Typed):
    ty = float

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = float(value)
        super().__set__(instance, value)

class String(Typed):
    ty = str

class List(Typed):
    ty = list

class Dict(Typed):
    """
    Note
    ----
    !!! Name might collide with addict.Dict!
    """
    ty = dict

class NdArray(Typed):
    ty = np.ndarray


class Ranged(Parameter):
    """Base class for Parameters with value bounds
    """
    def __init__(self, *args, lb=-math.inf, lb_op=operator.lt,
                 ub=math.inf, ub_op=operator.gt, **kwargs):
        self.lb = lb
        self.lb_op = lb_op
        self.ub = ub
        self.ub_op = ub_op

        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if isinstance(self, List):
            if any(self.lb_op(i, self.lb) for i in value):
                raise ValueError("Value exceeds lower bound")
            elif any(self.ub_op(i, self.ub) for i in value):
                raise ValueError("Value exceeds upper bound")

        # NdArray type
        elif isinstance(self, NdArray):
            if self.lb_op(value.any(), self.lb):
                raise ValueError("Value exceeds lower bound")
            elif self.ub_op(value.any(), self.ub):
                raise ValueError("Value exceeds upper bound")

        # Rest
        else:
            if self.lb_op(value, self.lb):
                raise ValueError("Value exceeds lower bound")
            elif self.ub_op(value, self.ub):
                raise ValueError("Value exceeds upper bound")

        super().__set__(instance, value)

class Unsigned(Ranged):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lb=0, lb_op=operator.lt, **kwargs)

class UnsignedInteger(Integer, Unsigned):
    pass

class UnsignedFloat(Float, Unsigned):
    pass

class Sized(Parameter):
    def __init__(self, *args, maxlen, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if len(value) > self.maxlen:
            raise ValueError("Too big!")
        super().__set__(instance, value)

class SizedString(String, Sized):
    pass

class DependentlySized(Parameter):
    """Class for checking the correct shape of Parameters with multiple entries
    that is dependent on other Parameters
    """
    def __init__(self, *args, dep, **kwargs):
        self.dep = dep
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        expected_size = getattr(instance, self.dep)
        if len(value) != expected_size:
                raise ValueError("Expected size {}".format(expected_size))
        super().__set__(instance, value)

class DependentlySizedString(String, DependentlySized):
    pass

class DependentlySizedList(List, DependentlySized):
    pass

class DependentlySizedUnsignedList(List, Unsigned, DependentlySized):
    pass

class DependentlySizedUnsignedNdArray(NdArray, Unsigned, DependentlySized):
    pass

class Switch(Parameter):
    """Class for selecting one entry from a list
    """
    def __init__(self, *args, valid, **kwargs):
        if not isinstance(valid, list):
            raise TypeError("Expected a list for valid entries")
        self.valid = valid

        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if value not in self.valid:
            raise ValueError("Value has to be in {}".format(self.valid))

        super().__set__(instance, value)
