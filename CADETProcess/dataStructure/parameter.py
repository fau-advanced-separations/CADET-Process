from abc import abstractmethod
import math
import operator

import numpy as np

from CADETProcess import CADETProcessError

from .dataStructure import Descriptor

class Parameter(Descriptor):
    """Class for defining model parameters.

    Parameters
    ----------
    default : NoneType
        Default value of the object with None as default.
    unit : str
        Units of the parameter.
    description : str
        Description of the parameter.

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
    
    @property
    def default(self):
        return self._default
    
    @default.setter
    def default(self, value):
        if value is not None:
            self._check(value, recursive=True)
        self._default = value
        
    def __get__(self, instance, cls):
        """ !!!TODO!!! Consider raising ValueError if Parameter not set"""
        try:
            return Descriptor.__get__(self, instance, cls)
        except KeyError:
            return self.default
    
    def __set__(self, instance, value):
        super().__set__(instance, value)
        try:
            if self.name in instance._parameter_names:
                instance._parameters[self.name] = value
        except AttributeError:
            pass
        
    def _check(self, value, recursive=False):
        return True


class Callable(Parameter):
    def __set__(self, instance, value):
        if value is None:
            try:
                del(instance.__dict__[self.name])
            except KeyError:
                pass
            return
        
        if not callable(value):
            raise TypeError("Expected obect to be callable")
        
        super().__set__(instance, value)
    

class Typed(Parameter):
    ty = object

    def __set__(self, instance, value):
        if value is None:
            try:
                del(instance.__dict__[self.name])
            except KeyError:
                pass
            return
        
        if Typed._check(self, value):
            super().__set__(instance, value)
        
    def _check(self, value, recursive=False):
        if not isinstance(value, self.ty):
            raise TypeError("Expected {}".format(self.ty))
        
        if recursive:
            return super()._check(value, recursive)
        
        return True
    
class Container(Typed):
    @abstractmethod
    def check_content_range():
        return

    @abstractmethod
    def check_content_size():
        return
    
    @abstractmethod
    def get_default_values():
        return
    
    def _check(self, value, recursive=False):
        if isinstance(value, (int, float)):
            value = self.ty((value,))
            
        if recursive:
            return super()._check(value, recursive)
        
        return True
    

class Bool(Typed):
    ty = bool

class Integer(Typed):
    ty = int
    
class Tuple(Typed):
    ty = tuple

class Float(Typed):
    """Cast ints to float"""
    ty = float

    def __set__(self, instance, value):
        if isinstance(value, (int, np.number)):
            value = float(value)
        
        super().__set__(instance, value)
        
    def _check(self, value, recursive=False):
        if isinstance(value, int):
            value = float(value)
            
        if recursive:
            return super()._check(value, recursive)
        
        return True

class String(Typed):
    ty = str

class List(Container):
    ty = list
    
    def check_content_range(self, value):
        if any(self.lb_op(i, self.lb) for i in value):
            raise ValueError("Value exceeds lower bound")
        elif any(self.ub_op(i, self.ub) for i in value):
            raise ValueError("Value exceeds upper bound")
        
    def check_content_size(self, instance, value):
        shape = [getattr(instance, dep) for dep in self.dep]
        expected_length = np.prod(shape)
        
        if len(value) != expected_length:
            raise ValueError("Expected size {}".format(expected_length))
        
    def get_default_values(self, instance):
        if super().default is None:
            return None
        
        shape = [getattr(instance, dep) for dep in self.dep]
        
        return np.prod(shape) * [super().default]
    

class Dict(Typed):
    """
    Note
    ----
    !!! Name might collide with addict.Dict!
    """
    ty = dict
    
class NdArray(Container):
    ty = np.ndarray
    
    def check_content_range(self, value):
        if np.any(self.lb_op(value,self.lb)):
            raise ValueError("Value exceeds lower bound")
        elif np.any(self.ub_op(value,self.ub)):
            raise ValueError("Value exceeds upper bound")
    
    def get_expected_shape(self, instance):
        expected_shape = []
        
        for dep in self.dep:
            if isinstance(dep, str):
                dim = getattr(instance, dep)
                if isinstance(dim, np.ndarray):
                    dim = list(dim.shape)
                elif isinstance(dim, int):
                    dim = [dim]
            elif isinstance(dep, int):
                dim = [dep]
            else:
                raise AttributeError("Cannot find dependency")
            expected_shape += dim
        
        expected_shape = tuple(expected_shape)
        
        return expected_shape
        
    def check_content_size(self, instance, value):
        expected_shape = self.get_expected_shape(instance) 

        if value.shape != expected_shape:
            raise ValueError("Expected shape {}".format(expected_shape))
        
    def get_default_values(self, instance):
        expected_shape = self.get_expected_shape(instance) 
        
        return super().default * np.ones(expected_shape)

class Dimensionalized(Parameter):
    """
    Only works for NdArrays
    """
    n_dim = None
    def __init__(self, *args, **kwargs):
        if not isinstance(self, NdArray):
            raise Exception('Only NdArrays can have dimensions.')

        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if value is None:
            del(instance.__dict__[self.name])
            return
        
        if self.n_dim != value.ndim:
            raise ValueError("Expected dimensions {}".format(self.n_dim))
        super().__set__(instance, value)

class Vector(NdArray, Dimensionalized):
    n_dim = 1

class Matrix(NdArray, Dimensionalized):
    n_dim = 2
    
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
        if value is None:
            del(instance.__dict__[self.name])
            return

        if Ranged._check(self, value):
            super().__set__(instance, value)
    
    def _check(self, value, recursive=False):
        if isinstance(self, Container):
            self.check_content_range(value)
        else:
            if self.lb_op(value, self.lb):
                raise ValueError("Value exceeds lower bound")
            elif self.ub_op(value, self.ub):
                raise ValueError("Value exceeds upper bound")

        if recursive:
            return super()._check(value, recursive)
        
        return True
        
class RangedFloat(Float, Ranged):
    pass

class RangedInteger(Integer, Ranged):
    pass
            
class Unsigned(Ranged):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lb=0, lb_op=operator.lt, **kwargs)

class UnsignedInteger(Integer, Unsigned):
    pass

class UnsignedFloat(Float, Unsigned):
    pass

class Sized(Parameter):
    def __init__(self, *args, minlen, maxlen, **kwargs):
        self.minlen = minlen
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if value is None:
            del(instance.__dict__[self.name])
            return
            
        if len(value) > self.maxlen:
            raise ValueError("Too big!")
        if len(value) < self.minlen:
            raise ValueError("Too small!")
        super().__set__(instance, value)
    
class SizedString(String, Sized):
    pass

class SizedTuple(Tuple, Sized):
    pass

class SizedNdArray(NdArray, Sized):
    pass

class DependentlySized(Parameter):
    """Class for checking the correct shape of Parameters with multiple entries
    that is dependent on other Parameters
    """
    def __init__(self, *args, dep, **kwargs):
        if not isinstance(dep, tuple):
            dep = (dep,)
                          
        self.dep = dep
        super().__init__(*args, **kwargs)
        

    def __set__(self, instance, value):
        if value is None:
            del(instance.__dict__[self.name])
            return
        
        if isinstance(self, Container):
            self.check_content_size(instance, value)
        else:
            size = getattr(instance, self.dep)
            if len(value) != size:
                raise ValueError("Expected size {}".format(size))
        
        super().__set__(instance, value)

    def __get__(self, instance, cls):
        if not isinstance(self, Container):
            return super().__get__(instance, cls)
        
        try:
            return Descriptor.__get__(self, instance, cls)
        except KeyError:
            return self.get_default_values(instance)
        
class DependentlySizedString(String, DependentlySized):
    pass

class DependentlySizedList(List, DependentlySized):
    pass

class DependentlySizedRangedList(List, Ranged, DependentlySized):
    pass

class DependentlySizedUnsignedList(List, Unsigned, DependentlySized):
    pass

class DependentlySizedNdArray(NdArray, DependentlySized):
    pass

class DependentlySizedUnsignedNdArray(NdArray, Unsigned, DependentlySized):
    pass

class Polynomial(DependentlySizedNdArray):
    def __init__(self, *args, n_coeff=None, **kwargs):
        if n_coeff is not None:
            if 'dep' in kwargs:
                raise ValueError("Duplicate entries for n_coeff")
            kwargs['dep'] = (n_coeff,)
        
        kwargs['default'] = 0
            
        super().__init__(*args, **kwargs)
    
    def __set__(self, instance, value):
        shape = self.get_expected_shape(instance)
        degree = shape[0]

        if value is None:
            del(instance.__dict__[self.name])
            return
    
        if isinstance(value, (int, float)):
            value = np.array((value))
                
        if isinstance(value, (tuple, list)):
            missing = degree - len(value)
            value += missing*(0,)

        value = np.array(value, ndmin=1)
    
        _value = self.get_default_values(instance)
        _value[0:value.shape[0]] = value
       
        super().__set__(instance, _value)

class NdPolynomial(DependentlySizedNdArray):
    """Dependently sized polynomial for n entries.
    
    Important: Use [entries x n_coeff] for dependencies.
    """
    def __init__(self, *args, n_entries=None, n_coeff=None, **kwargs):
        try:
            dep = kwargs['dep']
            if not isinstance(dep, tuple):
                dep = (dep,)
        except KeyError:
            dep = (dep,)
            pass
        
        if n_entries is None and n_coeff is None and len(dep) < 2:
            raise ValueError("Missing entries for shape")
            
        if n_entries is not None:
            if len(dep) > 1:
                raise ValueError("Found duplicate n_entries for shape")
            _n_entries = n_entries
            if n_coeff is not None:
                if len(dep) > 0:
                    raise ValueError("Found duplicate n_entries for shape")
        else:
            _n_entries = dep[0]
                
        if n_coeff is not None:
            if len(dep) > 1:
                raise ValueError("Found duplicate n_entries for shape")
            _n_coeff = n_coeff
            if n_entries is not None:
                if len(dep) > 0:
                    raise ValueError("Found duplicate n_entries for shape")
        else:
            if n_entries is not None:
                _n_coeff = dep[0]
            else:
                _n_coeff = dep[1]
            

        dep = (_n_entries, _n_coeff)
            
        kwargs['dep'] = dep
        kwargs['default'] = 0
        
        super().__init__(*args, **kwargs)
        
    def __set__(self, instance, value):
        n_entries, n_coeff = self.get_expected_shape(instance)
            
        if value is None:
            del(instance.__dict__[self.name])
            return

        if isinstance(value, (int, float)):
            value = n_entries * [value]
                
        if len(value) != n_entries:
            raise ValueError("Number of entries does not match")
        
        _value = self.get_default_values(instance)
        
        for i, v in enumerate(value):
            if isinstance(v, (int, float, np.number)):
                v = [v]
            if isinstance(v, (list, tuple)):
                missing = n_coeff - len(v)
                v += missing*(0,)
            _value[i,:] = np.array(v)
       
        super().__set__(instance, _value)

    
class Switch(Parameter):
    """Class for selecting one entry from a list
    """
    def __init__(self, *args, valid, **kwargs):
        if not isinstance(valid, list):
            raise TypeError("Expected a list for valid entries")
        self.valid = valid

        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if Switch._check(self, value):
            super().__set__(instance, value)
        
    def _check(self, value, recursive=False):
        if value not in self.valid:
            raise ValueError("Value has to be in {}".format(self.valid))

        if recursive:
            return super()._check(value, recursive)
        
        return True
