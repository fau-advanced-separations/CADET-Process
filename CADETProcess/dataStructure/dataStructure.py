from abc import ABCMeta
from collections import OrderedDict
from inspect import Parameter, Signature
from functools import wraps

def make_signature(names):
    return Signature(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in names)


class StructMeta(type):
    """Base class for classes that use Descriptors.

    See also
    Descriptor
    Parameters
    """
    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(cls, clsname, bases, clsdict):
        fields = [key for key, val in clsdict.items()
                if isinstance(val, Descriptor)]

        for name in fields:
            clsdict[name].name = name
        
        clsdict['descriptors'] = fields

        clsobj = super().__new__(cls, clsname, bases, dict(clsdict))

        sig = make_signature(fields)
        setattr(clsobj, '__signature__', sig)

        return clsobj
    
    
class Descriptor():
    """Base class for descriptors.

    Descriptors are used to efficiently implement class attributes that 
    require checking type, value, size etc. 
    For using Descriptors, a class must inherit from StructMeta.
    
     - ``self`` is the Descriptor managing the attribute of the ``instance``.
     - ``instance`` is the object which holds the actual ``value``.
     - ``value`` is the value of the ``instance`` attribute.

    See also
    --------
    StructMeta
    Parameters
    """
    def __init__(self, *args, **kwargs):
        pass

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

class Structure(metaclass=StructMeta):
    def __init__(self, *args, **kwargs):
        bound = self.__signature__.bind_partial(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, val)

def frozen_attributes(cls):
    """Decorate classes to prevent setting attributes after the init method.
    """
    cls._is_frozen = False

    def frozensetattr(self, key, value):
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(
                "{} object has no attribute {}".format(cls.__name__, key)
            )
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._is_frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls
