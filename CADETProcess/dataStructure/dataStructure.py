from abc import ABC
from collections import OrderedDict
from inspect import Parameter, Signature
from functools import wraps


def make_signature(names):
    return Signature(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in names)


class StructMeta(type):
    """Base class for classes that use Descriptors.

    See Also
    --------
    Descriptor
    Parameters

    """
    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(cls, clsname, bases, clsdict):
        descriptors = [
            key for key, val in clsdict.items()
            if isinstance(val, Descriptor)
        ]

        for name in descriptors:
            clsdict[name].name = name

        clsdict['descriptors'] = descriptors

        clsobj = super().__new__(cls, clsname, bases, dict(clsdict))

        parameters = []

        try:
            parameters += clsobj._parameters
        except AttributeError:
            pass

        # Collect parameters and descriptors from parent classes
        for base in bases:
            try:
                parameters += base._parameters
            except AttributeError:
                pass
            try:
                descriptors += base.descriptors
            except AttributeError:
                pass

        # Register all parameters
        if len(parameters) > 1:
            clsobj._parameters = parameters

        # Register descriptor fields as arguments in __init__.
        # The order matters here, the list(dict.fromkeys(descriptors)) procecure.
        args = list(dict.fromkeys(descriptors))
        sig = make_signature(args)
        setattr(clsobj, '__signature__', sig)

        return clsobj


class Descriptor(ABC):
    """Base class for descriptors.

    Descriptors are used to efficiently implement class attributes that
    require checking type, value, size etc.
    For using Descriptors, a class must inherit from StructMeta.

     - ``self`` is the Descriptor managing the attribute of the ``instance``.
     - ``instance`` is the object which holds the actual ``value``.
     - ``value`` is the value of the ``instance`` attribute.

    See Also
    --------
    StructMeta
    Parameters

    """

    def __init__(self, *args, **kwargs):
        pass

    def __get__(self, instance, cls):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value is None:
            try:
                del instance.__dict__[self.name]
            except KeyError:
                pass

            return

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
                f"{cls.__name__} object has no attribute {key}"
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
