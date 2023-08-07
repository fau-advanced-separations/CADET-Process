from abc import ABC
from collections import OrderedDict
from inspect import Parameter, Signature
from functools import wraps
from warnings import warn

from addict import Dict


# %% Descriptors
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


def make_signature(names):
    return Signature(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in names)


class StructMeta(type):
    """
    Metaclass for creating classes that use Descriptors.

    This metaclass enables classes to have ordered descriptors and provides
    additional functionality related to descriptor management. The underlying
    structure uses an OrderedDict to maintain the order of class attributes.

    The metaclass mainly interacts with the `Descriptor` class, and classes
    that use this metaclass can benefit from this specialized handling of descriptors.

    Attributes
    ----------
    _descriptors : list
        List of descriptors associated with a class.
    _parameters : list
        List of parameters aggregated from the class and its bases.
    _sized_parameters : list
        List of parameters that have a `size` attribute.
    _polynomial_parameters : list
        List of parameters with `fill_values` attribute.
    _required_parameters : list
        List of parameters that have a default value of None.

    See Also
    --------
    Structure : Base class that typically uses this metaclass.
    Descriptor : Class that represents the descriptors this metaclass operates on.
    Parameters : Base class for model parameters with e.g. type or bound constraints.


    Methods
    -------
    __prepare__(name, bases) -> OrderedDict
        Prepares the namespace for the class body to be executed.
    """

    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(cls, clsname, bases, clsdict):
        # Extract descriptor keys
        descriptors = [
            key for key, val in clsdict.items()
            if isinstance(val, Descriptor)
        ]

        # Assign name attribute for each descriptor
        for name in descriptors:
            clsdict[name].name = name

        clsdict['_descriptors'] = descriptors

        # Create the new class object
        clsobj = super().__new__(cls, clsname, bases, dict(clsdict))

        # Aggregate parameters from the current class and its bases
        parameters = []
        try:
            parameters += clsobj._parameters
        except AttributeError:
            pass

        for base in bases:
            base_parameters = getattr(base, '_parameters', [])
            parameters += base_parameters

        setattr(clsobj, '_parameters', parameters)

        # Categorize parameters based on their attributes
        sized_parameters = []
        polynomial_parameters = []
        required_parameters = []

        for param in parameters:
            descriptor = getattr(clsobj, param)

            # Skip if it's not an instance of Descriptor
            if not isinstance(descriptor, Descriptor):
                continue

            if hasattr(descriptor, 'size'):
                sized_parameters.append(param)
            if hasattr(descriptor, 'fill_values'):
                polynomial_parameters.append(param)
            if descriptor.default is None:
                required_parameters.append(param)

        setattr(clsobj, '_sized_parameters', sized_parameters)
        setattr(clsobj, '_polynomial_parameters', polynomial_parameters)
        setattr(clsobj, '_required_parameters', required_parameters)

        # Collect descriptors from base classes
        for base in bases:
            descriptors += getattr(base, '_descriptors', [])

        # Remove duplicates from the descriptors list
        args = list(dict.fromkeys(descriptors))

        # Register descriptor fields as arguments in __init__
        sig = make_signature(args)
        setattr(clsobj, '__signature__', sig)

        return clsobj


# %% Stucture / ParameterHandler
class Structure(metaclass=StructMeta):
    """
    A class representing a structured data entity.

    This class is designed to work in conjunction with the `StructMeta` metaclass
    to handle descriptors and related parameters.

    Attributes
    ----------
    _parameters : dict
        Dictionary of parameters associated with the instance.
    _sized_parameters : list
        List of parameters that have a `size` attribute.
    _polynomial_parameters : list
        List of parameters with `fill_values` attribute.
    _required_parameters : list
        List of parameters that have a default value of None.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Structure instance.

        Parameters are bound to the instance based on the `__signature__`
        defined by the metaclass.

        Parameters
        ----------
        *args
            Positional arguments representing parameters.
        **kwargs
            Keyword arguments representing parameters.
        """
        self._parameters = Dict({
            param: getattr(self, param)
            for param in self._parameters
        })

        bound = self.__signature__.bind_partial(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, val)

    @property
    def parameters(self):
        """dict: Parameters of the instance."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Set parameters for the instance.

        Parameters
        ----------
        parameters : dict
            A dictionary of parameters to set.

        Raises
        ------
        ValueError
            If any of the provided parameters is not valid.
        """
        for param, value in parameters.items():
            if param not in self._parameters:
                raise ValueError('Not a valid parameter.')
            if value is not None:
                setattr(self, param, value)

    @property
    def sized_parameters(self):
        """dict: Sized parameters of the instance."""
        parameters = {
            key: value for key, value in self.parameters.items()
            if key in self._sized_parameters
        }
        return Dict(parameters)

    @property
    def polynomial_parameters(self):
        """dict: Polynomial parameters of the instance."""
        parameters = {
            key: value for key, value in self.parameters.items()
            if key in self._polynomial_parameters
        }
        return Dict(parameters)

    @property
    def required_parameters(self):
        """list: Parameters that have no default value."""
        return self._required_parameters

    @property
    def missing_parameters(self):
        """list: Parameters that are required but not set."""
        missing_parameters = []
        for param in self.required_parameters:
            if getattr(self, param) is None:
                missing_parameters.append(param)

        return missing_parameters

    def check_required_parameters(self):
        """
        Verify if all required parameters are set.

        Returns
        -------
        bool
            True if all required parameters are set. False otherwise.

        Raises
        ------
        Warning
            If any of the required parameters are missing.
        """
        if len(self.missing_parameters) == 0:
            return True
        else:
            for param in self.missing_parameters:
                warn(f'Missing parameter "{param}".')
            return False


def frozen_attributes(cls):
    """Decorate classes to prevent setting attributes after the init method."""
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
