from __future__ import annotations

from abc import ABC, ABCMeta
from collections import OrderedDict
from functools import wraps
from inspect import Parameter, Signature
from typing import Any, Callable, Iterable, Iterator, Optional, Type
from warnings import warn

from addict import Dict


# %% Descriptors
class Descriptor(ABC):
    """
    Base class for descriptors.

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __get__(self, instance: Any, owner: Optional[type[Any]]) -> Any:
        """
        Get the attribute value from the instance's dictionary.

        Parameters
        ----------
        instance : Any
            The instance accessing the attribute.
        owner : Optional[type[Any]]
            The owner class of the descriptor.

        Returns
        -------
        Any
            The attribute value from the instance's dictionary.
        """
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Set the attribute value in the instance's dictionary.

        Parameters
        ----------
        instance : Any
            The instance to set the attribute on.
        value : Any
            The value to set.
        """
        if value is None:
            try:
                del instance.__dict__[self.name]
            except KeyError:
                pass
            return
        instance.__dict__[self.name] = value

    def __delete__(self, instance: Any) -> None:
        """
        Delete the attribute from the instance's dictionary.

        Parameters
        ----------
        instance : Any
            The instance to delete the attribute from.
        """
        del instance.__dict__[self.name]


class ProxyList:
    """A proxy list that dynamically updates attributes of container elements."""

    def __init__(self, aggregator: Aggregator, instance: Any) -> None:
        """Initialize Proxy List."""
        self.aggregator = aggregator
        self.instance = instance

    def _get_values_from_aggregator(self) -> Any:
        """Fetch the latest values from the aggregator."""
        return self.aggregator._get_values_from_container(self.instance, check=True)

    def __getitem__(self, index: int) -> Any:
        """Retrieve an item from the aggregated parameter list (live view)."""
        return self._get_values_from_aggregator()[index]

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Modify an individual element in the aggregated parameter list.

        Ensures changes propagate to the underlying objects.
        """
        current_value = self._get_values_from_aggregator()
        current_value[index] = value
        self.aggregator.__set__(self.instance, current_value)

    def __iter__(self) -> Iterator:
        """Iterate over aggregated values."""
        return iter(self._get_values_from_aggregator())

    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self._get_values_from_aggregator())

    def __repr__(self) -> str:
        """str: String representation for debugging."""
        return f"ProxyList({self._get_values_from_aggregator().__repr__()})"

    def __eq__(self, other: ProxyList) -> bool:
        """Equality comparison."""
        return list(self._get_values_from_aggregator()) == other


class Aggregator:
    """Descriptor aggregating parameters from iterable container of other objects."""

    def __init__(
        self,
        parameter_name: str,
        container: str,
        *args: dict,
        **kwargs: dict,
    ) -> None:
        """
        Initialize the Aggregator.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to be aggregated.
        container : str
            Name of the iterable attribute in the instance that contains the other
            objects from which parameters will be aggregated.
        *args : tuple, optional
            Additional positional arguments.
        **kwargs : dict, optional
            Additional keyword arguments.
        """
        self.parameter_name = parameter_name
        self.container = container

    def _container_obj(self, instance: Any) -> Iterable:
        """
        Retrieve the iterable container of the instance.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the container from.

        Returns
        -------
        obj : iterable
            Iterable container of the instance.

        Raises
        ------
        TypeError
            If the container is not iterable.
        """
        container = getattr(instance, self.container)

        if not hasattr(container, "__iter__"):
            raise TypeError(f"{self.container} attribute is not iterable")

        return container

    def _n_instances(self, instance: Any) -> int:
        return len(self._container_obj(instance))

    def _get_values_from_container(self, instance: Any, check: bool = False) -> Any:
        container = self._container_obj(instance)

        value = [getattr(el, self.parameter_name) for el in container]

        if check:
            value = self._prepare(instance, value, recursive=True)
            self._check(instance, value, recursive=True)

        return value

    def __get__(self, instance: Any, cls: Type) -> ProxyList:
        """
        Retrieve the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        cls : type[Any], optional
            Class to which the descriptor belongs. By default None.

        Returns
        -------
        np.array
            Descriptor values aggregated in a numpy array.
        """
        if instance is None:
            return self

        return ProxyList(self, instance)

    def __set__(self, instance: Any, value: Iterable) -> None:
        """
        Set the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to set the descriptor value for.
        value : Iterable
            Value to set. Note, this assumes that each element of the value maps to
            each element of the container.
        """
        if value is not None:
            value = self._prepare(instance, value, recursive=True)
            self._check(instance, value, recursive=True)

        container = self._container_obj(instance)

        for el, el_value in zip(container, value):
            setattr(el, self.parameter_name, el_value)

    def _prepare(self, instance: Any, value: Any, recursive: bool = False) -> Any:
        """
        Prepare value for setting if necessary.

        Override this method if type-casting or other operations are necessary.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to cast.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False. Only works in
            case of overriding.

        Returns
        -------
        Any
            Prepared value.
        """
        return value

    def _check(self, instance: Any, value: Iterable, recursive: bool = False) -> None:
        """
        Check the given value.

        Override this method for specific checks.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Iterable
            Value to set. Note, this assumes that each element of the value maps to
            each element of the container.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.
        """
        container = self._container_obj(instance)

        if len(value) != len(container):
            raise ValueError(
                f"Unexpected length. Expected {len(container)} entries, got {len(value)}."
            )

        return


def make_signature(names: list[str]) -> Signature:
    """
    Create a signature object from a list of parameter names.

    Parameters
    ----------
    names : list[str]
        List of parameter names.

    Returns
    -------
    Signature
        A Signature object for the given parameter names.
    """
    return Signature(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names)


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
    _aggregated_parameters : list
        List of parameters that aggregate other instances parameters.
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
    def __prepare__(cls, name: str, bases: tuple) -> OrderedDict:
        """
        Prepare the class namespace using an OrderedDict.

        Parameters
        ----------
        name : str
            Name of the class being defined.
        bases : tuple
            Tuple of base classes.

        Returns
        -------
        OrderedDict
            Ordered dictionary for the class namespace.
        """
        return OrderedDict()

    def __new__(cls: type, clsname: str, bases: tuple, clsdict: OrderedDict) -> Any:
        """
        Create a new class with descriptors and aggregators.

        Parameters
        ----------
        cls : type
            The metaclass itself.
        clsname : str
            Name of the class being created.
        bases : tuple
            Tuple of base classes.
        clsdict : OrderedDict
            Ordered dictionary of class attributes.

        Returns
        -------
        Any
            The newly created class object.
        """
        # Extract descriptor keys
        descriptors = [key for key, val in clsdict.items() if isinstance(val, Descriptor)]

        # Assign name attribute for each descriptor
        for name in descriptors:
            clsdict[name].name = name

        clsdict["_descriptors"] = descriptors

        # Extract aggregator keys
        aggregators = [key for key, val in clsdict.items() if isinstance(val, Aggregator)]

        # Assign name attribute for each descriptor
        for name in aggregators:
            clsdict[name].name = name

        clsdict["_aggregators"] = aggregators

        # Create the new class object
        clsobj = super().__new__(cls, clsname, bases, dict(clsdict))

        # Aggregate parameters from the current class and its bases
        parameters = []
        try:
            parameters += clsobj._parameters
        except AttributeError:
            pass

        for base in bases:
            base_parameters = getattr(base, "_parameters", [])
            parameters += base_parameters

        setattr(clsobj, "_parameters", parameters)

        # Categorize parameters based on their attributes
        sized_parameters = []
        polynomial_parameters = []
        required_parameters = []
        optional_parameters = []

        for param in parameters:
            descriptor = getattr(clsobj, param)

            # Skip if it's not an instance of Descriptor
            if not isinstance(descriptor, Descriptor):
                continue

            if hasattr(descriptor, "size"):
                sized_parameters.append(param)
            if hasattr(descriptor, "fill_values"):
                polynomial_parameters.append(param)
            if descriptor.default is None and not descriptor.is_optional:
                required_parameters.append(param)
            if descriptor.is_optional:
                optional_parameters.append(param)

        setattr(clsobj, "_sized_parameters", sized_parameters)
        setattr(clsobj, "_polynomial_parameters", polynomial_parameters)
        setattr(clsobj, "_required_parameters", required_parameters)
        setattr(clsobj, "_optional_parameters", optional_parameters)

        # Collect descriptors from base classes
        for base in bases:
            descriptors += getattr(base, "_descriptors", [])

        # Remove duplicates from the descriptors list
        args = list(dict.fromkeys(descriptors))

        # Register descriptor fields as arguments in __init__
        sig = make_signature(args)
        setattr(clsobj, "__signature__", sig)

        return clsobj


class AbstractStructMeta(StructMeta, ABCMeta):
    """Base class to allow for abstract metaclass structures."""

    pass


# %% Stucture / ParameterHandler
class Structure(metaclass=AbstractStructMeta):
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        bound = self.__signature__.bind_partial(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, val)

        self._parameters_dict = Dict()
        for param in self._parameters:
            value = getattr(self, param)
            if param in self._optional_parameters and value is None:
                continue
            self._parameters_dict[param] = value

    @property
    def parameters(self) -> dict:
        """dict: Parameters of the instance."""
        parameters = self._parameters_dict

        parameters.update(self.aggregated_parameters)

        return parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """
        Set parameters for the instance.

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
                raise ValueError("Not a valid parameter.")
            if value is not None:
                setattr(self, param, value)
                self._parameters_dict[param] = value

    @property
    def sized_parameters(self) -> Dict:
        """dict: Sized parameters of the instance."""
        parameters = {
            key: value for key, value in self.parameters.items() if key in self._sized_parameters
        }
        return Dict(parameters)

    @property
    def aggregated_parameters(self) -> Dict:
        """dict: Aggregated parameters of the instance."""
        parameters = {key: getattr(self, key) for key in self._aggregators}
        return Dict(parameters)

    @property
    def polynomial_parameters(self) -> Dict:
        """dict: Polynomial parameters of the instance."""
        parameters = {
            key: value
            for key, value in self.parameters.items()
            if key in self._polynomial_parameters
        }
        return Dict(parameters)

    @property
    def required_parameters(self) -> list:
        """list: Parameters that have no default value."""
        return self._required_parameters

    @property
    def missing_parameters(self) -> list:
        """list: Parameters that are required but not set."""
        missing_parameters = []
        for param in self.required_parameters:
            if getattr(self, param) is None:
                missing_parameters.append(param)

        return missing_parameters

    def check_required_parameters(self) -> bool:
        """
        Verify if all required parameters are set.

        Returns
        -------
        bool
            True if all required parameters are set. False otherwise.

        Warning:
            If any of the required parameters are missing.
        """
        if len(self.missing_parameters) == 0:
            return True
        else:
            for param in self.missing_parameters:
                warn(f'Missing parameter "{param}".')
            return False


def frozen_attributes(cls: type) -> type:
    """
    Class decorator to prevent setting new attributes after initialization.

    Parameters
    ----------
    cls : type
        The class to decorate.

    Returns
    -------
    type
        The decorated class with frozen attributes after initialization.
    """
    cls._is_frozen = False

    def frozensetattr(self: Structure, key: str, value: Any) -> None:
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"{cls.__name__} object has no attribute {key}")
        object.__setattr__(self, key, value)

    def init_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Structure, *args: Any, **kwargs: Any) -> None:
            func(self, *args, **kwargs)
            self._is_frozen = True

        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls
