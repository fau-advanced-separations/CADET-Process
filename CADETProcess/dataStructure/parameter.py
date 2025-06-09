import copy
import math
import operator
import typing as tp
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from .dataStructure import Descriptor


class ParameterBase(Descriptor):
    """
    Base class for model parameters with potential constraints or type-casting.

    Unlike mere data members, parameters can have default values, support type
    constraints, and cast from certain types to their target type.

    Attributes
    ----------
    default : Any
        Default value of the parameter.
    unit : str
        Unit of the parameter.
    description : str
        Description or context of the parameter.

    Notes
    -----
    1. Supports deep copying of default values, allowing mutable defaults without side effects.
    2. Subclasses can further specify type constraints (like `Typed`).
    3. They can also define
      - immutable parameters (like `Constant`) and
      - options-based parameters (`Switch`).

    See Also
    --------
    Descriptor
    Structure
    Constant
    Switch
    Typed
    Bool
    Integer
    Tuple
    Float
    String
    Dictionary
    """

    def __init__(
        self,
        *args: Any,
        default: Optional[Any] = None,
        is_optional: bool = False,
        unit: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Parameter instance.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        default : Any, optional
            Default value for the parameter. Defaults to None.
        is_optional : bool, optional
            If True, parameter is not added to list of required parameters.
            Defaults to False
        unit : str, optional
            Unit of the parameter. Defaults to None.
        description : str, optional
            Description of the parameter. Defaults to None.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        self.default = default
        self.is_optional = is_optional
        self.unit = unit
        self.description = description
        super().__init__(*args, **kwargs)

    @property
    def default(self) -> Any:
        """Any: Get or set the default value of the parameter."""
        return copy.deepcopy(self._default)

    @default.setter
    def default(self, value: Any) -> None:
        """
        Set the default value of the parameter.

        Parameters
        ----------
        value : Any
            Value to set as default.
        """
        if value is not None:
            val = self._prepare(None, value, recursive=True)
            self._check(None, val, recursive=True)

        self._default = value

    def get_default_value(self, instance: Any) -> Any:
        """
        Return default values if necessary.

        Override this method if type-casting for default values is necessary.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the default value for.

        Returns
        -------
        Any
            Default value.
        """
        default = self.default

        if default is not None:
            default = self._prepare(instance, default, recursive=True)
            if default is not None:
                self._check(instance, default, recursive=True)

        return default

    def __get__(self, instance: Any, cls: type) -> Any:
        """
        Retrieve the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        cls : Type[Any]
            Class to which the descriptor belongs.

        Returns
        -------
        Any
            Descriptor value or the default value if the descriptor value isn't set.
        """
        if instance is None:
            return self

        try:
            value = Descriptor.__get__(self, instance, cls)
        except KeyError:
            value = self.get_default_value(instance)

        if value is not None:
            self._check(instance, value, recursive=True)

        return value

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Set the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to set the descriptor value for.
        value : Any
            Value to set.
        """
        if value is None:
            value = self.get_default_value(instance)

        if value is not None:
            value = self._prepare(instance, value, recursive=True)
            self._check(instance, value, recursive=True)

        try:
            if self.name in instance._parameters:
                instance._parameters_dict[self.name] = value
        except AttributeError:
            pass

        super().__set__(instance, value)

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
            If True, prepare values recursively.

        Returns
        -------
        Any
            Prepared value.
        """
        return value

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> Any:
        """
        Check the given value.

        Override this method for specific checks.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.
        """
        return


# %% Constant Parameters


class Constant(ParameterBase):
    """
    Parameter that is immutable once set.

    Attributes
    ----------
    value : Any
        The immutable value of the parameter.

    Notes
    -----
    Once set, the value of a Constant parameter cannot be modified.
    """

    def __init__(self, value: Any, *args: Any, **kwargs: Any) -> None:
        """
        Initialize a Constant instance.

        Parameters
        ----------
        value : Any
            Constant value for the parameter.
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        super().__init__(*args, default=value, **kwargs)

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Disallow modification of the value of a Constant parameter.

        Raises
        ------
        ValueError
            If trying to modify the constant parameter.
        """
        raise ValueError("Cannot modify constant parameter.")


class Switch(ParameterBase):
    """
    Parameter that can be set to one of several predefined options.

    Attributes
    ----------
    valid : list
        List of valid options for the parameter.

    Notes
    -----
    Assign a value to this parameter from the `valid` list.
    """

    def __init__(self, *args: Any, valid: list, **kwargs: Any) -> None:
        """
        Initialize a Switch instance.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        valid : list
            List of valid options for the parameter.
        **kwargs : Any
            Arbitrary keyword arguments.

        Raises
        ------
        TypeError
            If `valid` is not a list.
        ValueError
            If the `valid` list is empty.
        """
        if not isinstance(valid, list):
            raise TypeError("Expected a list for valid entries")
        if not valid:
            raise ValueError("The valid options list cannot be empty.")
        self.valid = valid

        super().__init__(*args, **kwargs)

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Verify if the value belongs to the valid options.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            The value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.

        Raises
        ------
        ValueError
            If the value isn't one of the valid options.
        """
        if value not in self.valid:
            raise ValueError(f"Value must be one of {self.valid}")

        if recursive:
            super()._check(instance, value, recursive)


# %% Typed Parameters


class Typed(ParameterBase):
    """
    Mixin for parameters constrained to a specific type.

    `Typed` extends the base `Parameter` class with type constraints. When instantiated
    with a specific type (`ty`), it ensures values are of that type or can be cast
    to that type. If `ty` is not specified during instantiation and is not predefined
    by a subclass, an error is raised.

    Attributes
    ----------
    ty : type
        Desired type for the parameter. Defaults to accepting any type. Subclasses
        can directly set this attribute.

    Methods
    -------
    cast_value(value) -> Any:
        Attempts to cast the value to the target type. By default, returns the value
        as is. Subclasses can override for specific casting behavior.
    _prepare(instance, value, recursive=False) -> Any:
        Prepares and optionally type-casts the value before checking its type. This
        method uses `cast_value` to attempt to cast the value to the required type.
    _check(instance, value, recursive=False) -> bool:
        Validates if the value matches the desired type (`ty`). Raises a TypeError if
        validation fails.

    Notes
    -----
    - If `ty` is specified during instantiation, any value assigned to this parameter
      undergoes validation against this type.
    - Override `cast_value` in subclasses for custom casting logic.
    - Assigning `None` to the parameter removes its current value from the instance.
    - An error is raised during instantiation if `ty` is neither provided nor predefined
      by a subclass.

    See Also
    --------
    Parameter
    Bool
    Integer
    Tuple
    Float
    String
    Dictionary
    """

    def __init__(self, *args: Any, ty: Optional[type] = None, **kwargs: Any) -> None:
        """
        Initialize a Typed instance.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        ty : type, optional
            The desired type for the parameter.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        if ty is not None:
            self.ty = ty
        elif not hasattr(self, "ty"):
            raise ValueError("Type must be provided either in a subclass or during instantiation.")

        super().__init__(*args, **kwargs)

    def cast_value(self, value: Any) -> Any:
        """
        Cast the type of the given value.

        This method is a placeholder. Override it if type-casting is necessary.

        Parameters
        ----------
        value : Any
            Value to cast.

        Returns
        -------
        Any
            The unmodified value.
        """
        return value

    def _prepare(self, instance: Any, value: Any, recursive: bool = False) -> Any:
        """
        Prepare and optionally type-cast the value before type validation.

        This method is called before `_check` to provide an opportunity to
        process or type-cast the value. By default, it uses the `cast_value`
        method to attempt casting the value to the desired type.

        Parameters
        ----------
        instance : Any
            The instance to which this descriptor belongs.
        value : Any
            The value to be prepared or type-cast.
        recursive : bool, optional
            If True, the preparation is done recursively by calling the parent's
            _prepare method. This can be useful if `Typed` is combined with other
            descriptor classes that might also need to process the value.
            Defaults to False.

        Returns
        -------
        Any
            The potentially type-cast value, ready for type validation.
        """
        value = self.cast_value(value)

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Validate the value's type against `ty`.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.

        Raises
        ------
        TypeError
            If the value's type doesn't match `ty`.
        """
        if not isinstance(value, self.ty):
            raise TypeError(f"Expected type {self.ty}, got {type(value)}")

        if recursive:
            super()._check(instance, value, recursive)


class Bool(Typed):
    """
    Parameter descriptor constrained to boolean values.

    Notes
    -----
    This class also supports casting integers 0 and 1 to their boolean equivalents.
    """

    ty = bool

    def cast_value(self, value: Any) -> Union[bool, Any]:
        """
        Convert integers 0 and 1 to their respective boolean values.

        Parameters
        ----------
        value : Any
            Value to be cast.

        Returns
        -------
        Union[bool, Any]
            Boolean equivalent if value is 0 or 1; otherwise, the original value.
        """
        if isinstance(value, (int, np.bool_)) and value in [0, 1]:
            value = bool(value)
        return value


class Integer(Typed):
    """Parameter descriptor constrained to integer values."""

    ty = int


class Float(Typed):
    """
    Parameter descriptor constrained to float values.

    Notes
    -----
    This class also supports casting integers and numpy numbers to floats.
    """

    ty = float

    def cast_value(self, value: Any) -> Union[float, Any]:
        """
        Convert integers and numpy numbers to float.

        Parameters
        ----------
        value : Any
            Value to be cast.

        Returns
        -------
        Union[float, Any]
            Float equivalent if value is an integer or numpy number;
            otherwise, the original value.
        """
        try:
            value = float(np.array(value).squeeze())
        except ValueError:
            raise TypeError("Cannot cast value to float.")

        return value


class String(Typed):
    """Parameter descriptor constrained to string values."""

    ty = str


class Tuple(Typed):
    """Parameter descriptor constrained to tuple values."""

    ty = tuple


class List(Typed):
    """Parameter descriptor constrained to list values."""

    ty = list


class Dictionary(Typed):
    """Parameter descriptor constrained to dictionary (`dict`) values."""

    ty = dict


class NdArray(Typed):
    """
    Parameter descriptor constrained to np.ndarray values.

    Notes
    -----
    The `cast_value` method automatically converts lists to numpy arrays and
    wraps scalars (int or float) into single-element numpy arrays.
    """

    ty = np.ndarray

    def cast_value(self, value: Any) -> Any | np.ndarray:
        """
        Cast lists or scalars (int or float) to numpy arrays.

        Parameters
        ----------
        value : Any
            The value to be casted.

        Returns
        -------
        np.ndarray or Any
            If the value is a list or scalar (int or float),
            it returns its numpy array equivalent.
            Otherwise, it returns the value unchanged.
        """
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, (int, float)):
            value = np.array((value,))

        return value


class Callable(ParameterBase):
    """
    Parameter descriptor constrained to callable objects.

    Designed to ensure a given parameter is callable. This is distinct from using a type
    constraint since built-in functions (e.g., those implemented in C) won't be captured
    by `types.FunctionTypes`.

    Examples
    --------
    Here's how you might use the `Callable` class:

    >>> class MyModel:
    ...     func = Callable()
    >>> model = MyModel()
    >>> model.func = print  # This is fine as print is callable
    >>> model.func = "not_callable"  # This will raise a TypeError

    See Also
    --------
    Parameter
    Typed
    """

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Check if the given value is callable.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.

        Raises
        ------
        TypeError
            If the value is not callable.
        """
        if not callable(value):
            raise TypeError("Expected object to be callable")

        if recursive:
            super()._check(instance, value, recursive)


# Also check dtype
class TypedList(List, Typed):
    """Parameter descriptor for lists with elements of e specific dtype."""

    ty = list

    def __init__(self, *args: Any, dtype: Optional[type] = None, **kwargs: Any) -> None:
        """
        Initialize a Typed instance.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        ty : type, optional
            The desired type for the parameter.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        if dtype is not None:
            self.dtype = dtype
        elif not hasattr(self, "dtype"):
            raise ValueError("dtype must be provided either in a subclass or during instantiation.")

        super().__init__(*args, **kwargs)

    def _prepare(self, instance: Any, value: Any, recursive: bool = False) -> Any:
        """
        Prepare and optionally cast the array dtype.

        This method is called before `_check` to provide an opportunity to
        process or type-cast the value. By default, it uses the `cast_value`
        method to attempt casting the value to the desired type.

        Parameters
        ----------
        instance : Any
            The instance to which this descriptor belongs.
        value : Any
            The value to be prepared or type-cast.
        recursive : bool, optional
            If True, the preparation is done recursively by calling the parent's
            _prepare method. This can be useful if `Typed` is combined with other
            descriptor classes that might also need to process the value.
            Defaults to False.

        Returns
        -------
        Any
            The potentially type-cast value, ready for type validation.
        """
        try:
            value = np.array(value, dtype=self.dtype).tolist()
        except ValueError:
            raise ValueError(f"could not convert elements to {self.dtype}")

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value

    def check_dtype(self, value: Any) -> None:
        """
        Validate if the dtype of values is correct.

        Parameters
        ----------
        value : Any
            Value to check against the range.

        Raises
        ------
        TypeError
            If the dtype is not correct.
        """
        value_array = np.array(value)

        if value_array.dtype != self.dtype:
            raise ValueError(f"Value entries must be of type {self.dtype}")

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Validate the value against the range.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.
        """
        self.check_dtype(value)

        if recursive:
            super()._check(instance, value, recursive)


class IntegerList(TypedList):
    """List of integers."""

    dtype = int


class FloatList(TypedList):
    """List of floats."""

    dtype = float


# %% Ranged Parameters


class Ranged(ParameterBase):
    """
    Descriptor for parameters within specified bounds.

    Allows setting values constrained by provided lower and upper bounds. The actual comparisons
    against the bounds can be customized using the `lb_op` and `ub_op` comparison functions.

    Attributes
    ----------
    lb : numeric
        The lower bound of the parameter. Default is negative infinity.
    lb_op : callable
        A callable that defines the comparison operation against the lower bound.
        Default is less than (<).
    ub : numeric
        The upper bound of the parameter. Default is positive infinity.
    ub_op : callable
        A callable that defines the comparison operation against the upper bound.
        Default is greater than (>).

    Examples
    --------
    Constraining a parameter between 0 and 10:

    >>> class MyClass:
    ...     value = Ranged(lb=0, ub=10)
    >>> obj = MyClass()
    >>> obj.value = 5  # This is valid
    >>> obj.value = -5  # Raises an error

    Notes
    -----
    - By default, values are checked to be strictly within the bounds (exclusive).
      To change this behavior, use the `lb_op` and `ub_op` parameters.
    - The `check_range` method can be overridden for custom range validation logic,
      especially if the data structure isn't a simple scalar value
      (e.g. for np.ndarrays).

    See Also
    --------
    ParameterBase
    """

    def __init__(
        self,
        *args: Any,
        lb: float = -math.inf,
        lb_op: tp.Callable = operator.lt,
        ub: float = math.inf,
        ub_op: tp.Callable = operator.gt,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Ranged descriptor.

        Parameters
        ----------
        *args :
            Parameters for Parameter Base.
        lb : numeric, optional
            Lower bound. Defaults to negative infinity.
        lb_op : callable, optional
            Comparison for the lower bound. Defaults to less than.
        ub : numeric, optional
            Upper bound. Defaults to positive infinity.
        ub_op : callable, optional
            Comparison for the upper bound. Defaults to greater than.
        **kwargs: Optional
            Additional Parameters for ParameterBase.

        """
        self.lb = lb
        self.lb_op = lb_op
        self.ub = ub
        self.ub_op = ub_op

        super().__init__(*args, **kwargs)

    def check_range(self, value: Any) -> None:
        """
        Validate if the value is within the defined range.

        Override this method if other methods for checking ranges are required
        (e.g. for np.ndarrays).

        Parameters
        ----------
        value : Any
            Value to check against the range.

        Raises
        ------
        ValueError
            If the value is outside the specified bounds.
        """
        if self.lb_op(value, self.lb):
            raise ValueError(f"Value {value} is below the lower bound of {self.lb}")
        elif self.ub_op(value, self.ub):
            raise ValueError(f"Value {value} is above the upper bound of {self.ub}")

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Validate the value against the range.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.
        """
        self.check_range(value)

        if recursive:
            super()._check(instance, value, recursive)


# Ranged scalar parameters
class RangedInteger(Integer, Ranged):
    """Parameter descriptor for integers parameters constrained within bounds."""

    pass


class RangedFloat(Float, Ranged):
    """Parameter descriptor for float parameters constrained within bounds."""

    pass


# Ranged list / array parameters
class RangedArray(Ranged):
    """
    Parameter descriptor for arrays with elements constrained within some bounds.

    This class extends the Ranged descriptor to support array-like structures
    (lists, numpy arrays, etc.). Each element in the array is individually checked
    against the specified bounds.

    Examples
    --------
    Constraining elements of an array parameter between 0 and 10:

    >>> class MyClass:
    ...     values = RangedArray(lb=0, ub=10)
    >>> obj = MyClass()
    >>> obj.values = [5, 7, 2]  # This is valid
    >>> obj.values = [5, -1, 2]  # Raises an error indicating the second element is
        below the lower bound.

    Notes
    -----
    - The class uses numpy for efficient element-wise comparison.
    - In case of out-of-bound values, the raised exception specifies the index/indices
      of such values.

    See Also
    --------
    Ranged
    """

    def check_range(self, value: npt.ArrayLike) -> None:
        """
        Check each element of an array-like structure against specified bounds.

        Parameters
        ----------
        value : array-like
            The array whose elements need to be checked against the range.

        Raises
        ------
        ValueError
            If any element(s) of the array are outside the specified bounds. The raised exception
            indicates the index/indices of out-of-bound values.
        """
        value_array = np.array(value)

        if np.any(self.lb_op(value_array, self.lb)):
            idx = np.where(self.lb_op(value_array, self.lb))[0]
            raise ValueError(
                f"Element(s) at index/indices {idx} below the lower bound of {self.lb}"
            )

        elif np.any(self.ub_op(value_array, self.ub)):
            idx = np.where(self.ub_op(value_array, self.ub))[0]
            raise ValueError(
                f"Element(s) at index/indices {idx} above the upper bound of {self.ub}"
            )


class RangedList(List, RangedArray):
    """Parameter descriptor for lists with elements constrained within bounds."""

    pass


class RangedNdArray(NdArray, RangedArray):
    """Parameter descriptor for numpy arrays with elements constrained within bounds."""

    pass


# Unsigned scalar parameters
class Unsigned(Ranged):
    """Parameter descriptor for non-negative parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize unsigned parameter."""
        super().__init__(*args, lb=0, lb_op=operator.lt, **kwargs)


class UnsignedInteger(Integer, Unsigned):
    """Parameter descriptor for unsigned integer parameters."""

    pass


class UnsignedFloat(Float, Unsigned):
    """Parameter descriptor for unsigned floating-point parameters."""

    pass


# Unsigned list / array parameters
class UnsignedArray(Unsigned, RangedArray):
    """Parameter descriptor for arrays with non-negative elements."""

    pass


class UnsignedList(List, UnsignedArray):
    """Parameter descriptor for lists with non-negative elements."""

    pass


class UnsignedNdArray(NdArray, UnsignedArray):
    """Parameter descriptor for numpy arrays with non-negative elements."""

    pass


# %% Sized Parameters


class Sized(ParameterBase):
    """
    Descriptor for parameters with size that potentially depends on instance attributes.

    Attributes
    ----------
    size : tuple
        Expected size or dimensions of the parameter. Individual elements can be
        either integers or strings indicating other instance parameters that influence
        the size.

    Methods
    -------
    is_independent : bool
        Determines whether the size is independent of other parameters.
    get_size(value) -> int
        Calculates the size of the given value. Override for custom behavior.
    get_expected_size(instance) -> int
        Computes the expected size based on the instance's other attributes.
    check_size(instance, value)
        Validates that the provided value's size matches the expected size.
    """

    def __init__(self, *args: Any, size: Union[int, tuple], **kwargs: Any) -> None:
        """
        Initialize the Sized descriptor.

        Parameters
        ----------
        *args :
            Parameters for Parameter Base.
        size : int or tuple
            The expected size or dimensions of the parameter. Individual elements
            can be either integers or strings (indicating other instance parameters).
        **kwargs :
            Additional parameters for Parameter Base.
        """
        if not isinstance(size, tuple):
            size = (size,)

        self.size = size

        super().__init__(*args, **kwargs)

    @property
    def is_independent(self) -> bool:
        """
        Determine whether the size is independent of other parameters.

        Returns
        -------
        bool
            True if the size is independent,
            False if it depends on other instance attributes.
        """
        flag = True
        if any(isinstance(i, str) for i in self.size):
            flag = False
        return flag

    def get_size(self, value: Any) -> int | tuple[int, ...]:
        """
        Determine the size of the provided value.

        Parameters
        ----------
        value : Any
            The value for which the size needs to be calculated.

        Returns
        -------
        int | tuple[int, ...]
            Size of the value.
        """
        return len(value)

    def get_expected_size(self, instance: Any) -> int:
        """
        Compute the expected size based on the instance's other attributes.

        Parameters
        ----------
        instance : object
            The instance whose attributes determine the expected size.

        Returns
        -------
        int
            Computed expected size based on instance attributes.

        Raises
        ------
        ValueError
            If an attribute, on which the size depends, is not set.
        """
        if not self.is_independent and instance is None:
            raise ValueError("Parameter is not independent, need instance get expected size!")

        size = []
        for i in self.size:
            if isinstance(i, int):
                size.append(i)
                continue

            i_value = getattr(instance, i)
            if i_value is None:
                raise ValueError(f"Value for {i} not set.")

            size.append(i_value)

        return np.prod(size)

    def check_size(self, instance: Any, value: Any) -> None:
        """
        Validate that the provided value's size matches the expected size.

        Parameters
        ----------
        instance : object
            The instance associated with the parameter.
        value : Any
            The value whose size needs to be validated.

        Raises
        ------
        ValueError
            If the value's size does not match the expected size.
        """
        size = self.get_size(value)
        try:
            expected_size = self.get_expected_size(instance)
        except ValueError:
            if np.array(value).squeeze().ndim == 0:
                return
            raise

        if size != expected_size:
            raise ValueError(f"Expected size {expected_size}")

    def _prepare(self, instance: Any, value: Any, recursive: bool = False) -> Any:
        """
        Prepare the value by typecasting and adjusting its size.

        This method handles typecasting of the provided value to the expected type
        (`self.ty`) and potentially adjusts its size. If the size of the parameter
        depends on other instance attributes, it will retrieve that expected size and
        adjust the value accordingly.

        Parameters
        ----------
        instance : object
            The instance associated with the parameter.
        value : Any
            The value to be prepared.
        recursive : bool, optional
            If True, recursively prepares the value using the superclass's method.
            Defaults to False.

        Returns
        -------
        Any
            The prepared value, which has been potentially typecasted and resized.

        Raises
        ------
        ValueError
            If the value cannot be cast to the expected type.
        """
        if not isinstance(value, self.ty):
            try:
                expected_size = self.get_expected_size(instance)
            except ValueError:
                expected_size = None

            if isinstance(value, (int, float)):
                value = self.ty((value,))
            else:
                raise ValueError("Cannot cast value from given value.")

            if expected_size == 0:
                value = None
            elif expected_size is not None:
                value = np.prod(expected_size) * value

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value

    def _check(self, instance: Any, value: Any, recursive: bool = False) -> None:
        """
        Validate the provided value's size.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.
        """
        self.check_size(instance, value)

        if recursive:
            super()._check(instance, value, recursive)


class SizedList(List, Sized):
    """Descriptor for lists whose size may depend on other instance attributes."""

    pass


class SizedTuple(Tuple, Sized):
    """Descriptor for tuples whose size may depend on other instance attributes."""

    pass


class SizedNdArray(NdArray, Sized):
    """Descriptor for NumPy arrays whose size may depend on other instance attributes."""

    def get_size(self, value: np.ndarray) -> tuple[int, ...]:
        """
        Determine the size of the provided value.

        Parameters
        ----------
        value : Any
            The value for which the size needs to be determined.

        Returns
        -------
        tuple[int, ...]
            Size of the value.
        """
        return value.shape

    def get_expected_size(self, instance: Optional[Any]) -> tuple:
        """
        Calculate the expected size of a numpy array based on the instance's other attributes.

        Returns
        -------
        tuple
            The expected shape of the array.
        """
        if not self.is_independent and instance is None:
            raise ValueError("Parameter is not independent, need instance!")

        expected_size = []
        for i in self.size:
            if isinstance(i, int):
                expected_size += [i]
                continue

            dim = getattr(instance, i)
            if dim is None:
                raise ValueError(f"Value for {i} not set.")

            if isinstance(dim, np.ndarray):
                dim = list(dim.shape)
            elif isinstance(dim, int):
                dim = [dim]

            expected_size += dim

        expected_size = tuple(expected_size)

        return expected_size

    def _prepare(
        self,
        instance: object,
        value: Union[int, float, npt.ArrayLike],
        recursive: bool = False,
    ) -> np.ndarray:
        """
        Prepare the value for a NumPy array, ensuring it has the expected shape.

        If the value is a scalar (int or float), this method will transform it into a NumPy
        array with the expected shape, filled with the scalar value. If the value is
        already an array, this method ensures its shape matches the expected shape.

        Parameters
        ----------
        instance : object
            The instance associated with the parameter.
        value : Any
            The value to be prepared.
        recursive : bool, optional
            If True, recursively prepares the value using the superclass's method.
            Defaults to False.

        Returns
        -------
        np.ndarray
            The prepared NumPy array with the correct shape.

        Raises
        ------
        ValueError
            If the value cannot be cast to a NumPy array with the expected shape.
        """
        if np.array(value).squeeze().ndim == 0:
            try:
                expected_size = self.get_expected_size(instance)
            except ValueError:
                expected_size = None

            if expected_size is not None:
                value = value * np.ones(expected_size)
        else:
            try:
                value_array = np.array(value)
            except ValueError:
                raise TypeError("Cannot cast value from given value.")
            self.check_size(instance, value_array)

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value


class SizedRangedList(RangedList, SizedList):
    """Descriptor for ranged lists whose size depends on other instance attributes."""

    pass


class SizedUnsignedList(UnsignedList, SizedList):
    """Descriptor for unsigned lists whose size depends on other instance attributes."""

    pass


class SizedUnsignedNdArray(UnsignedNdArray, SizedNdArray):
    """Descriptor for unsigned numpy arrays whose size depends on other instance attributes."""

    pass


# Also check dtype
class SizedFloatList(FloatList, SizedList):
    """Descriptor for lists of floats whose size depends on other instance attributes."""

    pass


class SizedRangedIntegerList(RangedList, IntegerList, SizedList):
    """Descriptor for ranged lists of integers whose size depends on other instance attributes."""

    pass


class SizedUnsignedIntegerList(UnsignedList, IntegerList, SizedList):
    """Descriptor for unsigned lists of integers whose size depends on other instance attributes."""

    pass


# %% Dimensionalized Parameters


class DimensionalizedArray(NdArray):
    """
    Parameter descriptor constrained to np.arrays with a specific dimensionality.

    The descriptor ensures that the ndarray assigned matches the specified
    number of dimensions (n_dim).

    Attributes
    ----------
    n_dim : int
        The number of dimensions the array must have.

    Examples
    --------
    To create a descriptor for 2-dimensional arrays:

    >>> class MyClass:
    ...     arr = DimensionalizedArray(n_dim=2)
    >>> obj = MyClass()
    >>> obj.arr = np.array([[1, 2], [3, 4]])  # This is valid
    >>> obj.arr = np.array([1, 2, 3, 4])  # Raises a ValueError

    Notes
    -----
    The n_dim attribute can be set during initialization.
    """

    n_dim = None

    def __init__(self, n_dim: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the DimensionalizedArray descriptor.

        Parameters
        ----------
        n_dim : int, optional
            The number of dimensions the array must have. If not specified,
            it must be set before usage.
        *args : optional
            Parameters for NdArray.
        **kwargs : optional
            Additional Parameters for NdArray.

        Raises
        ------
        ValueError
            If n_dim is not an integer.
        """
        super().__init__(*args, **kwargs)

        if n_dim is not None:
            if not isinstance(n_dim, int):
                raise ValueError("Dimensionality (n_dim) must be an integer.")
            self.n_dim = n_dim

        # Ensure the dimension is set and valid
        if self.n_dim is None:
            raise ValueError("Dimensionality (n_dim) must be set during initialization.")

    def _check(self, instance: Any, value: npt.ArrayLike, recursive: bool = False) -> None:
        """
        Check the dimensionality of the given value.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        value : Any
            Value to check.
        recursive : bool, optional
            If True, perform the check recursively. Defaults to False.

        Raises
        ------
        ValueError
            If the number of dimensions of the ndarray doesn't match n_dim.
        """
        if self.n_dim != np.array(value).ndim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {np.array(value).ndim}.")

        if recursive:
            super()._check(instance, value, recursive)


class Vector(DimensionalizedArray):
    """
    Parameter descriptor for one-dimensional numpy arrays (vectors).

    Attributes
    ----------
    n_dim : int
        Dimensionality of the numpy array, set to 1 for vectors.

    Examples
    --------
    >>> class MyModel:
    ...     coordinates = Vector()
    >>> model.coordinates = np.array([1, 2, 3])  # Valid
    >>> model.coordinates = np.array([[1, 2], [3, 4]])  # Raises ValueError

    See Also
    --------
    Dimensionalized
    NdArray
    """

    n_dim = 1


class Matrix(DimensionalizedArray):
    """
    Parameter descriptor for two-dimensional numpy arrays (matrices).

    This descriptor ensures that the ndarray assigned is two-dimensional.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the numpy array, set to 2 for matrices.

    Examples
    --------
    >>> class MyModel:
    ...     data = Matrix()
    >>> model = MyModel()
    >>> model.data = np.array([[1, 2], [3, 4]])  # Valid
    >>> model.data = np.array([1, 2, 3, 4])  # Raises ValueError

    See Also
    --------
    DimensionalizedArray
    NdArray
    """

    n_dim = 2


# %% Polynomial Parameters


class NdPolynomial(SizedNdArray):
    """
    Dependently sized polynomial for n entries.

    This descriptor represents a polynomial whose size or dimensions may depend on other
    instance attributes. The polynomial can also be thought of as a 2D array, where each
    row represents a polynomial of a certain degree.

    Important: Use [entries x n_coeff] for dependencies.

    Parameters
    ----------
    n_entries : int, optional
        Number of polynomials or rows. Default is None.
    n_coeff : int, optional
        Number of coefficients for each polynomial or columns. Default is None.

    Attributes
    ----------
    size : tuple
        The shape of the polynomial array, determined from n_entries and n_coeff.

    Notes
    -----
    Currently, NdPolynomial is implemented as `SizedNdArray`.
    Consequently, no default values can be set since their size would depend on the
    dependent variables.
    In theory, this could be split into `NdPolynomial` and `SizedNdPolynomial`,
    but there is currently no use for this distinction.

    Methods
    -------
    fill_values(dims, value) -> np.ndarray:
        Fills values to generate the polynomial matrix of the desired size.
    _prepare(instance, value, recursive=False) -> np.ndarray:
        Prepare the given polynomial matrix s.t. it adheres to the expected size.
    """

    def __init__(
        self,
        *args: Any,
        n_entries: Optional[int] = None,
        n_coeff: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an NdPolynomial descriptor with specific entries and coefficients.

        Parameters
        ----------
        n_entries : int, optional
            Set the number of polynomials or rows. If not provided, it must be inferred
            from 'size'.
        n_coeff : int, optional
            Define the number of coefficients for each polynomial or columns. If not
            provided, it must be inferred from 'size'.
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments, can include 'size' which denotes the shape of
            the polynomial array.

        Raises
        ------
        ValueError
            If both 'n_entries' and 'n_coeff' are missing and 'size' is not provided or
            is incomplete.
            If there's a mismatch or redundancy in provided dimensions.

        Notes
        -----
        The shape of the polynomial array is determined from 'n_entries' and 'n_coeff'
        or from the 'size' keyword argument.
        """
        if "default" in kwargs and kwargs["default"] != 0:
            raise ValueError("Default value for NdPolynomial must always be 0.")

        try:
            size = kwargs["size"]
            if not isinstance(size, tuple):
                size = (size,)
        except KeyError:
            size = tuple()

        if n_entries is None and n_coeff is None and len(size) < 2:
            raise ValueError("Missing value for n_coeff for shape")

        if n_entries is not None:
            if len(size) > 1:
                raise ValueError("Found duplicate n_entries for shape")
            _n_entries = n_entries
            if n_coeff is not None:
                if len(size) > 0:
                    raise ValueError("Found duplicate n_entries for shape")
        else:
            _n_entries = size[0]

        if n_coeff is not None:
            if len(size) > 1:
                raise ValueError("Found duplicate n_entries for shape")
            _n_coeff = n_coeff
            if n_entries is not None:
                if len(size) > 0:
                    raise ValueError("Found duplicate n_entries for shape")
        else:
            if n_entries is not None:
                _n_coeff = size[0]
            else:
                _n_coeff = size[1]

        size = (_n_entries, _n_coeff)

        kwargs["size"] = size

        super().__init__(*args, **kwargs)

    def fill_values(
        self, dims: tuple[int], value: Union[int, float, np.ndarray, list]
    ) -> np.ndarray:
        """
        Fill values to generate the polynomial matrix of the desired size.

        Parameters
        ----------
        dims : tuple of int
            Dimensions (n_entries, n_coeff) of the polynomial matrix.
        value : Union[int, float, np.ndarray, list]
            Value(s) to be filled in the polynomial matrix.

        Returns
        -------
        np.ndarray
            Polynomial matrix of the desired size filled with given values.

        Raises
        ------
        ValueError
            If there's a mismatch between the provided values and expected dimensions.
        """
        if len(dims) == 1:
            n_entries = 1
            n_coeff = dims[0]
            single_entry = True
            value = np.array(value, ndmin=2)
        else:
            n_entries = dims[0]
            n_coeff = dims[1]
            single_entry = False

        if single_entry and n_entries > 1:
            raise ValueError("Can only set single entry if n_entries == 1.")

        if isinstance(value, np.ndarray) and value.size == 1:
            value = float(value.squeeze())

        if isinstance(value, (int, float)):
            value = n_entries * [value]
        elif isinstance(value, np.ndarray):
            value = value.tolist()

        if len(value) != n_entries:
            raise ValueError("Number of entries does not match")

        _value = np.zeros((n_entries, n_coeff))

        for i, v in enumerate(value):
            if isinstance(v, (int, float, np.number)):
                v = [v]
            if isinstance(v, (list, tuple)):
                missing = n_coeff - len(v)
                v += missing * (0,)
            _value[i, :] = np.array(v)

        if single_entry:
            _value = _value[0]

        return _value

    def _prepare(
        self,
        instance: object,
        value: Union[int, float, np.ndarray, list],
        recursive: bool = False,
    ) -> np.ndarray:
        """
        Prepare the given polynomial matrix s.t. it adheres to the expected size.

        Parameters
        ----------
        instance : object
            The instance whose attributes might influence the size.
        value : Union[int, float, np.ndarray, list]
            Value(s) to be filled in the polynomial matrix.
        recursive : bool, optional
            If True, perform the operation recursively. Defaults to False.

        Returns
        -------
        np.ndarray
            Polynomial matrix of the desired size prepared as per requirements.
        """
        if instance is not None:
            dims = self.get_expected_size(instance)
            value = self.fill_values(dims, value)

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value


class Polynomial(NdPolynomial):
    """
    Represent a single polynomial using coefficients.

    This class serves as a simplified version of NdPolynomial, specifically tailored for
    single polynomials. It is always defined by its coefficients, removing the need for
    an 'n_entries' parameter.

    Use this class when only a single polynomial representation is required.

    Attributes
    ----------
    size : tuple
        Size of the coefficients for the polynomial. Derived from NdPolynomial but omits
        'n_entries'.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize a Polynomial descriptor with a specific coefficient length.

        Parameters
        ----------
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments. Typically used for any parameters inherited
            from NdPolynomial, except 'n_entries'.

        Notes
        -----
        The Polynomial descriptor is a special case of NdPolynomial with 'n_entries'
        always set to 1. Therefore, it's only defined by its coefficients.
        """
        super().__init__(*args, n_entries=1, **kwargs)
        self.size = self.size[1:]


# %% Modulated Parameters


class DependentlyModulated(Sized):
    """
    Mixin for checking parameter shapes based on other instance attributes.

    This mixin ensures that the size of a parameter is modulo an expected size. If this
    condition is not met, a ValueError is raised.
    """

    def check_mod_value(self, instance: Any, value: Any) -> None:
        """
        Check if the size of the parameter modulo its expected size is zero.

        By default, this method checks the modulo condition, but can be overridden
        by subclasses to incorporate custom behaviors.

        Parameters
        ----------
        instance : Any
            The instance associated with the parameter.
        value : Any
            The value whose size needs to be validated.

        Raises
        ------
        ValueError
            If the modulo condition of the size does not meet the expected criteria.
        """
        size = self.get_size(value)
        expected_size = self.get_expected_size(instance)

        size %= expected_size

        if size != 0:
            raise ValueError(
                f"The size of the value modulo the expected size is not zero. "
                f"Size: {size}, Expected Size: {expected_size}"
            )

    check_size = check_mod_value


class DependentlyModulatedUnsignedList(UnsignedList, SizedList, DependentlyModulated):
    """List of unsigned values whose size is dependent on other attributes."""

    pass
