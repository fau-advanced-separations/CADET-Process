from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt

from CADETProcess.normalize import (
    AutoNormalizer,
    LinearNormalizer,
    LogNormalizer,
    NormalizerBase,
    NullNormalizer,
)

__all__ = [
    "ParameterBase",
    "RangedParameter",
    "ChoiceParameter",
    "LinearConstraint",
    "LinearEqualityConstraint",
    "ParameterMapperBase",
    "ParameterSpace",
]


@dataclass
class ParameterDependency:
    """
    Class to specify a parameter depdendency.

    Attributes
    ----------
    independent_parameters: list[ParameterBase]
        list of dependent parameters
    transform : callable
        Transformation function to derive the value from dependencies.
    """

    dependent_parameter: ParameterBase
    independent_parameters: list[ParameterBase]
    transform: Callable


@dataclass
class ParameterBase:
    """

    Base class for a parameter. Subclasses may override `validate` to apply constraints.

    Attributes
    ----------
    name : str
        Name of the parameter.
    mappers: ParameterMapper
        Parameter mapper.
    """

    name: str
    mappers: ParameterMapperBase | list[ParameterMapperBase] | None = None

    def __post_init__(self) -> None:
        """
        Normalise *mappers* to a list so the rest of the class can iterate.

        without type-checking.
        """
        if self.mappers is None:
            return

        if not isinstance(self.mappers, list):
            self.mappers = [self.mappers]

        for mapper in self.mappers:
            if not isinstance(mapper, ParameterMapperBase):
                raise TypeError(
                    f"All mappers must inherit from ParameterMapperBase, "
                    f"got {type(mapper).__name__!r}"
                )

    def validate(self, value: Any) -> None:
        """
        Validate a value against the parameter's constraints.

        Parameters
        ----------
        value : Any
            The value to validate.

        Notes
        -----
        Subclasses must implement `validate` to enforce parameter rules.
        """
        pass

    def set_value(self, value: Any) -> None:
        """
        Set the value in the mapped objects.

        Iterates over parameter mappers and sets the value.
        """
        self.validate(value)

        if self.mappers is None:
            return
        for mapper in self.mappers:
            mapper.set_value(value)


@dataclass
class RangedParameter(ParameterBase):
    """
    A scalar parameter bounded by a lower and upper limit.

    Attributes
    ----------
    parameter_type : type, default=int
        Expected type of the parameter value (int or float).
    lb : float, default=-inf
        Lower bound (inclusive).
    ub : float, default=inf
        Upper bound (inclusive).
    normalizer : NormalizerBase
        Normalization method.
    """

    parameter_type: type[float | int] = int
    lb: float = -math.inf
    ub: float = math.inf
    normalization: Literal["auto", "linear", "log"] | None = None
    normalizer: Optional[NormalizerBase] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate bounds after initialization.

        Raises
        ------
        ValueError
            If lower bound is greater than or equal to upper bound.
        ValueError
            If infinite bounds input for normalization.
        """
        if self.lb >= self.ub:
            raise ValueError("Lower bound must be < upper bound.")

        self.normalizer = self._build_normalizer()

    def _build_normalizer(self) -> NormalizerBase:
        if self.normalization is None:
            return NullNormalizer(lb_input=self.lb, ub_input=self.ub)

        if np.isinf(self.lb) or np.isinf(self.ub):
            raise ValueError("Normalization requires finite bounds.")

        if self.normalization == "linear":
            return LinearNormalizer(lb_input=self.lb, ub_input=self.ub)
        elif self.normalization == "log":
            return LogNormalizer(lb_input=self.lb, ub_input=self.ub)
        elif self.normalization == "auto":
            return AutoNormalizer(lb_input=self.lb, ub_input=self.ub)
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")

    def validate(self, value: Any) -> None:
        """
        Validate that the value matches the type and lies within bounds.

        Parameters
        ----------
        value : Any
            The value to check.

        Raises
        ------
        TypeError
            If the value type is not correct.
        ValueError
            If the value is outside the specified bounds.
        """
        if not isinstance(value, self.parameter_type):
            raise TypeError("Unexpected Type")

        if not self.lb <= value <= self.ub:
            raise ValueError("Value exceeds bounds")

    def normalize(self, value: float) -> float:
        """Normalize a parameter."""
        return self.normalizer.normalize(value)

    def denormalize(self, value: float) -> float:
        """Denormalize a parameter."""
        return self.normalizer.denormalize(value)


@dataclass(kw_only=True)
class ChoiceParameter(ParameterBase):
    """
    A parameter constrained to a finite set of valid values.

    Attributes
    ----------
    valid_values : list of Any
        List of allowed choices.
    """

    valid_values: list[Any]

    def validate(self, value: Any) -> None:
        """
        Validate that the value is one of the allowed choices.

        Parameters
        ----------
        value : Any
            The value to check.

        Raises
        ------
        ValueError
            If value is not in the list of valid values.
        """
        if value not in self.valid_values:
            raise ValueError(
                f"{value!r} is not a valid choice; "
                f"must be one of {self.valid_values!r}."
            )


@dataclass
class LinearConstraint:
    """
    Represents a linear inequality constraint of the form: lhs · parameters <= b.

    Attributes
    ----------
    parameters : list of RangedParameter
        Parameters involved in the constraint.
    lhs : list of float or float
        Coefficients applied to parameters.
    b : float
        Right-hand side of the inequality.
    """

    parameters: list[RangedParameter]
    lhs: list[float] = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalize and validate the constraint structure.

        Raises
        ------
        ValueError
            If number of coefficients does not match number of parameters.
        """
        if not isinstance(self.parameters, list):
            self.parameters = [self.parameters]
        if np.isscalar(self.lhs):
            self.lhs = [float(self.lhs)] * len(self.parameters)
        if len(self.lhs) != len(self.parameters):
            raise ValueError("Length of lhs must match number of parameters.")
        self.b = float(self.b)


@dataclass
class LinearEqualityConstraint:
    """
    Represents a linear equality constraint of the form: lhs · parameters = b.

    Attributes
    ----------
    parameters : list of RangedParameter
        Parameters involved in the constraint.
    lhs : list of float or float
        Coefficients applied to parameters.
    b : float
        Right-hand side of the equation.
    """

    parameters: list[RangedParameter]
    lhs: list[float] = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalize and validate the constraint structure.

        Raises
        ------
        ValueError
            If number of coefficients does not match number of parameters.
        """
        if not isinstance(self.parameters, list):
            self.parameters = [self.parameters]
        if np.isscalar(self.lhs):
            self.lhs = [float(self.lhs)] * len(self.parameters)
        if len(self.lhs) != len(self.parameters):
            raise ValueError("Length of lhs must match number of parameters.")
        self.b = float(self.b)


def _traverse_mixed_path(root: Any, keys: Sequence[str]) -> tuple[Any, str]:
    """
    Follow *keys* up to the parent of the leaf and return (parent, leaf_key).

    Traverses dict keys *or* object attributes at each hop.
    Raises KeyError / AttributeError if the path is incomplete.
    """
    current = root
    for k in keys[:-1]:
        current = current[k] if isinstance(current, Mapping) else getattr(current, k)
    return current, keys[-1]


@dataclass(slots=True)
class ParameterMapperBase:
    """Abstract base that writes a single parameter into multiple targets."""

    evaluation_objects: list[Any] = field(repr=False)

    def set_value(self, value: Any) -> None:
        """
        Broadcast a parameter value to every evaluation object in.

        ``self.evaluation_objects``.

        The method loops over the list of *evaluation objects* and delegates the
        actual write-operation to the subclass-specific :py:meth:`_set_value`
        implementation.

        Parameters
        ----------
        value : Any
            The value that should be written into each evaluation object.

        """
        for obj in self.evaluation_objects:
            self._set_value(obj, value)

    def _set_value(self, evaluation_object: Any, value: Any) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class ParameterDotPathSetter(ParameterMapperBase):
    """
    Writes *value* to the leaf referenced by a dot-separated path.

    • Traverses dicts *and* objects in the same path.
    """

    path: str

    def __post_init__(self) -> None:
        self._keys: Sequence[str] = self.path.split(".")

    def _set_value(self, evaluation_object: Any, value: Any) -> None:
        parent, leaf = _traverse_mixed_path(evaluation_object, self._keys)

        if isinstance(parent, Mapping):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)


@dataclass
class ParameterSpace:
    """Container for managing parameters and constraints."""

    parameters: list[ParameterBase] = field(default_factory=list)
    linear_constraints: list[LinearConstraint] = field(default_factory=list)
    linear_equality_constraints: list[LinearEqualityConstraint] = field(
        default_factory=list
    )
    parameter_dependencies: list[ParameterDependency] = field(default_factory=list)

    @property
    def n_parameters(self) -> int:
        """Returns the number of parameters in the space."""
        return len(self.parameters)

    def add_parameter(self, parameter: ParameterBase) -> None:
        """
        Add a parameter to the parameter space.

        Parameters
        ----------
        parameters: ParameterBase
            The parameter to be added.

        Raises
        ------
        ValueError
            If a parameter with the same name already exists.
        """
        if any(p.name == parameter.name for p in self.parameters):
            raise ValueError(f"Parameter name '{parameter.name}' already used.")
        self.parameters.append(parameter)

    def add_linear_constraint(self, linear_constraint: LinearConstraint) -> None:
        """
        Add a linear constraint to the parameter space.

        Parameters
        ----------
        linear_constraint: LinearConstraint,
            Linear constraint.

        """
        self.linear_constraints.append(linear_constraint)

    def add_linear_equality_constraint(
        self, linear_equality_constraint: LinearEqualityConstraint
    ) -> None:
        """
        Add a linear equality constraint to the parameter space.

        Parameters
        ----------
        linear_equality_constraint: LinearEqualityConstraint,
            Linear equality constraint.
        """
        self.linear_equality_constraints.append(linear_equality_constraint)

    def add_parameter_dependency(
        self, parameter_dependency: ParameterDependency
    ) -> None:
        """
        Add a parameter dependency to the parameter space.

        Parameters
        ----------
        parameter_dependency : ParameterDependency,
            The parameter dependency.

        TODO: Check that parameter is not already dependent.
        TODO: Should we add the dependency to the Parameter itself?
        """
        if any(
            d.dependent_parameter == parameter_dependency.dependent_parameter
            for d in self.parameter_dependencies
        ):
            raise ValueError(
    f"Parameter {parameter_dependency.dependent_parameter.name} is already dependent."
            )
        self.parameter_dependencies.append(parameter_dependency)

    @property
    def dependent_parameters(self) -> list[ParameterBase]:
        """
        Get all dependent parameters.

        Returns
        -------
        list of ParameterBase
            Parameters that depend on others.
        """
        return [param.dependent_parameter for param in self.parameter_dependencies]

    @property
    def independent_parameters(self) -> list[ParameterBase]:
        """
        Get all independent parameters.

        Returns
        -------
        list of ParameterBase
            Parameters that are independent of others.
        """
        return [p for p in self.parameters if p not in self.dependent_parameters]

    def get_dependent_variables(
        self,
        x_independent: npt.ArrayLike
    ) -> npt.NDArray[Any]:
        """
        Compute values for all parameters.

        Parameters
        ----------
        x_independent : ArrayLike
            Values of the independent parameters (list, tuple, ndarray …).

        Returns
        -------
        npt.NDArray[Any]
            Independent values first, then dependent values.
        """
        x_independent = np.asarray(x_independent, dtype=object).ravel()

        param_values: dict[str, Any] = {}
        for param, value in zip(self.independent_parameters, x_independent):
            param_values[param.name] = value

        changed = True
        while changed:
            changed = False
            for dep in self.parameter_dependencies:
                if dep.dependent_parameter.name in param_values:
                    continue
                try:
                    indep_vals = [param_values[p.name]
                                  for p in dep.independent_parameters
                                  ]
                except KeyError:
                    continue
                val = dep.transform(*indep_vals)
                if param_values.get(dep.dependent_parameter.name) != val:
                    param_values[dep.dependent_parameter.name] = val
                    changed = True

        ordered = self.independent_parameters + self.dependent_parameters
        return np.asarray([param_values[p.name] for p in ordered], dtype=object)
