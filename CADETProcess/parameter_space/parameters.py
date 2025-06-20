from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

__all__ = [
    "ParameterBase",
    "RangedParameter",
    "ChoiceParameter",
    "LinearConstraint",
    "LinearEqualityConstraint",
    "ParameterMapping",
    "ParameterSpace",
]


@dataclass
class ParameterDependency:
    """
    Class to specify a parameter depdendence.

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
class ParameterBase(ABC):
    """
    Abstract base class for all parameters.

    Subclasses must implement the `validate` method.

    Attributes
    ----------
    name : str
        Name of the parameter.
    """

    name: str
    dependency: ParameterDependency | None = None

    @abstractmethod
    def validate(self, value: Any) -> None:
        """
        Validate a value against the parameter's constraints.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        bool
            True if value is valid. False otherwise

        Notes
        -----
        Subclasses must implement `validate` to enforce parameter rules.
        Changed it to return None rather than bool since it raises exceptions
        in the subclass method and do not return a bool.
        Also, apparantly, it is not standard practice that abstract base classes
        and dataclasses should be used together. i.e, an abstract base class should
        not also be simultaneously a data class.

        """
        pass

    def set_value(self, value: Any) -> None:
        """
        Set the value in the mapped objects.

        Iterates over parameter mappers and sets the value.
        TODO: implement this method.
        TODO: What's the purpose of this method? Call the mapper?
        TODO: Raise error if value is not independent?
        """
        pass


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
    normalization : Literal["auto", "linear", "normal"]
        Method to normalize parameter.
    TODO: Is normalization a Literal or an instance of some NormalizationBase?
    """

    parameter_type: type[float | int] = int
    lb: float = -math.inf
    ub: float = math.inf
    normalization: Literal["auto", "linear", "normal"] | None = None

    def __post_init__(self) -> None:
        """
        Validate bounds after initialization.

        Raises
        ------
        ValueError
            If lower bound is greater than or equal to upper bound.
        """
        if self.lb >= self.ub:
            raise ValueError("Lower bound must be < upper bound.")

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

    def normalize(self) -> None:
        """Normalize parameter."""
        pass


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
class BaseLinearConstraint(ABC):
    """
    Base class for linear constraints.

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
    lhs: list[float] = field(default_factory=list)
    b: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, list):
            self.parameters = [self.parameters]
        if not self.lhs or np.isscalar(self.lhs):
            self.lhs = [float(self.lhs or 1.0)] * len(self.parameters)
        if len(self.lhs) != len(self.parameters):
            raise ValueError("Length of lhs must match number of parameters.")
        self.b = float(self.b)

    @abstractmethod
    def is_satisfied(self, values: list[float]) -> bool:
        pass


@dataclass
class LinearConstraint(BaseLinearConstraint):
    """Linear inequality: lhs · parameters <= b."""

    def is_satisfied(self, values: list[float]) -> bool:
        """
        Check whether the linear inequality constraint is satisfied.

        Evaluates if the constraint is satisfied given the provided parameter values.

        Parameters
        ----------
        values : list of float
            The values of the parameters involved in this constraint.
            The order should correspond to the order of `self.parameters`.

        Returns
        -------
        bool
            True if the constraint is satisfied (i.e., lhs · values <= b),
            false otherwise.
        """
        lhs_value = sum(lhs_coef * v for lhs_coef, v in zip(self.lhs, values))
        return lhs_value <= self.b


@dataclass
class LinearEqualityConstraint(BaseLinearConstraint):
    """Linear equality: lhs · parameters == b."""

    def is_satisfied(self, values: list[float]) -> bool:
        """
        Check whether the linear equality constraint is satisfied.

        Evaluates if the constraint is satisfied given the provided parameter values.

        Parameters
        ----------
        values : list of float
            The values of the parameters involved in this constraint.
            The order should correspond to the order of `self.parameters`.

        Returns
        -------
        bool
            True if the constraint is satisfied (i.e., lhs · values == b),
            false otherwise.
        """
        lhs_value = sum(lhs_coef * v for lhs_coef, v in zip(self.lhs, values))
        return np.isclose(lhs_value, self.b)


@dataclass
class ParameterMapping:
    """
    Maps a parameter to an evaluation path and function.

    Attributes
    ----------
    parameter : ParameterBase
        The parameter being mapped.
    evaluation_objects : list[Any]
        The objects to which the parameter is mapped.
    setter: Callable
        Function to set mapped parameter.
    """

    parameter: ParameterBase
    evaluation_objects: list[Any]
    setter: Callable


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

        TODO: Check that all parameters are actually part of the parameter space.
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

        TODO: Check that all parameters are actually part of the parameter space.
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
    def _parameter_dependencies(self) -> dict[ParameterBase, ParameterDependency]:
        return {dep.dependent_parameter: dep for dep in self.parameter_dependencies}

    @property
    def dependent_parameters(self) -> list[ParameterBase]:
        """
        Get all dependent parameters.

        Returns
        -------
        list of ParameterBase
            Parameters that depend on others.
        """
        return list(self._parameter_dependencies.keys())

    @property
    def independent_parameters(self) -> list[ParameterBase]:
        """
        Get all independent parameters.

        Returns
        -------
        list of ParameterBase
            Parameters that are independent of others.
        """
        return [p for p in self.parameters if p not in self._parameter_dependencies]
