from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from CADETProcess.dataStructure import (
    check_nested,
    get_nested_value,
    get_nested_attribute,
    generate_nested_dict,
)


@dataclass
class ParameterBase:
    """
    Abstract base class for all parameters.

    Subclasses must implement the `validate` method.

    Attributes
    ----------
    name : str
        Name of the parameter.
    dependencies : list of ParameterBase or None
        Other parameters this parameter depends on.
    transform : callable or None
        Transformation function to derive the value from dependencies.
    """

    name: str
    dependencies: list[ParameterBase] | None = None
    transform: callable | None = None

    def validate(self, value):
        """
        Validate a value against the parameter's constraints.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        NotImplementedError
            Always, unless overridden in subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement `validate` to enforce parameter rules."
        )


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
    """

    parameter_type: type[float | int] = int
    lb: float = -math.inf
    ub: float = math.inf

    def __post_init__(self):
        """
        Validate bounds after initialization.

        Raises
        ------
        ValueError
            If lower bound is greater than or equal to upper bound.
        """
        if self.lb >= self.ub:
            raise ValueError("Lower bound must be < upper bound.")

    def validate(self, value):
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

    def validate(self, value):
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

    def __post_init__(self):
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

    def __post_init__(self):
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
class ParameterMapping:
    """
    Maps a parameter to an evaluation path and function.

    Attributes
    ----------
    param : ParameterBase
        The parameter being mapped.
    path : str
        A path or key to the parameter's location.
    eval_obj : callable
        Function used to evaluate the mapped parameter.
    """

    param: ParameterBase
    path: str
    eval_obj: callable




class ParameterSpace:
    """
    Container for managing parameters and constraints.

    Provides methods to add parameters and define linear relationships.
    """

    def __init__(self):
        self._parameters: list[ParameterBase] = []
        self._linear_constraints: list[LinearConstraint] = []
        self._linear_equality_constraints = []

    @property
    def parameters(self) -> list[ParameterBase]:
        """
        Get all registered parameters.

        Returns
        -------
        list of ParameterBase
            All parameters in the space.
        """
        return self._parameters

    @property
    def dependent_parameters(self) -> list[ParameterBase]:
        """
        Get all dependent parameters.

        Returns
        -------
        list of ParameterBase
            Parameters that depend on others.
        """
        return [param for param in self._parameters if param.dependencies]

    def add_parameter(self, parameter: ParameterBase):
        """
        Add a parameter to the parameter space.

        Ensures that parameter is valid, not duplicated, and has a unique name.

        Raises
        ------
        TypeError
            If the object is not an instance of ParameterBase.
        ValueError
            If a parameter with the same name already exists.
        """
        if not isinstance(parameter, ParameterBase):
            raise TypeError(f"Expected ParameterBase, got {type(parameter).__name__}")

        if any(p.name == parameter.name for p in self._parameters):
            raise ValueError(f"Parameter name '{parameter.name} already used.")

        self._parameters.append(parameter)

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """
        Return all defined linear inequality constraints.

        Returns
        -------
        list of LinearConstraint
            List of inequality constraints.
        """
        return self._linear_constraints

    @property
    def linear_equality_constraints(self) -> list[LinearEqualityConstraint]:
        """
        Return all defined linear equality constraijnts.

        Returns
        -------
        list of LinearEqualityConstraint
            List of equality constraints.
        """
        return self._linear_equality_constraints

    def add_linear_constraint(self, parameters, lhs, b):
        """
        Add a linear constraint of the form a · x <= b.

        Parameters
        ----------
        parameters : list of RangedParameter or RangedParameter
            Parameters involved in the constraint.
        lhs : float or list of float
            Coefficients for each parameter.
        b : float
            Right-hand side of the constraint.
        """
        if not isinstance(parameters, list):
            parameters = [parameters]
        if not isinstance(lhs, list):
            lhs = [lhs]

        if len(parameters) != len(lhs):
            raise ValueError(
                "The number of parameters must match the number of coefficients (lhs)")

        constraint = LinearConstraint(parameters=parameters, lhs=lhs, b=b)
        self._linear_constraints.append(constraint)

    def add_linear_equality_constraint(self, parameters, lhs, b):
        """
        Add a linear equality uaconstraint of the form a · x = b.

        Parameters
        ----------
        parameters : list of RangedParameter or RangedParameter
            Parameters involved in the constraint.
        lhs : float or list of float
            Coefficients for each parameter.
        b : float
            Right-hand side of the constraint.
        """
        if not isinstance(parameters, list):
            parameters = [parameters]
        if not isinstance(lhs, list):
            lhs = [lhs]

        if len(parameters) != len(lhs):
            raise ValueError
            ("The number of parameters must match the number of coefficients (lhs)")

        constraint = LinearEqualityConstraint(parameters=parameters, lhs=lhs, b=b)
        self._linear_equality_constraints.append(constraint)
