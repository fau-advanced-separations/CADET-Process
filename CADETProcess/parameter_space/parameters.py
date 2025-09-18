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
    cur = root
    for k in keys[:-1]:
        cur = cur[k] if isinstance(cur, Mapping) else getattr(cur, k)
    return cur, keys[-1]


@dataclass(slots=True)
class ParameterMapperBase:
    """Abstract base that writes a single parameter into multiple targets."""

    evaluation_objects: list[Any] = field(repr=False)

    def set_value(self, value: Any) -> None:
        """
        Broadcast *value* into every object listed in ``self.evaluation_objects``.

        Parameters
        ----------
        value : Any
            The parameter value to write into each evaluation object.
        """
        for obj in self.evaluation_objects:
            self._set_value(obj, value)


@dataclass(slots=True)
class ParameterDotPathSetter(ParameterMapperBase):
    """
    Writes *value* to the leaf referenced by a dot-separated path.

    Traverses dicts **and** objects in the same path; missing hops raise.
    """

    path: str

    _keys: Sequence[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._keys = self.path.split(".")

    def _set_value(self, evaluation_object: Any, value: Any) -> None:
        parent, leaf = _traverse_mixed_path(evaluation_object, self._keys)
        if isinstance(parent, Mapping):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)


@dataclass
class ParameterSpace:
    """Container for managing parameters and constraints.

    @TODO add methods to check and evaluate bounds and linear cons
    """

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

    @property
    def n_variables(self) -> int:
        """Number of independent (optimization) variables."""
        return len(self.independent_parameters)

    @property
    def lower_bounds(self) -> np.ndarray:
        """
        Lower bounds for independent variables.

        Returns
        -------
        np.ndarray
            Vector of length `n_variables`; parameters without explicit bounds are
            treated as unbounded (-inf).
        """
        lbs: list[float] = []
        for p in self.independent_parameters:
            lb = getattr(p, "lb", None)
            lbs.append(-np.inf if lb is None else float(lb))
        return np.asarray(lbs, dtype=float)

    @property
    def upper_bounds(self) -> np.ndarray:
        """
        Upper bounds for independent variables.

        Returns
        -------
        np.ndarray
            Vector of length `n_variables`; parameters without explicit bounds are
            treated as unbounded (+inf).
        """
        ubs: list[float] = []
        for p in self.independent_parameters:
            ub = getattr(p, "ub", None)
            ubs.append(+np.inf if ub is None else float(ub))
        return np.asarray(ubs, dtype=float)

    def check_bounds(
        self,
        x: npt.ArrayLike,
        cv_bounds_tol: Optional[float | npt.ArrayLike] = 0.0,
    ) -> bool:
        """
        Check if all bound constraints are satisfied for the **independent** variables.

        Parameters
        ----------
        x : ArrayLike
            Values of the independent optimization variables (untransformed space)
            in the order of ``self.independent_parameters``.
        cv_bounds_tol : float or ArrayLike, optional
            Tolerance for checking bound constraints. If a scalar is provided, the
            same tolerance is applied to all variables; otherwise must match
            ``n_variables``. Default is 0.0.

        Returns
        -------
        flag : bool
            ``True`` if every variable satisfies ``lb - tol <= x <= ub + tol``,
            ``False`` otherwise.

        Raises
        ------
        ValueError
            If the length of ``x`` (or of ``cv_bounds_tol`` when array-like) does
            not match the number of independent variables.

        """
        vals = np.asarray(x, dtype=float).ravel()
        n = self.n_variables
        if vals.size != n:
            raise ValueError(f"Length of `x` ({vals.size}) does not match {n}.")

        if np.isscalar(cv_bounds_tol):
            tol = np.full(n, float(cv_bounds_tol), dtype=float)
        else:
            tol = np.asarray(cv_bounds_tol, dtype=float).ravel()
            if tol.size != n:
                raise ValueError(
                    f"Length of `cv_bounds_tol` ({tol.size}) does not match {n}."
                )

        lbs = self.lower_bounds
        ubs = self.upper_bounds

        below = vals < (lbs - tol)
        above = vals > (ubs + tol)
        return not (np.any(below) or np.any(above))

    @property
    def ordered_parameters(self) -> list[ParameterBase]:
        """
        Independent parameters followed by dependent parameters.

        This mirrors the ordering returned by `get_dependent_variables`.
        """
        return self.independent_parameters + self.dependent_parameters

    def set_values(
        self,
        x_independent: npt.ArrayLike,
        *,
        compute_dependents: bool = True,
        validate_bounds: bool = False,
        cv_bounds_tol: Optional[float | npt.ArrayLike] = 0.0,
    ) -> None:
        """
        Set values on all parameters (independent first, then dependent).

        Parameters
        ----------
        x_independent : ArrayLike
            Values for the independent parameters, in the order of
            `self.independent_parameters`.
        compute_dependents : bool, default=True
            If True (recommended), compute dependent parameter values using
            `get_dependent_variables` before setting. If False, `x_independent`
            is assumed to already contain values for *all* parameters in the
            order `independent + dependent`.
        validate_bounds : bool, default=False
            If True, check bound feasibility on the independent variables using
            `check_bounds` before setting.
        cv_bounds_tol : float or ArrayLike, default=0.0
            Tolerance passed to `check_bounds` when `validate_bounds` is True.

        Notes
        -----
        - This calls each parameter's own `set_value`, which in turn runs its
          `validate` method and writes into its mappers.
        - Ordering: independent parameters first, then dependent parameters.
        """
        if validate_bounds:
            ok = self.check_bounds(x_independent, cv_bounds_tol=cv_bounds_tol)
            if not ok:
                raise ValueError("Independent values violate bound constraints.")

        if compute_dependents:
            all_values = self.get_dependent_variables(x_independent)
            if all_values.size != len(self.ordered_parameters):
                raise RuntimeError(
                    "Computed number of parameter values does not match the number of parameters."
                )
        else:
            all_values = np.asarray(x_independent, dtype=object).ravel()
            expected = len(self.ordered_parameters)
            if all_values.size != expected:
                raise ValueError(
                    f"When compute_dependents=False, expected {expected} values "
                    f"(independent + dependent), got {all_values.size}."
                )

        for param, value in zip(self.ordered_parameters, all_values):
            param.set_value(value)

    @property
    def evaluation_objects(self) -> set[Any]:
        """
        Unique evaluation objects gathered from all parameter mappers.

        Returns
        -------
        set[Any]
            A set of unique evaluation objects found across all parameters'
            mappers.
        """
        return {
            evaluation_object
            for p in self.parameters if p.mappers
            for m in p.mappers if m.evaluation_objects
            for evaluation_object in m.evaluation_objects
        }
