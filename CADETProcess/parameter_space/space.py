"""
ParameterSpace: parameter semantics, feasibility, and evaluation object ownership.

``ParameterSpace`` owns the evaluation objects and parameters. Mappers are wired
when parameters are added and stored internally; callers interact only with the
high-level API. ``set_values`` resolves dependent parameters, validates values,
and writes through the wired mappers.

Coordinate convention
---------------------
**Independent-only is the canonical input format** at the ``OptimizationProblem``
level: all ``evaluate_*`` and ``check_*`` methods there default to
independent-only vectors (length ``n_variables``) and resolve dependent
parameters internally.  Pass ``get_dependent_values=False`` when supplying a
full parameter vector to skip dependency resolution.

At the ``ParameterSpace`` level the convention differs.  Read and inspection
methods decorated with ``@resolves_dependencies`` accept a full parameter
vector by default and expand an independent-only vector when called with
``resolve_dependencies=True``.  This reflects the fact that bounds and linear
constraints are evaluated on the full physical vector (``A @ x``).

Write methods such as ``set_values`` accept **independent values only** as the
source of truth.  Dependent parameters are always recomputed internally,
validated as part of the resulting full vector, and then written through the
wired mappers.  Caller-supplied dependent values are not accepted.

Use ``get_dependent_values(x_independent)`` to explicitly expand an independent
vector to the full physical vector for inspection or storage.

Three callers, three spaces:

* **User** — physical units, named parameters.
* **Optimizer** — normalized independent variables in ``[0, 1]``; use
  ``TransformedSpace``.
* **Internal sampling/population** — independent variables for sampling and
  optimization; full physical vectors may be materialized afterward for
  validation, storage, or reporting.

Normalization
-------------
``normalize`` and ``denormalize`` are thin element-wise utilities. They apply
each ``RangedParameter`` normalizer and always map between physical
``ParameterSpace`` bounds and ``[0, 1]``.

Constraints are expressed in physical units and are not automatically
transformed. A linear physical constraint involving a variable with a nonlinear
normalizer becomes nonlinear in normalized coordinates. Such combinations raise
when transformed constraint matrices are assembled by ``TransformedSpace``; see
``CADETProcess.parameter_space.transformed_space``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from CADETProcess.parameter_space.transformed_space import TransformedSpace

import numpy as np
import numpy.typing as npt

from CADETProcess.numerics import round_to_significant_digits
from CADETProcess.parameter_space.constraints import (
    LinearConstraint,
    LinearEqualityConstraint,
)
from CADETProcess.parameter_space.dependencies import ParameterDependency
from CADETProcess.parameter_space.mappers import (
    CallableMapper,
    DotPathMapper,
    ParameterMapperBase,
)
from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    ParameterBase,
    RangedParameter,
)

__all__ = ["ParameterSpace"]


class ParameterSpace:
    """Container for parameters, evaluation objects, constraints, and dependencies.

    The space is the interface between an optimizer (or sampler) and a set of model
    objects.  It has two responsibilities: defining the feasible domain and writing a
    parameter vector *x* into the evaluation objects.

    Typical usage::

        space = ParameterSpace()
        space.add_evaluation_object(process)
        space.add_parameter(
            RangedParameter("length", float, lb=0.1, ub=1.0),
            path="column.length",
        )
        space.set_values({"length": 0.5})
    """

    # ── Method decorators ─────────────────────────────────────────────────────

    def denormalizes(func: Callable) -> Callable:  # type: ignore[misc]
        """Adapter for methods that operate on physical coordinates.

        The decorated method always receives *x* in physical units.  The
        decorator adds a ``denormalize=True`` entry point that converts
        normalized coordinates to physical before invoking the method, without
        changing what the implementation sees or its contract.

        Apply to methods that are part of the optimizer-facing interface but
        implemented by ``ParameterSpace`` — i.e., methods that
        ``TransformedSpace`` naturally delegates to with ``denormalize=True``.

        The ``denormalize`` parameter is injected into the wrapped function's
        ``__signature__`` so that ``help()`` and IDE introspection show it.
        """

        @wraps(func)
        def denormalizes_wrapper(
            self: Any, x: Any, *args: Any, denormalize: bool = False, **kwargs: Any
        ) -> Any:
            if denormalize:
                x = self.denormalize(x)
            return func(self, x, *args, **kwargs)

        original = inspect.signature(func)
        denormalize_param = inspect.Parameter(
            "denormalize", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool
        )
        new_params = list(original.parameters.values()) + [denormalize_param]
        denormalizes_wrapper.__signature__ = original.replace(parameters=new_params)
        return denormalizes_wrapper

    def resolves_dependencies(func: Callable) -> Callable:  # type: ignore[misc]
        """Adapter for methods that operate on a fully resolved parameter vector.

        The decorated method always receives a full parameter vector of length
        ``n_parameters`` (independent + dependent).  The decorator adds a
        ``resolve_dependencies=True`` entry point that expands an independent-
        only vector via ``self.get_dependent_values`` before invoking the
        method, without changing what the implementation sees or its contract.

        Hot paths that have already resolved dependencies pass the full vector
        directly (``resolve_dependencies=False``, the default) to avoid
        redundant resolution.

        The ``resolve_dependencies`` parameter is injected into the wrapped
        function's ``__signature__`` so that ``help()`` and IDE introspection
        show it.
        """

        @wraps(func)
        def resolves_dependencies_wrapper(
            self: Any, x: Any, *args: Any, resolve_dependencies: bool = False, **kwargs: Any
        ) -> Any:
            if resolve_dependencies:
                x = self.get_dependent_values(x)
            return func(self, x, *args, **kwargs)

        original = inspect.signature(func)
        resolve_param = inspect.Parameter(
            "resolve_dependencies",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool,
        )
        new_params = list(original.parameters.values()) + [resolve_param]
        resolves_dependencies_wrapper.__signature__ = original.replace(parameters=new_params)
        return resolves_dependencies_wrapper

    def __init__(self) -> None:
        self._evaluation_objects: list[Any] = []
        self._parameters: list[ParameterBase] = []
        self._mappers: dict[str, ParameterMapperBase] = {}
        self._linear_constraints: list[LinearConstraint] = []
        self._linear_equality_constraints: list[LinearEqualityConstraint] = []
        self._dependencies: list[ParameterDependency] = []
        self._transformed_space_cache: Any = None

    @property
    def transformed_space(self) -> "TransformedSpace":
        """Normalized optimizer view of this space.

        Lazily constructed and cached; invalidated whenever parameters,
        dependencies, or constraints are added.
        """
        from CADETProcess.parameter_space.transformed_space import TransformedSpace

        if self._transformed_space_cache is None:
            self._transformed_space_cache = TransformedSpace(self)
        return self._transformed_space_cache

    # ── Evaluation objects ────────────────────────────────────────────────────

    def add_evaluation_object(self, obj: Any) -> None:
        """Register an evaluation object with the space.

        All subsequent ``add_parameter`` calls without an explicit
        ``evaluation_objects`` argument will target this object.

        Parameters
        ----------
        obj : Any
            A picklable model object (process, flowsheet, …).

        Raises
        ------
        ValueError
            If *obj* is already registered.
        """
        if any(obj is existing for existing in self._evaluation_objects):
            raise ValueError(f"Evaluation object {obj!r} is already registered.")
        self._evaluation_objects.append(obj)

    @property
    def evaluation_objects(self) -> list[Any]:
        """Registered evaluation objects, in insertion order."""
        return list(self._evaluation_objects)

    # ── Parameters ───────────────────────────────────────────────────────────

    def add_parameter(
        self,
        parameter: ParameterBase,
        *,
        path: Optional[str] = None,
        evaluation_objects: Optional[list[Any]] = None,
        mapper: Optional[ParameterMapperBase] = None,
    ) -> None:
        """Register a parameter and wire its mapper.

        Exactly one of *path* or *mapper* may be supplied.  Supplying neither
        registers a parameter with no write target, which is valid for parameters
        that only participate in dependency transforms and whose value need not be
        written to any object.

        Parameters
        ----------
        parameter : ParameterBase
            The parameter to register.
        path : str, optional
            Dot-separated attribute/key path.  A ``DotPathMapper`` is created
            automatically and targets ``evaluation_objects`` (or all registered
            objects if not specified).
        evaluation_objects : list, optional
            Subset of evaluation objects this parameter maps to.  Only valid
            together with *path*.
        mapper : ParameterMapperBase, optional
            Pre-built mapper.  Use this (or ``add_parameter_with_callable``) when
            a dot-path is insufficient.

        Raises
        ------
        ValueError
            If a parameter with the same name is already registered.
        ValueError
            If both *path* and *mapper* are supplied.
        ValueError
            If *evaluation_objects* is given without *path*.
        ValueError
            If *evaluation_objects* contains objects not registered with the space.
        """
        if any(p.name == parameter.name for p in self._parameters):
            raise ValueError(
                f"A parameter named {parameter.name!r} is already registered."
            )
        if path is not None and mapper is not None:
            raise ValueError("Supply at most one of 'path' or 'mapper', not both.")
        if evaluation_objects is not None and path is None:
            raise ValueError("'evaluation_objects' requires 'path'.")

        if path is not None:
            targets = self._resolve_targets(evaluation_objects)
            mapper = DotPathMapper(targets, path)

        self._parameters.append(parameter)
        if mapper is not None:
            self._mappers[parameter.name] = mapper
        self._transformed_space_cache = None

    def add_parameter_with_callable(
        self,
        parameter: ParameterBase,
        fn: Callable[[Any, Any], None],
        *,
        evaluation_objects: Optional[list[Any]] = None,
    ) -> None:
        """Register a parameter whose write is handled by a callable.

        Convenience wrapper around ``add_parameter`` with a ``CallableMapper``.

        Parameters
        ----------
        parameter : ParameterBase
            The parameter to register.
        fn : Callable[[Any, Any], None]
            Called as ``fn(obj, value)`` for each evaluation object.
        evaluation_objects : list, optional
            Subset of evaluation objects to target.
        """
        targets = self._resolve_targets(evaluation_objects)
        self.add_parameter(parameter, mapper=CallableMapper(targets, fn))

    def _resolve_targets(self, evaluation_objects: Optional[Any]) -> list[Any]:
        """Return the target list, defaulting to all registered objects."""
        if evaluation_objects is None:
            if not self._evaluation_objects:
                raise ValueError(
                    "No evaluation objects are registered. "
                    "Call add_evaluation_object() before add_parameter(), "
                    "or pass evaluation_objects explicitly."
                )
            return list(self._evaluation_objects)
        if not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        unknown = [o for o in evaluation_objects if o not in self._evaluation_objects]
        if unknown:
            raise ValueError(
                f"The following objects are not registered with this space: {unknown!r}. "
                "Call add_evaluation_object() first."
            )
        return list(evaluation_objects)

    @property
    def parameters(self) -> list[ParameterBase]:
        """All registered parameters, in insertion order."""
        return list(self._parameters)

    @property
    def n_parameters(self) -> int:
        """Total number of registered parameters (independent + dependent)."""
        return len(self._parameters)

    # ── Dependencies ──────────────────────────────────────────────────────────

    def add_dependency(
        self,
        dependent: ParameterBase,
        independent_parameters: list[ParameterBase],
        transform: Callable,
    ) -> None:
        """Declare that *dependent* is computed from *independent_parameters*.

        The optimizer sees only independent parameters; ``set_values`` resolves
        dependent parameters before writing.  Chains are supported; cycles raise
        at declaration time.

        Note: bounds and linear constraints on *dependent* cannot be enforced
        before the transform is evaluated.  They are checked lazily in
        ``set_values`` via each parameter's ``validate`` method.  See the module
        docstring for the recommended sampling pattern.

        Parameters
        ----------
        dependent : ParameterBase
            The parameter whose value is computed.
        independent_parameters : list[ParameterBase]
            Parameters whose values are passed to *transform*, in order.
        transform : Callable
            Called as ``transform(*values)``; must return the value for *dependent*.

        Raises
        ------
        ValueError
            If *dependent* already has a dependency declared.
        ValueError
            If any parameter is not registered with this space.
        ValueError
            If the dependency would introduce a cycle.
        """
        all_names = {p.name for p in self._parameters}
        for p in [dependent, *independent_parameters]:
            if p.name not in all_names:
                raise ValueError(
                    f"Parameter {p.name!r} is not registered with this space."
                )
        if any(d.dependent_parameter.name == dependent.name for d in self._dependencies):
            raise ValueError(
                f"Parameter {dependent.name!r} already has a dependency declared."
            )

        self._dependencies.append(
            ParameterDependency(
                dependent_parameter=dependent,
                independent_parameters=list(independent_parameters),
                transform=transform,
            )
        )
        self._check_no_cycles()
        self._transformed_space_cache = None

    def _check_no_cycles(self) -> None:
        """Raise ValueError if the dependency graph contains a cycle."""
        dep_map: dict[str, list[str]] = {
            d.dependent_parameter.name: [p.name for p in d.independent_parameters]
            for d in self._dependencies
        }
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(name: str) -> None:
            if name in in_stack:
                raise ValueError(
                    f"Cycle detected in parameter dependencies involving {name!r}."
                )
            if name in visited:
                return
            in_stack.add(name)
            for parent in dep_map.get(name, []):
                dfs(parent)
            in_stack.discard(name)
            visited.add(name)

        for name in dep_map:
            dfs(name)

    @property
    def dependent_parameters(self) -> list[ParameterBase]:
        """Parameters that are computed from others, in registration order."""
        dependent_names = {d.dependent_parameter.name for d in self._dependencies}
        return [p for p in self._parameters if p.name in dependent_names]

    @property
    def independent_parameters(self) -> list[ParameterBase]:
        """Parameters not dependent on others; these form the optimizer input."""
        dependent_names = {d.dependent_parameter.name for d in self._dependencies}
        return [p for p in self._parameters if p.name not in dependent_names]

    @property
    def n_variables(self) -> int:
        """Number of independent (optimizer-facing) variables."""
        return len(self.independent_parameters)

    @property
    def continuous_parameters(self) -> list[RangedParameter]:
        """Independent float-typed parameters."""
        return [
            p for p in self.independent_parameters
            if isinstance(p, RangedParameter) and p.parameter_type is float
        ]

    @property
    def integer_parameters(self) -> list[RangedParameter]:
        """Independent integer-typed parameters."""
        return [
            p for p in self.independent_parameters
            if isinstance(p, RangedParameter) and p.parameter_type is int
        ]

    @property
    def categorical_parameters(self) -> list[ChoiceParameter]:
        """Independent categorical (choice) parameters."""
        return [
            p for p in self.independent_parameters
            if isinstance(p, ChoiceParameter)
        ]

    def resolve(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve a named independent assignment to the full parameter assignment.

        Dependent parameter values are computed from the independent values in
        topological order.

        Parameters
        ----------
        assignment : Mapping
            Values for the independent parameters by name, in physical units.
            Order-insensitive; the result is ordered by registration.

        Returns
        -------
        dict
            Values for all parameters (independent and dependent), ordered by
            registration.

        Raises
        ------
        TypeError
            If *assignment* is not a Mapping.
        ValueError
            If *assignment* contains unknown names, misses an independent
            parameter, or supplies a value for a dependent parameter.
        RuntimeError
            If dependency declarations leave a parameter unresolvable.
        """
        if not isinstance(assignment, Mapping):
            raise TypeError(
                "resolve takes a named assignment (Mapping of parameter name to "
                "value); numeric vectors are an encoding owned by "
                "TransformedSpace — decode first: "
                "space.resolve(space.transformed_space.decode(x))."
            )
        independent = self.independent_parameters
        independent_names = {p.name for p in independent}
        registered_names = {p.name for p in self._parameters}
        unknown = [n for n in assignment if n not in registered_names]
        if unknown:
            raise ValueError(f"Unknown parameter names: {unknown!r}.")
        dependent_supplied = [n for n in assignment if n not in independent_names]
        if dependent_supplied:
            raise ValueError(
                f"Assignment supplies values for dependent parameters "
                f"{dependent_supplied!r}; dependent values are resolved, not set."
            )
        missing = [p.name for p in independent if p.name not in assignment]
        if missing:
            raise ValueError(
                f"Assignment misses independent parameters: {missing!r}."
            )

        values: dict[str, Any] = dict(assignment)
        changed = True
        while changed:
            changed = False
            for dep in self._dependencies:
                name = dep.dependent_parameter.name
                if name in values:
                    continue
                try:
                    args = [values[p.name] for p in dep.independent_parameters]
                except KeyError:
                    continue
                values[name] = dep.transform(*args)
                changed = True
        unresolved = [p.name for p in self._parameters if p.name not in values]
        if unresolved:
            raise RuntimeError(
                f"Could not resolve parameter values for {unresolved!r}. "
                "Check dependency declarations."
            )
        return {p.name: values[p.name] for p in self._parameters}

    def _resolve_all_values(self, x_independent: npt.ArrayLike) -> dict[str, Any]:
        """Resolve an independent-value vector; vector adapter over ``resolve``.

        Positions follow the registration order of the independent parameters.
        """
        x = np.asarray(x_independent, dtype=object).ravel()
        if x.size != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} independent values, got {x.size}."
            )
        return self.resolve(
            {p.name: v for p, v in zip(self.independent_parameters, x)}
        )

    @denormalizes
    def get_dependent_values(self, x_independent: npt.ArrayLike) -> np.ndarray:
        """Expand independent parameter values to the full parameter vector.

        Parameters
        ----------
        x_independent : array-like
            Values for the ``n_variables`` independent parameters in physical units.
            Pass ``denormalize=True`` to denormalize from normalized coordinates first.

        Returns
        -------
        np.ndarray
            Full parameter vector of length ``n_parameters``, in registration order
            (independent parameters first, then dependent parameters resolved from them).
        """
        all_values = self._resolve_all_values(x_independent)
        return np.array([all_values[p.name] for p in self._parameters], dtype=float)

    # ── Constraints ───────────────────────────────────────────────────────────

    def _validate_constraint_parameters(
        self, constraint: LinearConstraint | LinearEqualityConstraint
    ) -> None:
        """Raise if any constrained parameter is not a registered RangedParameter."""
        registered = {p.name: p for p in self._parameters}
        for p in constraint.parameters:
            if p.name not in registered:
                raise ValueError(
                    f"Constraint parameter {p.name!r} is not registered with this space."
                )
            if not isinstance(registered[p.name], RangedParameter):
                raise TypeError(
                    f"Constraint parameter {p.name!r} is not a RangedParameter; "
                    "linear constraints require numeric parameters."
                )

    def add_linear_constraint(self, constraint: LinearConstraint) -> None:
        """Register a linear inequality constraint on independent parameters."""
        self._validate_constraint_parameters(constraint)
        self._linear_constraints.append(constraint)
        self._transformed_space_cache = None

    def add_linear_equality_constraint(
        self, constraint: LinearEqualityConstraint
    ) -> None:
        """Register a linear equality constraint on independent parameters."""
        self._validate_constraint_parameters(constraint)
        self._linear_equality_constraints.append(constraint)
        self._transformed_space_cache = None

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """Registered linear inequality constraints."""
        return list(self._linear_constraints)

    @property
    def linear_equality_constraints(self) -> list[LinearEqualityConstraint]:
        """Registered linear equality constraints."""
        return list(self._linear_equality_constraints)

    def _assemble_physical_matrices(
        self,
        constraints: list[LinearConstraint] | list[LinearEqualityConstraint],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A, b)`` in physical coordinates over *all* registered parameters.

        Columns correspond to ``self.parameters`` (independent + dependent, in
        registration order).  Coefficients are placed at each referenced
        parameter's column index; no coordinate transform is applied.

        Use ``A_independent`` / ``A_eq_independent`` to obtain the independent-
        variable slice suitable for polytope samplers such as hopsy.
        """
        n = self.n_parameters
        n_c = len(constraints)
        if n_c == 0:
            return np.zeros((0, n)), np.zeros(0)
        param_index = {p.name: i for i, p in enumerate(self._parameters)}
        A = np.zeros((n_c, n))
        b = np.empty(n_c)
        for row_idx, constraint in enumerate(constraints):
            for p, coeff in zip(constraint.parameters, constraint.lhs):
                A[row_idx, param_index[p.name]] += coeff
            b[row_idx] = constraint.b
        return A, b

    @property
    def _independent_indices(self) -> list[int]:
        """Column indices of independent parameters in ``self.parameters``."""
        ind_names = {p.name for p in self.independent_parameters}
        return [i for i, p in enumerate(self._parameters) if p.name in ind_names]

    @property
    def A(self) -> np.ndarray:
        """Linear inequality constraint matrix, shape ``(m, n_parameters)``."""
        return self._assemble_physical_matrices(self._linear_constraints)[0]

    @property
    def b(self) -> np.ndarray:
        """Linear inequality constraint RHS in physical coordinates, shape ``(m,)``."""
        return self._assemble_physical_matrices(self._linear_constraints)[1]

    @property
    def A_independent(self) -> np.ndarray:
        """Inequality constraint matrix, independent columns only, shape ``(m, n_variables)``."""
        return self.A[:, self._independent_indices]

    @property
    def A_eq(self) -> np.ndarray:
        """Linear equality constraint matrix, shape ``(m, n_parameters)``."""
        return self._assemble_physical_matrices(self._linear_equality_constraints)[0]

    @property
    def b_eq(self) -> np.ndarray:
        """Linear equality constraint RHS in physical coordinates, shape ``(m,)``."""
        return self._assemble_physical_matrices(self._linear_equality_constraints)[1]

    @property
    def A_eq_independent(self) -> np.ndarray:
        """Equality constraint matrix sliced to independent columns, shape ``(m, n_variables)``."""
        return self.A_eq[:, self._independent_indices]

    @resolves_dependencies
    def evaluate_linear_constraints(self, x: npt.ArrayLike) -> np.ndarray:
        """Return ``A @ x - b``; positive entries mean a constraint violation.

        Parameters
        ----------
        x : array-like
            Full parameter vector (length ``n_parameters``) in physical units.
            Pass ``resolve_dependencies=True`` to expand an independent-only vector first.
        """
        x = np.asarray(x, dtype=float).ravel()
        return self.A @ x - self.b

    @resolves_dependencies
    def evaluate_linear_equality_constraints(self, x: npt.ArrayLike) -> np.ndarray:
        """Return ``A_eq @ x - b_eq``; non-zero entries mean a constraint violation.

        Parameters
        ----------
        x : array-like
            Full parameter vector (length ``n_parameters``) in physical units.
            Pass ``resolve_dependencies=True`` to expand an independent-only vector first.
        """
        x = np.asarray(x, dtype=float).ravel()
        return self.A_eq @ x - self.b_eq

    # ── Bounds ────────────────────────────────────────────────────────────────

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds for all parameters (independent + dependent); ``-inf`` when unbounded."""
        return np.array(
            [getattr(p, "lb", -np.inf) for p in self._parameters],
            dtype=float,
        )

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds for all parameters (independent + dependent); ``+inf`` when unbounded."""
        return np.array(
            [getattr(p, "ub", np.inf) for p in self._parameters],
            dtype=float,
        )

    @property
    def lower_bounds_independent(self) -> np.ndarray:
        """Lower bounds for independent variables only; ``-inf`` when unbounded."""
        return np.array(
            [getattr(p, "lb", -np.inf) for p in self.independent_parameters],
            dtype=float,
        )

    @property
    def upper_bounds_independent(self) -> np.ndarray:
        """Upper bounds for independent variables only; ``+inf`` when unbounded."""
        return np.array(
            [getattr(p, "ub", np.inf) for p in self.independent_parameters],
            dtype=float,
        )

    @resolves_dependencies
    def check_bounds(
        self,
        x: npt.ArrayLike,
        tol: float | npt.ArrayLike = 0.0,
    ) -> bool:
        """Return True when all parameters satisfy their bounds.

        Parameters
        ----------
        x : array-like
            Full parameter vector (length ``n_parameters``) in physical units.
            Pass ``resolve_dependencies=True`` to expand an independent-only vector
            (length ``n_variables``) first.
        tol : float or array-like
            Per-variable (or uniform) tolerance added to each bound.

        Raises
        ------
        ValueError
            If the length of *x* does not match ``n_parameters``.
        """
        vals = np.asarray(x, dtype=float).ravel()
        n = self.n_parameters
        if vals.size != n:
            raise ValueError(f"Expected {n} values, got {vals.size}.")
        tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), n)
        lbs = self.lower_bounds
        ubs = self.upper_bounds
        return bool(np.all(vals >= lbs - tol_arr) and np.all(vals <= ubs + tol_arr))

    @resolves_dependencies
    def evaluate_bounds(self, x: npt.ArrayLike) -> np.ndarray:
        """Return ``[lb - x, x - ub]`` over all parameters; positive = bound violation.

        Parameters
        ----------
        x : array-like
            Full parameter vector (length ``n_parameters``) in physical units.
            Pass ``resolve_dependencies=True`` to expand an independent-only vector first.
        """
        x = np.asarray(x, dtype=float).ravel()
        return np.concatenate([self.lower_bounds - x, x - self.upper_bounds])

    @resolves_dependencies
    def validate_x(
        self,
        x: npt.ArrayLike,
        tol: float = 0.0,
        tol_eq: float = 1e-6,
    ) -> bool | np.ndarray:
        """Return True if *x* satisfies bounds and all linear constraints.

        Parameters
        ----------
        x : array-like
            Full parameter vector, shape ``(n_parameters,)`` for a single point or
            ``(m, n_parameters)`` for a population.  Pass
            ``resolve_dependencies=True`` to expand an independent-only 1-D vector
            first (population input always requires the full vector).
        tol : float
            Tolerance applied to bounds and inequality constraints (inclusive).
        tol_eq : float
            Tolerance applied to equality constraints.

        Returns
        -------
        bool or np.ndarray of bool
            Scalar for a single point; 1-D boolean array for a population.
        """
        x_arr = np.asarray(x, dtype=float)
        population = x_arr.ndim == 2
        rows = x_arr if population else x_arr[np.newaxis, :]
        A, b = self.A, self.b
        Aeq, beq = self.A_eq, self.b_eq
        results = []
        for row in rows:
            valid = True
            if not self.check_bounds(row, tol=tol):
                warnings.warn("x violates parameter bounds.")
                valid = False
            if A.shape[0] > 0 and not bool(np.all(A @ row <= b + tol)):
                warnings.warn("x violates linear inequality constraints.")
                valid = False
            if Aeq.shape[0] > 0 and not bool(np.all(np.abs(Aeq @ row - beq) <= tol_eq)):
                warnings.warn("x violates linear equality constraints.")
                valid = False
            results.append(valid)
        if population:
            return np.array(results)
        return results[0]

    # ── Normalization ─────────────────────────────────────────────────────────

    def normalize(self, x: npt.ArrayLike) -> np.ndarray:
        """Map independent values to [0, 1] using per-parameter normalizers.

        ``ChoiceParameter``, integer parameters, and parameters with infinite
        bounds are returned unchanged.  Bounds are not enforced here; this is a coordinate transform
        utility, not a validation gate.  Constraints are *not* automatically
        transformed; a linear constraint involving a parameter with a non-linear
        normalizer becomes non-linear in the normalized space.  Full constraint
        transformation is deferred to ``TransformedSpace`` (PARAMETERS.md step 3).
        """
        vals = np.asarray(x, dtype=float).ravel()
        if vals.size != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} values, got {vals.size}."
            )
        result = vals.copy()
        for i, p in enumerate(self.independent_parameters):
            if (
                isinstance(p, RangedParameter)
                and p.parameter_type is float
                and np.isfinite(p.lb)
                and np.isfinite(p.ub)
            ):
                result[i] = p.normalizer._normalize(float(vals[i]))
        return result

    def denormalize(self, x_norm: npt.ArrayLike) -> np.ndarray:
        """Map normalized values back to the original parameter space.

        Bounds are not enforced here; out-of-range normalized values are mapped
        to their corresponding physical values without raising.  This allows
        ``check_bounds(x, denormalize=True)`` to correctly return False for
        values that are outside the normalized unit hypercube.
        """
        vals = np.asarray(x_norm, dtype=float).ravel()
        if vals.size != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} values, got {vals.size}."
            )
        result = vals.copy()
        for i, p in enumerate(self.independent_parameters):
            if (
                isinstance(p, RangedParameter)
                and p.parameter_type is float
                and np.isfinite(p.lb)
                and np.isfinite(p.ub)
            ):
                result[i] = p.normalizer._denormalize(float(vals[i]))
        return result

    # ── Writing ───────────────────────────────────────────────────────────────

    def set_values(
        self,
        assignment: Mapping[str, Any],
        *,
        validate_bounds: bool = False,
        tol: float | npt.ArrayLike = 0.0,
    ) -> None:
        """Resolve dependent parameters, validate, and write into evaluation objects.

        Validates *all* parameters (including dependent) via their ``validate``
        method, so out-of-bounds dependent values raise here even though they are
        not caught by ``check_bounds``.

        Parameters
        ----------
        assignment : Mapping
            Values for the independent parameters by name, in physical units.
            Order-insensitive.  Numeric vectors are an encoding owned by
            ``TransformedSpace``; decode first:
            ``space.set_values(space.transformed_space.decode(x))``.
        validate_bounds : bool
            When True, check that the resolved values satisfy parameter bounds
            before writing.
        tol : float or array-like
            Per-parameter (or uniform) tolerance added to each bound when
            *validate_bounds* is True.

        Raises
        ------
        TypeError
            If *assignment* is not a Mapping, or a parameter's ``validate``
            rejects its resolved value.
        ValueError
            If *assignment* violates the input contract (unknown names, missing
            independent parameters, supplied dependent values), if
            *validate_bounds* is True and a bound is violated, or if a
            parameter's ``validate`` rejects its resolved value.
        """
        if not isinstance(assignment, Mapping):
            raise TypeError(
                "set_values takes a named assignment (Mapping of parameter name "
                "to value); numeric vectors are an encoding owned by "
                "TransformedSpace — decode first: "
                "space.set_values(space.transformed_space.decode(x))."
            )
        all_values = self.resolve(assignment)
        if validate_bounds:
            tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), self.n_parameters)
            for i, p in enumerate(self._parameters):
                if not isinstance(p, RangedParameter):
                    continue
                value = float(all_values[p.name])
                if not (p.lb - tol_arr[i] <= value <= p.ub + tol_arr[i]):
                    raise ValueError("Values violate bound constraints.")

        for p in self._parameters:
            value = all_values[p.name]
            if isinstance(p, RangedParameter) and p.significant_digits is not None:
                value = float(round_to_significant_digits(value, p.significant_digits))
            value = p.validate(value)
            if p.name in self._mappers:
                self._mappers[p.name].set_value(value)

    def get_value(self, name: str) -> Any:
        """Read the current value of parameter *name* from its evaluation object.

        Returns ``None`` when no mapper is wired or the mapper does not support
        read-back.

        Raises
        ------
        KeyError
            If no parameter named *name* is registered.
        """
        self._get_parameter(name)  # raises KeyError if not found
        mapper = self._mappers.get(name)
        if mapper is None:
            return None
        return mapper.get_value()

    def set_value(self, name: str, value: Any) -> None:
        """Write *value* directly for a single parameter by name.

        Validates the value but does not resolve dependencies.  Use
        ``set_values`` to write an entire independent-variable vector with
        full dependency resolution.

        Raises
        ------
        KeyError
            If no parameter named *name* is registered.
        TypeError, ValueError
            If *value* fails the parameter's ``validate`` check.
        """
        p = self._get_parameter(name)
        value = p.validate(value)
        mapper = self._mappers.get(name)
        if mapper is not None:
            mapper.set_value(value)

    def _get_parameter(self, name: str) -> ParameterBase:
        """Return the parameter object for *name*, raising KeyError if absent."""
        for p in self._parameters:
            if p.name == name:
                return p
        raise KeyError(f"No parameter named {name!r}.")

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        pool_size: int = 100_000,
        include_dependent: bool = False,
    ) -> list[dict[str, Any]]:
        """Draw *n* feasible samples as named assignments.

        Delegates to :class:`HopsySampler` with the given *pool_size*.
        See :class:`~CADETProcess.parameter_space.sampling.SamplerBase` for
        the full postprocessing contract (significant-digits snap, integer
        rounding, categorical merge, dependency resolution, validation).

        Parameters
        ----------
        n : int
            Number of feasible samples to return.
        seed : int, optional
            Random seed. A random 32-bit seed is drawn when not specified.
        pool_size : int
            MCMC steps used to build the candidate pool.
        include_dependent : bool
            When True each assignment includes resolved dependent parameters.

        Returns
        -------
        list[dict]
            *n* named assignments, each ordered by registration.

        Raises
        ------
        ValueError
            If *n* feasible samples cannot be found within the *pool_size* budget.
        """
        from CADETProcess.parameter_space.sampling import HopsySampler

        return HopsySampler(pool_size=pool_size).sample(
            self, n, seed=seed, include_dependent=include_dependent
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"ParameterSpace("
            f"n_variables={self.n_variables}, "
            f"n_parameters={self.n_parameters}, "
            f"evaluation_objects={len(self._evaluation_objects)})"
        )
