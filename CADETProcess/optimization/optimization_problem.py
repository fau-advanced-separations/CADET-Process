"""OptimizationProblem: optimization policy on top of ParameterSpace.

Covers the parameter side (evaluation objects, variables, bounds, linear
constraints, dependencies, transforms, initial value sampling) and the
evaluation side (objectives, nonlinear constraints, callbacks, evaluators).
"""

from __future__ import annotations

import inspect
import logging
import math
import random
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional

import hopsy
import numpy as np
import numpy.typing as npt
from packaging.version import Version

from CADETProcess import CADETProcessError, log
from CADETProcess.dataStructure.deprecation import deprecated_alias
from CADETProcess.dataStructure.nested_dict import attribute_path_exists
from CADETProcess.evaluation_pipeline import EvaluationFailure, EvaluationPipeline
from CADETProcess.optimization.individual import Individual
from CADETProcess.optimization.population import Population
from CADETProcess.parameter_space import ParameterSpace
from CADETProcess.parameter_space.constraints import (
    LinearConstraint,
    LinearEqualityConstraint,
)
from CADETProcess.parameter_space.mappers import (
    IndexedMapper,
    make_preprocessing_mapper,
)
from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    ParameterBase,
    RangedParameter,
)
from CADETProcess.parameter_space.transformed_space import TransformedSpace

__all__ = ["OptimizationProblem"]


# ── Metric annotation ─────────────────────────────────────────────────────


class _MetricRecord:
    """Annotation for an objective, nonlinear constraint, or callback.

    Holds metadata only; the callable and its upstream evaluators are managed
    by ``OptimizationProblem``.  Evaluation is handled by
    ``_evaluate_individual``.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        n_metrics: int = 1,
        bad_metrics: float | npt.ArrayLike | None = None,
        evaluation_objects: list | None = None,
        evaluator_chain: list[str] | None = None,
        labels: list[str] | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        minimize: bool = True,
        bounds: list | None = None,
        comparison_operator: str = "le",
        frequency: int = 1,
        callbacks_dir: Any = None,
        keep_progress: bool = False,
    ) -> None:
        self.func = func
        self.name = name
        self.n_metrics = n_metrics
        if bad_metrics is None:
            self.bad_metrics = np.full(n_metrics, np.inf)
        elif np.isscalar(bad_metrics):
            self.bad_metrics = np.full(n_metrics, float(bad_metrics))
        else:
            self.bad_metrics = np.asarray(bad_metrics, dtype=float)
        self.evaluation_objects: list = list(evaluation_objects) if evaluation_objects else []
        self.evaluator_chain: list[str] = list(evaluator_chain) if evaluator_chain else []
        self._labels: list[str] | None = list(labels) if labels is not None else None
        self.args = args if args else ()
        self.kwargs = kwargs if kwargs is not None else {}
        self.minimize = minimize
        self.bounds: list = list(bounds) if bounds is not None else []
        self.comparison_operator = comparison_operator
        self.frequency = frequency
        self.callbacks_dir = callbacks_dir
        self.keep_progress = keep_progress

    @property
    def n_total_metrics(self) -> int:
        """Total metric count across all evaluation objects."""
        n_eval = len(self.evaluation_objects) if self.evaluation_objects else 1
        return n_eval * self.n_metrics

    @property
    def labels(self) -> list[str]:
        """Metric labels, expanded across evaluation objects when there are multiple."""
        if self._labels is not None:
            base = list(self._labels)
        else:
            try:
                base = list(self.func.labels)
            except AttributeError:
                if self.n_metrics > 1:
                    base = [f"{self.name}_{i}" for i in range(self.n_metrics)]
                else:
                    base = [self.name]
        if len(self.evaluation_objects) > 1:
            return [
                f"{eval_obj}_{label}"
                for label in base
                for eval_obj in self.evaluation_objects
            ]
        return base

    @labels.setter
    def labels(self, value: list[str] | None) -> None:
        if value is not None and len(value) != self.n_metrics:
            raise CADETProcessError(f"Expected {self.n_metrics} labels.")
        self._labels = list(value) if value is not None else None

    def cleanup(self, callbacks_dir: Any, current_iteration: int) -> None:
        """Remove stale callback files, optionally archiving progress snapshots."""
        if (
            not current_iteration % self.frequency == 0
            or current_iteration <= self.frequency
        ):
            return

        previous_iteration = current_iteration - self.frequency

        if self.callbacks_dir is not None:
            callbacks_dir = self.callbacks_dir

        if self.keep_progress:
            new_directory = callbacks_dir / "progress" / str(previous_iteration)
            new_directory.mkdir(exist_ok=True, parents=True)

        for file in callbacks_dir.iterdir():
            if not file.is_file():
                continue
            if self.keep_progress:
                shutil.copy(file, new_directory)
            file.unlink()

    def __str__(self) -> str:
        return self.name


# ── Helpers ────────────────────────────────────────────────────────────────


def _approximate_jac(
    xk: npt.ArrayLike,
    f: Callable,
    epsilon: float | npt.ArrayLike = 1e-3,
    **kwargs: Any,
) -> np.ndarray:
    """Forward finite-difference Jacobian of a vector-valued function.

    Parameters
    ----------
    xk : array-like
        Point at which to evaluate the Jacobian.
    f : callable
        Function ``f(x, **kwargs)`` returning a 1-D array.
    epsilon : float or array-like
        Step size.  A scalar uses the same step for all dimensions.
    **kwargs
        Passed through to ``f``.

    Returns
    -------
    np.ndarray, shape (n_outputs, n_inputs)
    """
    xk = np.asarray(xk, dtype=float).ravel()
    f0 = np.atleast_1d(np.array(f(xk, **kwargs), dtype=float))
    jac = np.zeros((len(f0), len(xk)), dtype=float)
    ei = np.zeros(len(xk), dtype=float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        try:
            fk = np.atleast_1d(np.array(f(xk + d, **kwargs), dtype=float))
            if not np.all(np.isfinite(fk)):
                raise ValueError("non-finite forward evaluation")
            jac[:, k] = (fk - f0) / d[k]
        except ValueError:
            try:
                fk = np.atleast_1d(np.array(f(xk - d, **kwargs), dtype=float))
                if not np.all(np.isfinite(fk)):
                    raise ValueError("non-finite backward evaluation")
                jac[:, k] = (f0 - fk) / d[k]
            except ValueError:
                jac[:, k] = 0.0
        ei[k] = 0.0
    return jac


# ── OptimizationProblem ────────────────────────────────────────────────────


class OptimizationProblem:
    """Optimization policy over ParameterSpace and EvaluationPipeline.

    Decides what to optimize: which callables are objectives, which are
    constraints, which are callbacks, and how failures are handled.
    Delegates parameter semantics (bounds, linear constraints,
    normalization, dependency resolution) to ``ParameterSpace`` and
    execution to ``EvaluationPipeline``.

    Parameters
    ----------
    name : str
        Problem name; used as a string representation.
    use_diskcache : bool
        When True (default) and *cache_directory* is set, evaluation results are
        cached to disk via pipefunc's disk backend and survive process restarts.
        When False, an in-memory LRU cache is used regardless of *cache_directory*.
    cache_directory : str, optional
        Directory for the disk cache.  Has no effect when ``use_diskcache=False``.
        When not set, an in-memory LRU cache is used.
    log_level : str
        Logging level for this problem's logger (e.g. ``"DEBUG"``, ``"INFO"``).
    """

    def __init__(
        self,
        name: str,
        use_diskcache: bool = True,
        cache_directory: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        self.name = name
        self.logger = log.get_logger(name, level=log_level)

        # Disk cache when requested: use_diskcache=True OR an explicit directory.
        effective_cache_dir = cache_directory if use_diskcache else None

        self._space = ParameterSpace()
        self._pipeline = EvaluationPipeline(self._space, cache_dir=effective_cache_dir)
        self._params: dict[str, ParameterBase] = {}
        self._path_registry: dict[tuple, str] = {}  # (path, obj_id, index_repr) → var_name

        # Evaluator registry: callable → name, name → wrapped callable, ordered list
        self._evaluator_names: dict[Callable, str] = {}        # func → output_name
        self._evaluator_func_by_name: dict[str, Callable] = {}  # output_name → wrapped callable
        self._evaluator_registry: list[tuple[str, Callable]] = []  # ordered (name, func)

        self._objectives: list[_MetricRecord] = []
        self._nonlinear_constraints: list[_MetricRecord] = []
        self._callbacks: list[_MetricRecord] = []
        self._meta_scores: list[_MetricRecord] = []
        self._multi_criteria_decision_functions: list = []

    # ── Evaluation objects ─────────────────────────────────────────────────────

    def add_evaluation_object(self, obj: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Register an evaluation object."""
        self._space.add_evaluation_object(obj)

    @property
    def evaluation_objects(self) -> list[Any]:
        """Registered evaluation objects, in insertion order."""
        return self._space.evaluation_objects

    @property
    def evaluation_objects_dict(self) -> dict[str, Any]:
        """Mapping of ``str(obj)`` → obj for all registered evaluation objects."""
        return {str(obj): obj for obj in self._space.evaluation_objects}

    # ── Variables ─────────────────────────────────────────────────────────────

    @deprecated_alias(transform="normalization")
    def add_variable(
        self,
        name: str,
        evaluation_objects: Any = -1,
        parameter_path: Optional[str] = None,
        lb: float = -math.inf,
        ub: float = math.inf,
        parameter_type: type[int] | type[float] = float,
        normalization: Optional[str] = None,
        indices: Optional[Any] = None,
        significant_digits: Optional[int] = None,
        pre_processing: Optional[Callable] = None,
    ) -> RangedParameter:
        """Add an optimization variable.

        Parameters
        ----------
        name : str
            Variable name.
        evaluation_objects : list, object, or -1
            Evaluation objects this variable targets.  ``-1`` (default) targets
            all registered objects; ``None`` creates a free variable with no
            write target.
        parameter_path : str, optional
            Dot-separated path to the attribute on the evaluation object.
            Defaults to *name* when evaluation objects are present.
        lb, ub : float
            Lower and upper bounds.
        parameter_type : {int, float}
            Scalar domain.  ``int`` restricts the variable to integral values.
        normalization : {'auto', 'log', 'linear', None}
            Normalization scheme applied to this variable.
        indices : int, tuple, or numpy index expression, optional
            Target specific array entries; an ``IndexedMapper`` is used.
        significant_digits : int, optional
            Round the value to this many significant digits before writing.
        pre_processing : callable, optional
            Applied to the value just before the final write.
            Receives the scalar value and must return the value to set on the
            attribute (may change type or shape, e.g. ``lambda w: [w, 1-w]``).
            Uses a ``CallableMapper`` internally; incompatible with *indices*.
        """
        if name in self._params:
            raise CADETProcessError("Variable already exists")

        param = RangedParameter(
            name,
            parameter_type,
            lb=lb,
            ub=ub,
            normalization=normalization,
            significant_digits=significant_digits,
        )

        # Resolve evaluation objects to a concrete list.
        if evaluation_objects is None:
            eval_objs: list[Any] = []
        elif evaluation_objects == -1:
            eval_objs = list(self._space.evaluation_objects)
        elif not isinstance(evaluation_objects, list):
            eval_objs = [evaluation_objects]
        else:
            eval_objs = list(evaluation_objects)

        # Resolve string references.
        objs_dict = self.evaluation_objects_dict
        eval_objs = [objs_dict[o] if isinstance(o, str) else o for o in eval_objs]

        # Default path to variable name when eval objects are present.
        if parameter_path is None and eval_objs:
            parameter_path = name
        if parameter_path is not None and not eval_objs:
            raise ValueError(
                "Cannot set parameter_path for a variable without evaluation objects."
            )

        # Validate path exists on each evaluation object.
        if eval_objs and parameter_path:
            for obj in eval_objs:
                if not attribute_path_exists(obj, parameter_path):
                    raise CADETProcessError(
                        f"'{parameter_path}' is not a valid parameter on {obj!r}"
                    )

        # Detect duplicate path / path+index registrations.
        if eval_objs and parameter_path:
            index_key = repr(indices) if indices is not None else None
            for obj in eval_objs:
                key = (parameter_path, id(obj), index_key)
                if key in self._path_registry:
                    existing = self._path_registry[key]
                    raise CADETProcessError(
                        f"Path '{parameter_path}' (index={indices!r}) is already "
                        f"registered as variable '{existing}'"
                    )
            for obj in eval_objs:
                key = (parameter_path, id(obj), index_key)
                self._path_registry[key] = name

        # Wire mapper.
        if not eval_objs:
            self._space.add_parameter(param)
        elif pre_processing is not None:
            mapper = make_preprocessing_mapper(eval_objs, parameter_path, pre_processing)
            self._space.add_parameter(param, mapper=mapper)
        elif indices is not None:
            self._space.add_parameter(
                param, mapper=IndexedMapper(eval_objs, parameter_path, indices)
            )
        else:
            self._space.add_parameter(
                param, path=parameter_path, evaluation_objects=eval_objs
            )

        self._params[name] = param
        return param

    def add_choice_variable(
        self,
        name: str,
        valid_values: list[Any],
        evaluation_objects: Any = -1,
        parameter_path: Optional[str] = None,
    ) -> ChoiceParameter:
        """Add a categorical variable with a finite set of allowed values.

        Parameters
        ----------
        name : str
            Variable name.
        valid_values : list
            Allowed choices.
        evaluation_objects : list, object, or -1
            Evaluation objects this variable targets.  ``-1`` (default) targets
            all registered objects; ``None`` creates a free variable with no
            write target.
        parameter_path : str, optional
            Dot-separated path to the attribute on the evaluation object.
            Defaults to *name* when evaluation objects are present.
        """
        if name in self._params:
            raise CADETProcessError("Variable already exists")

        param = ChoiceParameter(name, valid_values)

        if evaluation_objects is None:
            eval_objs: list[Any] = []
        elif evaluation_objects == -1:
            eval_objs = list(self._space.evaluation_objects)
        elif not isinstance(evaluation_objects, list):
            eval_objs = [evaluation_objects]
        else:
            eval_objs = list(evaluation_objects)

        objs_dict = self.evaluation_objects_dict
        eval_objs = [objs_dict[o] if isinstance(o, str) else o for o in eval_objs]

        if parameter_path is None and eval_objs:
            parameter_path = name
        if parameter_path is not None and not eval_objs:
            raise ValueError(
                "Cannot set parameter_path for a variable without evaluation objects."
            )

        if eval_objs and parameter_path:
            for obj in eval_objs:
                if not attribute_path_exists(obj, parameter_path):
                    raise CADETProcessError(
                        f"'{parameter_path}' is not a valid parameter on {obj!r}"
                    )

        if eval_objs and parameter_path:
            index_key = None
            for obj in eval_objs:
                key = (parameter_path, id(obj), index_key)
                if key in self._path_registry:
                    existing = self._path_registry[key]
                    raise CADETProcessError(
                        f"Path '{parameter_path}' is already "
                        f"registered as variable '{existing}'"
                    )
            for obj in eval_objs:
                key = (parameter_path, id(obj), index_key)
                self._path_registry[key] = name

        if not eval_objs:
            self._space.add_parameter(param)
        else:
            self._space.add_parameter(
                param, path=parameter_path, evaluation_objects=eval_objs
            )

        self._params[name] = param
        return param

    def check_duplicate_variables(self) -> bool:
        """Return True. Duplicates are rejected eagerly at add_variable time."""
        return True

    def remove_variable(self, var_name: str) -> None:
        """Remove a variable. Not yet implemented in the OptimizationProblem."""
        raise NotImplementedError(
            "Variable removal is not yet supported in the OptimizationProblem."
        )

    @property
    def variables(self) -> list[ParameterBase]:
        """All registered parameters (independent + derived), in registration order."""
        return self._space.parameters

    @property
    def variable_names(self) -> list[str]:
        """Names of all parameters in registration order."""
        return [p.name for p in self._space.parameters]

    @property
    def variables_dict(self) -> dict:
        """All optimization variables indexed by name."""
        return self._params

    @property
    def n_variables(self) -> int:
        """Total number of parameters (independent + derived)."""
        return len(self._space.parameters)

    @property
    def independent_variables(self) -> list[ParameterBase]:
        """Parameters that are not derived from other parameters."""
        return self._space.independent_parameters

    @property
    def independent_variable_names(self) -> list[str]:
        """Names of independent parameters."""
        return [p.name for p in self._space.independent_parameters]

    @property
    def n_independent_variables(self) -> int:
        """Number of independent (optimizer-facing) parameters."""
        return self._space.n_variables

    @property
    def dependent_variables(self) -> list[ParameterBase]:
        """Parameters computed from other parameters."""
        return self._space.dependent_parameters

    @property
    def dependent_variable_names(self) -> list[str]:
        """Names of derived parameters."""
        return [p.name for p in self._space.dependent_parameters]

    @property
    def n_dependent_variables(self) -> int:
        """Number of derived parameters."""
        return len(self._space.dependent_parameters)

    @property
    def continuous_variables(self) -> list[RangedParameter]:
        """Independent continuous (float) variables."""
        return self._space.continuous_parameters

    @property
    def n_continuous_variables(self) -> int:
        """Number of independent continuous variables."""
        return len(self._space.continuous_parameters)

    @property
    def integer_variables(self) -> list[RangedParameter]:
        """Independent integer variables."""
        return self._space.integer_parameters

    @property
    def n_integer_variables(self) -> int:
        """Number of independent integer variables."""
        return len(self._space.integer_parameters)

    @property
    def categorical_variables(self) -> list[ChoiceParameter]:
        """Independent categorical variables."""
        return self._space.categorical_parameters

    @property
    def n_categorical_variables(self) -> int:
        """Number of independent categorical variables."""
        return len(self._space.categorical_parameters)

    # ── Dependencies ──────────────────────────────────────────────────────────

    def add_variable_dependency(
        self,
        dependent_variable: str | RangedParameter,
        independent_variables: str | list,
        transform: Callable,
    ) -> None:
        """Declare that *dependent_variable* is computed from *independent_variables*.

        Parameters
        ----------
        dependent_variable : str or RangedParameter
            The variable whose value will be derived.
        independent_variables : str or list
            One or more variable names (or objects) that feed the transform.
        transform : callable
            Called as ``transform(*values)``; must return the derived value.
        """
        if not callable(transform):
            raise CADETProcessError("transform must be callable")

        derived_name = (
            dependent_variable if isinstance(dependent_variable, str)
            else dependent_variable.name
        )
        derived = self._params.get(derived_name)
        if derived is None:
            raise CADETProcessError(f"Variable '{derived_name}' does not exist")

        if isinstance(independent_variables, str):
            independent_variables = [independent_variables]
        elif not isinstance(independent_variables, list):
            independent_variables = [independent_variables]

        ind_params: list[RangedParameter] = []
        for v in independent_variables:
            vname = v if isinstance(v, str) else v.name
            p = self._params.get(vname)
            if p is None:
                raise CADETProcessError(f"Variable '{vname}' does not exist")
            ind_params.append(p)

        try:
            self._space.add_dependency(derived, ind_params, transform)
        except ValueError as exc:
            raise CADETProcessError(str(exc)) from exc

    def get_dependent_values(
        self,
        x_independent: npt.ArrayLike,
        untransform: bool = False,
    ) -> np.ndarray:
        """Return a full parameter vector from independent values.

        Returns values for all parameters (independent + derived) in
        registration order.

        Parameters
        ----------
        x_independent : array-like
            Independent parameter values.  May be 1-D (single individual) or
            2-D (population, shape n_individuals × n_independent_vars).
        untransform : bool
            When True, denormalize *x_independent* from normalized coordinates
            before resolving dependencies.
        """
        X = np.array(x_independent)
        was_1d = X.ndim == 1
        X_2d = np.atleast_2d(X).astype(float)

        if untransform:
            X_2d = np.array([self.untransform(row) for row in X_2d])

        results = np.array([self._resolve_full_vector(row) for row in X_2d])

        if was_1d:
            return results[0]
        return results

    def _resolve_full_vector(self, x: npt.ArrayLike) -> np.ndarray:
        """Resolve a single independent vector to a full parameter vector."""
        x = np.asarray(x, dtype=float).ravel()
        all_vals = self._space._resolve_all_values(x)
        return np.array([all_vals[p.name] for p in self._space.parameters])

    def get_independent_values(self, x_all: npt.ArrayLike) -> np.ndarray:
        """Extract independent values from a full parameter vector."""
        x_all = np.asarray(x_all, dtype=float).ravel()
        ind_names = {p.name for p in self._space.independent_parameters}
        return np.array(
            [v for p, v in zip(self._space.parameters, x_all) if p.name in ind_names]
        )

    def set_variables(self, x: npt.ArrayLike) -> None:
        """Write *x* (independent values) into evaluation objects."""
        self._space.set_values(self._space.transformed_space.decode(x))

    def get_variable_value(self, name: str) -> Any:
        """Read the current value of variable *name* from its evaluation object.

        Returns ``None`` when the variable has no path (no evaluation object
        wired) or the mapper does not support read-back.

        Raises
        ------
        KeyError
            If no variable named *name* is registered.
        """
        return self._space.get_value(name)

    # ── Bounds ────────────────────────────────────────────────────────────────

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds for all variables (independent + dependent); ``-inf`` when unbounded."""
        return self._space.lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds for all variables (independent + dependent); ``+inf`` when unbounded."""
        return self._space.upper_bounds

    @property
    def lower_bounds_independent(self) -> np.ndarray:
        """Lower bounds for independent variables only; ``-inf`` when unbounded."""
        return self._space.lower_bounds_independent

    @property
    def upper_bounds_independent(self) -> np.ndarray:
        """Upper bounds for independent variables only; ``+inf`` when unbounded."""
        return self._space.upper_bounds_independent

    def evaluate_bounds(self, x: npt.ArrayLike, get_dependent_values: bool = True) -> np.ndarray:
        """Return ``[lb - x, x - ub]``; positive entries mean a bound violation.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector (independent + dependent).
        get_dependent_values : bool
            When True (default), resolve the full vector from independent
            values first.  When False, the input is used as-is.
        """
        return self._space.evaluate_bounds(x, resolve_dependencies=get_dependent_values)

    def check_bounds(
        self, x: npt.ArrayLike, tol: float | npt.ArrayLike = 0.0
    ) -> bool:
        """Return True if all independent values satisfy their bounds."""
        return self._space.check_bounds(x, tol=tol, resolve_dependencies=True)

    # ── Linear constraints ─────────────────────────────────────────────────────

    def _resolve_constraint_params(
        self, opt_vars: str | list[str]
    ) -> list[RangedParameter]:
        if isinstance(opt_vars, str):
            opt_vars = [opt_vars]
        params = []
        for name in opt_vars:
            p = self._params.get(name)
            if p is None:
                raise CADETProcessError(f"Variable '{name}' does not exist")
            params.append(p)
        return params

    def add_linear_constraint(
        self,
        opt_vars: str | list[str],
        lhs: float | list[float] = 1,
        b: float = 0,
    ) -> None:
        """Add a linear inequality constraint: ``lhs · x <= b``.

        Parameters
        ----------
        opt_vars : str or list[str]
            Variable names involved in the constraint.
        lhs : float or list[float]
            Coefficients; a scalar is broadcast to all variables.
        b : float
            Right-hand side.
        """
        params = self._resolve_constraint_params(opt_vars)
        try:
            constraint = LinearConstraint(params, lhs, b)
        except ValueError as exc:
            raise CADETProcessError(str(exc)) from exc
        self._space.add_linear_constraint(constraint)

    def remove_linear_constraint(self, index: int) -> None:
        """Remove the linear inequality constraint at *index*."""
        self._space._linear_constraints.pop(index)

    def add_linear_equality_constraint(
        self,
        opt_vars: str | list[str],
        lhs: float | list[float] = 1,
        b: float = 0,
    ) -> None:
        """Add a linear equality constraint: ``lhs · x = b``."""
        params = self._resolve_constraint_params(opt_vars)
        try:
            constraint = LinearEqualityConstraint(params, lhs, b)
        except ValueError as exc:
            raise CADETProcessError(str(exc)) from exc
        self._space.add_linear_equality_constraint(constraint)

    def remove_linear_equality_constraint(self, index: int) -> None:
        """Remove the linear equality constraint at *index*."""
        self._space._linear_equality_constraints.pop(index)

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """Registered linear inequality constraints."""
        return self._space.linear_constraints

    @property
    def n_linear_constraints(self) -> int:
        """Number of registered linear inequality constraints."""
        return len(self._space.linear_constraints)

    @property
    def linear_equality_constraints(self) -> list[LinearEqualityConstraint]:
        """Registered linear equality constraints."""
        return self._space.linear_equality_constraints

    @property
    def n_linear_equality_constraints(self) -> int:
        """Number of registered linear equality constraints."""
        return len(self._space.linear_equality_constraints)

    @property
    def A(self) -> np.ndarray:
        """Inequality constraint matrix over all parameters, shape (m, n_parameters)."""
        return self._space.A

    @property
    def b(self) -> np.ndarray:
        """Inequality constraint RHS, shape (m,)."""
        return self._space.b

    @property
    def Aeq(self) -> np.ndarray:
        """Equality constraint matrix over all parameters, shape (m, n_parameters)."""
        return self._space.A_eq

    @property
    def beq(self) -> np.ndarray:
        """Equality constraint RHS, shape (m,)."""
        return self._space.b_eq

    def evaluate_linear_constraints(
        self, x: npt.ArrayLike, get_dependent_values: bool = True
    ) -> np.ndarray:
        """Return ``A @ x - b`` for inequality constraints; positive = violation.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector (independent + dependent).
        get_dependent_values : bool
            When True (default), resolve the full vector from independent
            values first.  When False, the input is used as-is.
        """
        return self._space.evaluate_linear_constraints(
            x, resolve_dependencies=get_dependent_values
        )

    def check_linear_constraints(
        self,
        x: npt.ArrayLike,
        tol: float = 0.0,
        get_dependent_values: bool = True,
    ) -> bool:
        """Return True if *x* satisfies all inequality constraints.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector.
        tol : float
            Tolerance added to each bound.
        get_dependent_values : bool
            When True (default), resolve the full vector from independent
            values first.  When False, the input is used as-is.
        """
        if self._space.A.shape[0] == 0:
            return True
        return bool(
            np.all(
                self.evaluate_linear_constraints(x, get_dependent_values=get_dependent_values)
                <= tol
            )
        )

    def evaluate_linear_equality_constraints(
        self, x: npt.ArrayLike, get_dependent_values: bool = True
    ) -> np.ndarray:
        """Return ``Aeq @ x - b_eq`` for equality constraints; non-zero = violation.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector (independent + dependent).
        get_dependent_values : bool
            When True (default), resolve the full vector from independent
            values first.  When False, the input is used as-is.
        """
        return self._space.evaluate_linear_equality_constraints(
            x, resolve_dependencies=get_dependent_values
        )

    def check_linear_equality_constraints(
        self,
        x: npt.ArrayLike,
        tol: float = 1e-6,
        get_dependent_values: bool = True,
    ) -> bool:
        """Return True if *x* satisfies all equality constraints."""
        if self._space.A_eq.shape[0] == 0:
            return True
        return bool(
            np.all(
                np.abs(
                    self.evaluate_linear_equality_constraints(
                        x, get_dependent_values=get_dependent_values
                    )
                )
                <= tol
            )
        )

    # ── Parameter and transformed space ───────────────────────────────────────

    @property
    def parameter_space(self) -> ParameterSpace:
        """The underlying `ParameterSpace` owning parameters and evaluation objects."""
        return self._space

    @property
    def transformed_space(self) -> TransformedSpace:
        """Normalized optimizer view of the parameter space."""
        return self._space.transformed_space

    # ── Transform / normalization ─────────────────────────────────────────────

    def transform(self, x: npt.ArrayLike) -> np.ndarray:
        """Map independent values from physical to normalized coordinates."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return np.array([self._space.normalize(row) for row in x])
        return self._space.normalize(x.ravel())

    def untransform(self, x: npt.ArrayLike) -> np.ndarray:
        """Map independent values from normalized to physical coordinates."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return np.array([self._space.denormalize(row) for row in x])
        return self._space.denormalize(x.ravel())

    @property
    def A_transformed(self) -> np.ndarray:
        """Inequality constraint matrix in normalized coordinates."""
        return self.transformed_space.A

    @property
    def b_transformed(self) -> np.ndarray:
        """Inequality constraint RHS in normalized coordinates."""
        return self.transformed_space.b

    @property
    def Aeq_transformed(self) -> np.ndarray:
        """Equality constraint matrix in normalized coordinates."""
        return self.transformed_space.A_eq

    @property
    def beq_transformed(self) -> np.ndarray:
        """Equality constraint RHS in normalized coordinates."""
        return self.transformed_space.b_eq

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_chebyshev_center(
        self, include_dependent_variables: bool = True
    ) -> np.ndarray:
        """Compute the Chebyshev center of the independent-variable polytope."""
        problem = hopsy.Problem(self._space.A_independent, self._space.b)
        problem = hopsy.add_box_constraints(
            problem,
            self._space.lower_bounds_independent,
            self._space.upper_bounds_independent,
            simplify=False,
        )
        if self._space._linear_equality_constraints:
            problem = hopsy.add_equality_constraints(
                problem, self._space.A_eq_independent, self._space.b_eq
            )
        chebyshev = hopsy.compute_chebyshev_center(problem, original_space=True)
        if Version(hopsy.__version__.strip('"')) < Version("1.7.0b"):
            chebyshev = chebyshev[:, 0]
        if include_dependent_variables:
            chebyshev = self.get_dependent_values(chebyshev)
        return chebyshev

    def create_initial_values(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None,
        burn_in: int = 100_000,
        include_dependent_variables: bool = False,
    ) -> np.ndarray:
        """Draw feasible initial values from the independent-variable polytope.

        Uses hopsy for uniform polytope sampling.  Derived-variable constraints
        are enforced by post-hoc filtering.

        Returns
        -------
        np.ndarray, shape (n_samples, n_variables or n_independent_variables)
        """
        burn_in = int(burn_in)
        if seed is None:
            seed = random.randint(0, 255)

        log_indices = [
            i
            for i, p in enumerate(self._space.independent_parameters)
            if isinstance(p, RangedParameter) and not p.normalizer.is_linear
        ]

        class _LogSpaceModel:
            def __init__(self, li: list[int]) -> None:
                self.log_space_indices = li

            def compute_negative_log_likelihood(self, x: np.ndarray) -> float:
                return float(np.sum(np.log(x[self.log_space_indices])))

        model = _LogSpaceModel(log_indices) if log_indices else None
        problem = hopsy.Problem(self._space.A_independent, self._space.b, model)
        problem = hopsy.add_box_constraints(
            problem,
            self._space.lower_bounds_independent,
            self._space.upper_bounds_independent,
            simplify=False,
        )
        if self._space._linear_equality_constraints:
            problem = hopsy.add_equality_constraints(
                problem, self._space.A_eq_independent, self._space.b_eq
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = hopsy.round(problem, simplify=False)
            mc = hopsy.MarkovChain(
                problem, proposal=hopsy.UniformCoordinateHitAndRunProposal
            )
            rng_hopsy = hopsy.RandomNumberGenerator(seed=seed)
            _, states = hopsy.sample(mc, rng_hopsy, n_samples=burn_in, thinning=2)

        independent_values = states[0]  # shape (burn_in, n_ind_vars)
        rng = np.random.default_rng(seed)

        values = []
        counter = 0
        while len(values) < n_samples:
            if counter > burn_in:
                raise CADETProcessError(
                    "Cannot find individuals that fulfill constraints."
                )
            counter += 1
            idx = int(rng.integers(0, burn_in))

            ind: list[float] = []
            for i_var, p in enumerate(self._space.independent_parameters):
                v = float(independent_values[idx, i_var])
                if isinstance(p, RangedParameter) and p.significant_digits is not None:
                    from CADETProcess.numerics import round_to_significant_digits
                    v = float(round_to_significant_digits(v, p.significant_digits))
                ind.append(v)

            if not self.check_individual(ind, check_nonlinear_constraints=False):
                continue

            values.append(
                self.get_dependent_values(ind) if include_dependent_variables else ind
            )

        return np.array(values, ndmin=2)

    # ── Individual / config validation ────────────────────────────────────────

    def check_individual(
        self,
        x: npt.ArrayLike,
        check_nonlinear_constraints: bool = True,
        untransform: bool = False,
        get_dependent_values: bool = True,
        cv_bounds_tol: float = 0.0,
        cv_lincon_tol: float = 0.0,
        cv_lineqcon_tol: float = 0.0,
        cv_nonlincon_tol: float = 0.0,
    ) -> bool:
        """Return True if *x* is feasible.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector (independent + dependent).
        untransform : bool
            When True, denormalize *x* from normalized coordinates first.
        get_dependent_values : bool
            When True (default), *x* contains independent values only and the
            full vector is resolved internally.  When False, *x* is treated as
            the full parameter vector.
        cv_bounds_tol, cv_lincon_tol, cv_lineqcon_tol, cv_nonlincon_tol : float
            Per-check violation tolerances.
        """
        x = np.asarray(x, dtype=float).ravel()
        if untransform:
            x = self.untransform(x)
        if get_dependent_values:
            x = self._resolve_full_vector(x)

        ind = self.get_independent_values(x)
        if not self.check_bounds(ind, cv_bounds_tol):
            return False
        if not self.check_linear_constraints(x, cv_lincon_tol, get_dependent_values=False):
            return False
        if not self.check_linear_equality_constraints(
            x, cv_lineqcon_tol, get_dependent_values=False
        ):
            return False
        if check_nonlinear_constraints and self._nonlinear_constraints:
            try:
                if not self.check_nonlinear_constraints(
                    x, cv_nonlincon_tol, get_dependent_values=False
                ):
                    return False
            except Exception:
                return False
        return True

    def check_config(self, ignore_linear_constraints: bool = False) -> bool:  # noqa: ARG002
        """Return True if the problem is properly configured."""
        return True

    # ── Evaluators ────────────────────────────────────────────────────────────

    @property
    def evaluators(self) -> list[Callable]:
        """Registered evaluators, in insertion order."""
        return [func for _, func in self._evaluator_registry]

    @property
    def evaluators_dict(self) -> dict[str, Callable]:
        """Mapping of evaluator name to callable."""
        return {name: func for name, func in self._evaluator_registry}

    @property
    def evaluators_dict_reference(self) -> dict[Callable, str]:
        """Mapping of callable to evaluator name."""
        return dict(self._evaluator_names)

    def add_evaluator(
        self,
        evaluator: Callable,
        name: Optional[str] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> None:
        """Register a callable as a named evaluator for use in objective chains.

        Parameters
        ----------
        evaluator : callable
            The evaluator function.
        name : str, optional
            Name; defaults to ``evaluator.__name__`` for functions/methods.
        args : tuple, optional
            Fixed positional arguments appended after the request argument.
        kwargs : dict, optional
            Fixed keyword arguments passed to the evaluator.

        Raises
        ------
        TypeError
            If *evaluator* is not callable.
        CADETProcessError
            If an evaluator with the same name already exists.
        """
        if not callable(evaluator):
            raise TypeError("Expected callable evaluator.")

        if name is None:
            if inspect.isfunction(evaluator) or inspect.ismethod(evaluator):
                name = evaluator.__name__
            else:
                name = str(evaluator)

        if name in self.evaluators_dict:
            raise CADETProcessError("Evaluator with same name already exists.")

        # Build a wrapped callable that bakes in fixed args/kwargs.
        _args = args if args is not None else ()
        _kwargs = kwargs if kwargs is not None else {}
        if _args or _kwargs:
            def _wrapped(
                req: Any,
                _fn: Callable = evaluator,
                _a: tuple = _args,
                _kw: dict = _kwargs,
            ) -> Any:
                return _fn(req, *_a, **_kw)
        else:
            _wrapped = evaluator

        self._evaluator_registry.append((name, evaluator))
        self._evaluator_names[evaluator] = name
        self._evaluator_func_by_name[name] = _wrapped

    # ── Objectives ────────────────────────────────────────────────────────────

    @property
    def objectives(self) -> list[_MetricRecord]:
        """Registered objective records."""
        return self._objectives

    @property
    def objective_names(self) -> list[str]:
        """Names of all objectives, in registration order."""
        return [obj.name for obj in self._objectives]

    @property
    def objective_labels(self) -> list[str]:
        """Flat list of metric labels across all objectives."""
        labels = []
        for obj in self._objectives:
            labels += obj.labels
        return labels

    @property
    def n_objectives(self) -> int:
        """Total number of objective metrics across all objectives and eval objects."""
        return sum(obj.n_total_metrics for obj in self._objectives)

    def add_objective(
        self,
        objective: Callable,
        name: Optional[str] = None,
        n_objectives: int = 1,
        minimize: bool = True,
        bad_metrics: Optional[float | list[float]] = None,
        evaluation_objects: Any = -1,
        labels: Optional[list[str]] = None,
        requires: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register an objective function.

        Parameters
        ----------
        objective : callable
            Objective function.  Receives an evaluation object (or *x* when
            no evaluation objects are registered) and must return a scalar or
            1-D array of length *n_objectives*.
        name : str, optional
            Name; defaults to ``objective.__name__``.
        n_objectives : int
            Number of metrics returned by the function.
        minimize : bool
            When False, the metric is a maximization objective.
        bad_metrics : float or list of floats, optional
            Fallback values returned on evaluation failure.
        evaluation_objects : {-1, None, object, list}
            Which evaluation objects to use.  ``-1`` (default) uses all
            registered objects; ``None`` passes *x* directly.
        labels : list[str], optional
            Metric labels; length must equal *n_objectives*.
        requires : callable or list of callables, optional
            Upstream evaluators whose output feeds this function.

        Raises
        ------
        TypeError
            If *objective* is not callable.
        CADETProcessError
            If a referenced evaluation object or evaluator is not registered.
        """
        if not callable(objective):
            raise TypeError("Expected callable objective.")

        if name is None:
            if inspect.isfunction(objective) or inspect.ismethod(objective):
                name = objective.__name__
            else:
                name = str(objective)

        if name in self.objective_names:
            warnings.warn("Objective with same name already exists.")

        # Resolve evaluation objects.
        if evaluation_objects is None:
            eval_objs: list[Any] = []
        elif evaluation_objects == -1:
            eval_objs = list(self.evaluation_objects)
        elif not isinstance(evaluation_objects, list):
            eval_objs = [evaluation_objects]
        else:
            eval_objs = list(evaluation_objects)
        for el in eval_objs:
            if el not in self.evaluation_objects:
                raise CADETProcessError(f"Unknown EvaluationObject: {el!r}")

        # Resolve evaluator chain and lazily register in pipeline.
        if requires is None:
            req_list: list = []
        elif not isinstance(requires, list):
            req_list = [requires]
        else:
            req_list = list(requires)
        evaluator_chain: list[str] = []
        for req in req_list:
            if req not in self._evaluator_names:
                raise CADETProcessError(f"Unknown Evaluator: {req!r}")
            evaluator_chain.append(self._evaluator_names[req])
        self._register_evaluator_chain(req_list)

        record = _MetricRecord(
            objective,
            name,
            n_metrics=n_objectives,
            bad_metrics=bad_metrics,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            labels=labels,
            args=args,
            kwargs=kwargs if kwargs else None,
            minimize=minimize,
        )
        self._objectives.append(record)

    # ── Nonlinear constraints ─────────────────────────────────────────────────

    @property
    def nonlinear_constraints(self) -> list[_MetricRecord]:
        """Registered nonlinear constraint records."""
        return self._nonlinear_constraints

    @property
    def nonlinear_constraint_names(self) -> list[str]:
        """Names of all nonlinear constraints."""
        return [nc.name for nc in self._nonlinear_constraints]

    @property
    def nonlinear_constraint_labels(self) -> list[str]:
        """Flat list of labels across all nonlinear constraints."""
        labels = []
        for nc in self._nonlinear_constraints:
            labels += nc.labels
        return labels

    @property
    def nonlinear_constraints_bounds(self) -> list[float]:
        """Flat list of per-metric bounds across all nonlinear constraints."""
        bounds: list[float] = []
        for nc in self._nonlinear_constraints:
            bounds += nc.bounds
        return bounds

    @property
    def n_nonlinear_constraints(self) -> int:
        """Total number of nonlinear constraint metrics."""
        return sum(nc.n_total_metrics for nc in self._nonlinear_constraints)

    def add_nonlinear_constraint(
        self,
        nonlincon: Callable,
        name: Optional[str] = None,
        n_nonlinear_constraints: int = 1,
        bad_metrics: Optional[float | list[float]] = None,
        evaluation_objects: Any = -1,
        bounds: float | list[float] = 0,
        comparison_operator: str = "le",
        labels: Optional[list[str]] = None,
        requires: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a nonlinear constraint function.

        Parameters
        ----------
        nonlincon : callable
            Constraint function; must return *n_nonlinear_constraints* values.
        name : str, optional
            Name; defaults to ``nonlincon.__name__``.
        n_nonlinear_constraints : int
            Number of metrics returned.
        bad_metrics : float or list of floats, optional
            Fallback values on failure.
        evaluation_objects : {-1, None, object, list}
            Evaluation objects to use.
        bounds : float or list of floats
            Per-metric upper limits (for ``le``) or lower limits (for ``ge``).
        comparison_operator : {'le', 'ge'}
            Direction of the constraint.
        labels : list[str], optional
            Metric labels.
        requires : callable or list of callables, optional
            Upstream evaluators.

        Raises
        ------
        TypeError
            If *nonlincon* is not callable.
        CADETProcessError
            If a referenced evaluation object or evaluator is not registered,
            or if *bounds* has the wrong length.
        """
        if not callable(nonlincon):
            raise TypeError("Expected callable constraint function.")

        if name is None:
            if inspect.isfunction(nonlincon) or inspect.ismethod(nonlincon):
                name = nonlincon.__name__
            else:
                name = str(nonlincon)

        if name in self.nonlinear_constraint_names:
            warnings.warn("Nonlinear constraint with same name already exists.")

        # Resolve evaluation objects.
        if evaluation_objects is None:
            eval_objs: list[Any] = []
        elif evaluation_objects == -1:
            eval_objs = list(self.evaluation_objects)
        elif not isinstance(evaluation_objects, list):
            eval_objs = [evaluation_objects]
        else:
            eval_objs = list(evaluation_objects)
        for el in eval_objs:
            if el not in self.evaluation_objects:
                raise CADETProcessError(f"Unknown EvaluationObject: {el!r}")

        # Normalize bounds.
        if isinstance(bounds, (int, float)):
            bounds_list = n_nonlinear_constraints * [float(bounds)]
        else:
            bounds_list = list(bounds)
        if len(bounds_list) != n_nonlinear_constraints:
            raise CADETProcessError(
                f"Expected {n_nonlinear_constraints} bounds, got {len(bounds_list)}"
            )

        # Resolve evaluator chain and lazily register in pipeline.
        if requires is None:
            req_list: list = []
        elif not isinstance(requires, list):
            req_list = [requires]
        else:
            req_list = list(requires)
        evaluator_chain: list[str] = []
        for req in req_list:
            if req not in self._evaluator_names:
                raise CADETProcessError(f"Unknown Evaluator: {req!r}")
            evaluator_chain.append(self._evaluator_names[req])
        self._register_evaluator_chain(req_list)

        record = _MetricRecord(
            nonlincon,
            name,
            n_metrics=n_nonlinear_constraints,
            bad_metrics=bad_metrics,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            labels=labels,
            args=args,
            kwargs=kwargs if kwargs else None,
            bounds=bounds_list,
            comparison_operator=comparison_operator,
        )
        self._nonlinear_constraints.append(record)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @property
    def callbacks(self) -> list[_MetricRecord]:
        """Registered callback records."""
        return self._callbacks

    @property
    def callback_names(self) -> list[str]:
        """Names of all callbacks."""
        return [cb.name for cb in self._callbacks]

    @property
    def n_callbacks(self) -> int:
        """Number of registered callbacks."""
        return len(self._callbacks)

    def add_callback(
        self,
        callback: Callable,
        name: Optional[str] = None,
        evaluation_objects: Any = -1,
        requires: Any = None,
        frequency: int = 1,
        callbacks_dir: Optional[str] = None,
        keep_progress: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a callback function.

        Parameters
        ----------
        callback : callable
            Callback function.
        name : str, optional
            Name; defaults to ``callback.__name__``.
        evaluation_objects : {-1, None, object, list}
            Evaluation objects to use.
        requires : callable or list of callables, optional
            Upstream evaluators.
        frequency : int
            Evaluate every *frequency* generations.
        callbacks_dir : str, optional
            Directory to store callback output.
        keep_progress : bool
            Retain progress files between calls.
        """
        if not callable(callback):
            raise TypeError("Expected callable callback.")
        if frequency < 1:
            raise ValueError(f"frequency must be a positive integer, got {frequency!r}")

        if name is None:
            if inspect.isfunction(callback) or inspect.ismethod(callback):
                name = callback.__name__
            else:
                name = str(callback)

        if name in self.callback_names:
            warnings.warn("Callback with same name already exists.")
            raise CADETProcessError("Callback with same name already exists.")

        if evaluation_objects is None:
            eval_objs: list[Any] = []
        elif evaluation_objects == -1:
            eval_objs = list(self.evaluation_objects)
        elif not isinstance(evaluation_objects, list):
            eval_objs = [evaluation_objects]
        else:
            eval_objs = list(evaluation_objects)
        for el in eval_objs:
            if el not in self.evaluation_objects:
                raise CADETProcessError(f"Unknown EvaluationObject: {el!r}")

        if requires is None:
            req_list: list = []
        elif not isinstance(requires, list):
            req_list = [requires]
        else:
            req_list = list(requires)
        evaluator_chain: list[str] = []
        for req in req_list:
            if req not in self._evaluator_names:
                raise CADETProcessError(f"Unknown Evaluator: {req!r}")
            evaluator_chain.append(self._evaluator_names[req])
        self._register_evaluator_chain(req_list)

        record = _MetricRecord(
            callback,
            name,
            n_metrics=1,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            args=args,
            kwargs=kwargs if kwargs else None,
            frequency=frequency,
            callbacks_dir=callbacks_dir,
            keep_progress=keep_progress,
        )
        self._callbacks.append(record)

    # ── Meta scores ───────────────────────────────────────────────────────────

    @property
    def meta_scores(self) -> list[_MetricRecord]:
        """Registered meta-score records."""
        return self._meta_scores

    @property
    def meta_score_names(self) -> list[str]:
        """Names of all meta scores."""
        return [ms.name for ms in self._meta_scores]

    @property
    def meta_score_labels(self) -> list[str]:
        """Flat list of labels across all meta scores."""
        labels = []
        for ms in self._meta_scores:
            labels += ms.labels
        return labels

    @property
    def n_meta_scores(self) -> int:
        """Number of meta scores."""
        return len(self._meta_scores)

    def add_meta_score(
        self,
        func: Callable,
        n_meta_scores: int = 1,
        labels: Any = None,
        bad_metrics: Any = None,
    ) -> None:
        """Register a meta-score function (post-objective ranking criterion)."""
        if not callable(func):
            raise TypeError("Expected callable meta-score function.")

        name = getattr(func, "__name__", str(func))
        record = _MetricRecord(
            func,
            name,
            n_metrics=n_meta_scores,
            bad_metrics=bad_metrics,
            labels=labels,
        )
        self._meta_scores.append(record)

    # ── Multi-criteria decision functions ─────────────────────────────────────

    @property
    def multi_criteria_decision_functions(self) -> list:
        """Registered multi-criteria decision functions."""
        return self._multi_criteria_decision_functions

    @property
    def n_multi_criteria_decision_functions(self) -> int:
        """Number of registered multi-criteria decision functions."""
        return len(self._multi_criteria_decision_functions)

    def add_multi_criteria_decision_function(self, func: Any) -> None:
        """Register a multi-criteria decision function."""
        self._multi_criteria_decision_functions.append(func)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def _register_evaluator_chain(self, req_list: list[Callable]) -> None:
        """Lazily register evaluators in the pipeline with dependency edges.

        Called by ``add_objective``, ``add_nonlinear_constraint``, and
        ``add_callback``.  Evaluators that are already registered in the
        pipeline are skipped.  The linear ordering of *req_list* implies the
        dependency chain: ``req_list[i]`` takes the output of
        ``req_list[i-1]`` as its single argument.
        """
        for i, ev_callable in enumerate(req_list):
            ev_name = self._evaluator_names[ev_callable]
            if ev_name not in self._pipeline._output_names:
                prev = (
                    [self._evaluator_names[req_list[i - 1]]] if i > 0 else None
                )
                self._pipeline.add_evaluator(
                    self._evaluator_func_by_name[ev_name],
                    output_name=ev_name,
                    requires=prev,
                )

    def _evaluate_individual(
        self,
        x: npt.ArrayLike,
        target_functions: list[_MetricRecord],
    ) -> np.ndarray:
        """Evaluate all *target_functions* for a single parameter vector.

        Writes *x* into evaluation objects, precomputes all evaluator outputs
        via the pipeline (sharing across metrics that need the same
        intermediate), then calls each metric's callable with the appropriate
        input.

        When ``set_values`` raises (e.g. out-of-bounds x from the optimizer),
        all metrics return their ``bad_metrics`` fallback.
        """
        x = np.asarray(x, dtype=float).ravel()

        def _bad_for(metric: _MetricRecord) -> np.ndarray:
            n = len(
                metric.evaluation_objects
                if metric.evaluation_objects
                else self._space.evaluation_objects or [None]
            )
            return np.tile(metric.bad_metrics, n)

        # Collect all unique evaluator output names needed across metrics.
        all_ev_names = list({n for m in target_functions for n in m.evaluator_chain})

        # Precompute evaluator outputs via pipeline when eval objects exist.
        # evaluate() handles set_values and caching via EvaluationContext.
        ev_cache: dict[tuple[int, str], Any] = {}
        eval_objs = self._space.evaluation_objects
        if all_ev_names and eval_objs:
            try:
                outputs = self._pipeline.evaluate(x, targets=all_ev_names)
            except CADETProcessError as e:
                self.logger.warning(
                    "Evaluation failed at x=%s: %s. Returning bad metrics.", x, e
                )
                return np.concatenate([_bad_for(m) for m in target_functions])
            except Exception:
                self.logger.warning(
                    "Unexpected error during pipeline evaluation at x=%s.",
                    x,
                    exc_info=True,
                )
                return np.concatenate([_bad_for(m) for m in target_functions])
            # evaluate() returns {target: list} for multiple objects, scalar for one.
            for ev_name, val in outputs.items():
                if len(eval_objs) == 1:
                    ev_cache[(0, ev_name)] = val
                else:
                    for i, v in enumerate(val):
                        ev_cache[(i, ev_name)] = v
        elif not all_ev_names:
            # No evaluator chain — set_values still needs to happen for inline metrics.
            try:
                self._space.set_values(self._space.transformed_space.decode(x))
            except CADETProcessError as e:
                self.logger.warning(
                    "set_values failed at x=%s: %s. Returning bad metrics.", x, e
                )
                return np.concatenate([_bad_for(m) for m in target_functions])
            except Exception:
                self.logger.warning(
                    "Unexpected error in set_values at x=%s.", x, exc_info=True
                )
                return np.concatenate([_bad_for(m) for m in target_functions])

        results = np.empty(0)
        for metric in target_functions:
            metric_eval_objs = (
                metric.evaluation_objects if metric.evaluation_objects
                else eval_objs or [None]
            )
            for i, eval_obj in enumerate(metric_eval_objs):
                if metric.evaluator_chain:
                    last_ev = metric.evaluator_chain[-1]
                    if eval_obj is None:
                        # No eval objects: run the chain inline on x.
                        current: Any = x
                        for ev_name in metric.evaluator_chain:
                            ev_func = self._evaluator_func_by_name[ev_name]
                            try:
                                current = ev_func(current)
                            except Exception as exc:
                                current = EvaluationFailure(
                                    stage=ev_name, reason=str(exc), exc=exc
                                )
                                break
                    else:
                        obj_idx = eval_objs.index(eval_obj)
                        current = ev_cache.get(
                            (obj_idx, last_ev),
                            EvaluationFailure(stage=last_ev, reason="not computed"),
                        )
                else:
                    current = x if eval_obj is None else eval_obj

                if isinstance(current, EvaluationFailure):
                    result = metric.bad_metrics
                else:
                    try:
                        result = np.atleast_1d(
                            np.asarray(
                                metric.func(current, *metric.args, **metric.kwargs),
                                dtype=float,
                            )
                        )
                        if len(result) != metric.n_metrics:
                            result = metric.bad_metrics
                    except Exception:
                        self.logger.warning(
                            "Metric '%s' failed at x=%s.", metric.name, x, exc_info=True
                        )
                        result = metric.bad_metrics

                results = np.hstack((results, result))

        return results

    def _evaluate_population(
        self,
        X: npt.ArrayLike,
        target_functions: list[_MetricRecord],
        parallelization_backend: Any = None,
    ) -> np.ndarray:
        """Evaluate *target_functions* for each row of *X*.

        Parameters
        ----------
        X : array-like, shape (n_individuals, n_variables)
            Population in physical (untransformed) space.
        parallelization_backend : ParallelizationBackendBase, optional
            When provided, rows are dispatched via ``backend.evaluate``.
            When None, evaluation is sequential.

        Returns
        -------
        np.ndarray, shape (n_individuals, n_metrics)
        """
        X = np.array(X, ndmin=2)

        def evaluate(x: npt.ArrayLike) -> np.ndarray:
            return self._evaluate_individual(x, target_functions)
        if parallelization_backend is None:
            rows = [evaluate(x) for x in X]
        else:
            rows = parallelization_backend.evaluate(evaluate, X)
        return np.array(rows, ndmin=2)

    # ── Public evaluation API ─────────────────────────────────────────────────

    def evaluate_objectives(
        self,
        X: npt.ArrayLike,
        untransform: bool = False,
        ensure_minimization: bool = False,
        parallelization_backend: Any = None,
        get_dependent_values: bool = True,
    ) -> np.ndarray:
        """Evaluate all objectives for each individual in *X*.

        Parameters
        ----------
        X : array-like
            1-D (single individual) or 2-D (population) of independent
            parameter values in physical space (or normalized when
            ``untransform=True``).
        untransform : bool
            When True, denormalize *X* from normalized coordinates first.
        ensure_minimization : bool
            When True, negate maximization objectives.
        parallelization_backend : ParallelizationBackendBase, optional
            When provided, individuals are evaluated in parallel.
        get_dependent_values : bool
            When True (default), *X* contains independent values only and
            dependent values will be resolved internally.  When False, *X* is
            a full parameter vector; the independent part is extracted first.

        Returns
        -------
        np.ndarray
            1-D when *X* is 1-D; 2-D (n_individuals, n_metrics) otherwise.
        """
        X = np.array(X)
        X_2d = np.array(X, ndmin=2)

        if untransform:
            X_2d = np.array([self.untransform(x) for x in X_2d])

        if not get_dependent_values:
            X_2d = np.array([self.get_independent_values(x) for x in X_2d])

        Y = self._evaluate_population(X_2d, self._objectives, parallelization_backend)
        Y_2d = Y.reshape(len(X_2d), -1)

        if ensure_minimization:
            Y_2d = self._apply_minimization_transform(Y_2d)

        if X.ndim == 1:
            return Y_2d[0]
        return Y_2d

    def evaluate_nonlinear_constraints(
        self,
        X: npt.ArrayLike,
        untransform: bool = False,
        parallelization_backend: Any = None,
        get_dependent_values: bool = True,
    ) -> np.ndarray:
        """Evaluate all nonlinear constraints for each individual in *X*.

        Parameters
        ----------
        X : array-like
            1-D (single individual) or 2-D (population) of independent
            parameter values in physical space (or normalized when
            ``untransform=True``).
        untransform : bool
            When True, denormalize *X* from normalized coordinates first.
        parallelization_backend : ParallelizationBackendBase, optional
            When provided, individuals are evaluated in parallel.
        get_dependent_values : bool
            When True (default), *X* contains independent values only and
            dependent values will be resolved internally.  When False, *X* is
            a full parameter vector; the independent part is extracted first.

        Returns
        -------
        np.ndarray
            1-D when *X* is 1-D; 2-D (n_individuals, n_metrics) otherwise.
        """
        X = np.array(X)
        X_2d = np.array(X, ndmin=2)

        if untransform:
            X_2d = np.array([self.untransform(x) for x in X_2d])

        if not get_dependent_values:
            X_2d = np.array([self.get_independent_values(x) for x in X_2d])

        Y = self._evaluate_population(X_2d, self._nonlinear_constraints, parallelization_backend)
        Y_2d = Y.reshape(len(X_2d), -1)

        if X.ndim == 1:
            return Y_2d[0]
        return Y_2d

    def evaluate_nonlinear_constraints_violation(
        self,
        X: npt.ArrayLike,
        untransform: bool = False,
        parallelization_backend: Any = None,
        get_dependent_values: bool = True,
    ) -> np.ndarray:
        """Evaluate nonlinear constraint violation for each individual in *X*.

        Violation is positive when the constraint is breached.  For ``le``
        constraints: ``violation = f(x) - bound``; for ``ge``: ``violation =
        bound - f(x)``.

        Parameters
        ----------
        X : array-like
            Independent parameter values (or full vector when
            *get_dependent_values* is True, or normalized when *untransform* is
            True).
        get_dependent_values : bool
            When True, *X* is treated as a full parameter vector; the
            independent part is extracted before evaluation.

        Returns
        -------
        np.ndarray
            Same shape convention as ``evaluate_nonlinear_constraints``.
        """
        factors = []
        for nc in self._nonlinear_constraints:
            factor = -1 if nc.comparison_operator == "ge" else 1
            factors += nc.n_total_metrics * [factor]

        G = self.evaluate_nonlinear_constraints(
            X,
            untransform=untransform,
            parallelization_backend=parallelization_backend,
            get_dependent_values=get_dependent_values,
        )

        factors_arr = np.array(factors)
        G_transformed = np.multiply(factors_arr, G)
        bounds_transformed = np.multiply(factors_arr, self.nonlinear_constraints_bounds)
        return G_transformed - bounds_transformed

    def check_nonlinear_constraints(
        self,
        x: npt.ArrayLike,
        cv_nonlincon_tol: float | np.ndarray = 0.0,
        get_dependent_values: bool = True,
    ) -> bool:
        """Return True if *x* satisfies all nonlinear constraints.

        Parameters
        ----------
        x : array-like
            Independent parameter values by default.  Pass
            ``get_dependent_values=False`` when supplying a full parameter
            vector (independent + dependent).
        cv_nonlincon_tol : float or array-like
            Per-constraint violation tolerance.
        get_dependent_values : bool
            When True (default), *x* contains independent values only.
            When False, *x* is a full parameter vector; the independent part
            is extracted before evaluation.
        """
        x = np.asarray(x, dtype=float).ravel()
        cv = np.atleast_1d(
            np.array(
                self.evaluate_nonlinear_constraints_violation(
                    x, get_dependent_values=get_dependent_values
                ),
                dtype=float,
            )
        )

        if np.isscalar(cv_nonlincon_tol):
            cv_nonlincon_tol = np.repeat(float(cv_nonlincon_tol), self.n_nonlinear_constraints)

        if len(cv_nonlincon_tol) != self.n_nonlinear_constraints:
            raise ValueError(
                f"Length of cv_nonlincon_tol ({len(cv_nonlincon_tol)}) does not "
                f"match number of constraints ({self.n_nonlinear_constraints})."
            )

        return bool(np.all(cv <= cv_nonlincon_tol))

    def objective_jacobian(
        self,
        x: npt.ArrayLike,
        untransform: bool = False,
        ensure_minimization: bool = False,
        dx: float = 1e-3,
    ) -> np.ndarray:
        """Compute the Jacobian of the objectives via forward finite differences.

        Parameters
        ----------
        x : array-like
            Independent parameter values in physical space (or normalized when
            ``untransform=True``).
        untransform : bool, default=False
            When True, each finite-difference probe is untransformed before
            evaluation, so the returned Jacobian is with respect to the
            transformed variables.
        ensure_minimization : bool, default=False
            When True, negate maximization objectives.
        dx : float, default=1e-3
            Finite-difference step size.

        Returns
        -------
        np.ndarray, shape (n_objectives, n_variables)
        """
        x = np.asarray(x, dtype=float).ravel()
        return _approximate_jac(
            x,
            self.evaluate_objectives,
            dx,
            untransform=untransform,
            ensure_minimization=ensure_minimization,
        )

    def nonlinear_constraint_jacobian(
        self,
        x: npt.ArrayLike,
        untransform: bool = False,
        dx: float = 1e-3,
    ) -> np.ndarray:
        """Compute the Jacobian of the nonlinear constraints via finite differences.

        Parameters
        ----------
        x : array-like
            Independent parameter values in physical space (or normalized when
            ``untransform=True``).
        untransform : bool, default=False
            When True, each finite-difference probe is untransformed before
            evaluation.
        dx : float, default=1e-3
            Finite-difference step size.

        Returns
        -------
        np.ndarray, shape (n_nonlinear_constraints, n_variables)
        """
        x = np.asarray(x, dtype=float).ravel()
        return _approximate_jac(
            x,
            self.evaluate_nonlinear_constraints,
            dx,
            untransform=untransform,
        )

    # ── Minimization transform ─────────────────────────────────────────────────

    def _apply_minimization_transform(self, F: np.ndarray) -> np.ndarray:
        """Negate columns that correspond to maximization objectives."""
        factors: list[int] = []
        for obj in self._objectives:
            n = obj.n_total_metrics
            factors += n * (-1 if not obj.minimize else 1,)
        return F * np.array(factors)

    def transform_maximization(self, F: Any, scores: Any = None) -> Any:
        """Negate maximization-objective columns in *F*.

        Parameters
        ----------
        F : array-like
            Objective values, shape (..., n_objectives).
        scores : {'objectives', 'meta_scores', None}
            Which metric type to apply the transform to.
        """
        if scores == "objectives":
            return self._apply_minimization_transform(np.array(F))
        return F

    # ── Callbacks evaluation ──────────────────────────────────────────────────

    def evaluate_callbacks(
        self,
        population: Any = None,
        current_iteration: int | Literal["final"] = 0,
        callbacks_dir: Any = None,
        parallelization_backend: Any = None,  # noqa: ARG002
    ) -> None:
        """Evaluate registered callbacks against a population.

        Parameters
        ----------
        population :
            Population or list of individuals passed to each callback.
            When None or no callbacks are registered the call is a no-op.
        current_iteration : int
            Current generation index; used to respect each callback's
            ``frequency`` setting.
        callbacks_dir : path-like, optional
            Base directory for callback output files.  Per-callback
            subdirectories are created automatically when more than one
            callback is registered.  A ``callbacks_dir`` set at registration
            time via ``add_callback`` overrides this value for that callback.
        parallelization_backend :
            Unused; retained for API compatibility with the optimizer.
        """
        if population is None or not self._callbacks:
            return
        _logger = logging.getLogger(__name__)
        eval_objs = self._space.evaluation_objects or []
        obj_index = {id(obj): i for i, obj in enumerate(eval_objs)}
        for cb in self._callbacks:
            if not (
                current_iteration == "final"
                or current_iteration % cb.frequency == 0
            ):
                continue

            # Resolve per-callback directory.
            if cb.callbacks_dir is not None:
                _cb_dir = Path(cb.callbacks_dir)
            elif callbacks_dir is not None:
                base = Path(callbacks_dir)
                _cb_dir = base / str(cb) if len(self._callbacks) > 1 else base
                _cb_dir.mkdir(exist_ok=True, parents=True)
            else:
                _cb_dir = None

            if _cb_dir is not None and current_iteration != "final":
                cb.cleanup(_cb_dir, current_iteration)

            metric_eval_objs = (
                cb.evaluation_objects if cb.evaluation_objects else eval_objs or [None]
            )
            try:
                sig = inspect.signature(cb.func).parameters
            except (ValueError, TypeError):
                sig = {}
            for individual in population:
                x_ind = self.untransform(individual.x_transformed)
                self._space.set_values(self._space.transformed_space.decode(x_ind))
                # Use the pipeline to get evaluator chain outputs, benefiting
                # from results already cached during objective/constraint
                # evaluation for this individual.
                ev_outputs: dict[str, Any] = {}
                if cb.evaluator_chain and eval_objs:
                    try:
                        ev_outputs = self._pipeline.evaluate(
                            x_ind, targets=cb.evaluator_chain
                        )
                    except Exception:
                        _logger.debug(
                            f"Pipeline evaluation for callback {cb.name!r} failed;"
                            f" falling back to direct chain execution.",
                            exc_info=True,
                        )

                for eval_obj in metric_eval_objs:
                    try:
                        if cb.evaluator_chain:
                            last = cb.evaluator_chain[-1]
                            if ev_outputs and last in ev_outputs:
                                raw = ev_outputs[last]
                                if isinstance(raw, list):
                                    idx = obj_index.get(id(eval_obj))
                                    if idx is not None and idx < len(raw):
                                        chain_result = raw[idx]
                                    else:
                                        chain_result = (
                                            eval_obj
                                            if eval_obj is not None
                                            else individual.x
                                        )
                                        for ev_name in cb.evaluator_chain:
                                            chain_result = self._evaluator_func_by_name[
                                                ev_name
                                            ](chain_result)
                                else:
                                    chain_result = raw
                            else:
                                # Pipeline unavailable; fall back to direct execution.
                                chain_result = (
                                    eval_obj if eval_obj is not None else individual.x
                                )
                                for ev_name in cb.evaluator_chain:
                                    chain_result = self._evaluator_func_by_name[
                                        ev_name
                                    ](chain_result)
                        else:
                            chain_result = (
                                eval_obj if eval_obj is not None else individual.x
                            )
                        kwargs = dict(cb.kwargs)
                        if "individual" in sig:
                            kwargs["individual"] = individual
                        if "evaluation_object" in sig:
                            kwargs["evaluation_object"] = eval_obj
                        if "callbacks_dir" in sig:
                            kwargs["callbacks_dir"] = _cb_dir
                        cb.func(chain_result, *cb.args, **kwargs)
                    except Exception as exc:
                        _logger.warning(
                            f"Callback {cb.name!r} failed at iteration"
                            f" {current_iteration}: {exc}",
                            exc_info=True,
                        )

    def evaluate_callbacks_population(self, *args: Any, **kwargs: Any) -> None:
        """Call ``evaluate_callbacks``; deprecated, use that method directly."""
        warnings.warn(
            "evaluate_callbacks_population is deprecated; use evaluate_callbacks.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_callbacks(*args, **kwargs)

    # ── Meta scores evaluation ────────────────────────────────────────────────

    def evaluate_meta_scores(
        self,
        X: npt.ArrayLike,
        untransform: bool = False,
        parallelization_backend: Any = None,
        get_dependent_values: bool = True,
    ) -> np.ndarray:
        """Evaluate meta-score functions for each individual in *X*.

        Meta scores run through the same evaluation pipeline as objectives:
        parameter values are written into evaluation objects, the evaluator
        chain runs, and each meta-score function receives the chain output.

        Parameters
        ----------
        X : array-like
            1-D (single individual) or 2-D (population) of independent
            parameter values in physical space (or normalized when
            ``untransform=True``).
        untransform : bool
            When True, denormalize *X* from normalized coordinates first.
        parallelization_backend : ParallelizationBackendBase, optional
            When provided, individuals are evaluated in parallel.
        get_dependent_values : bool
            When True, *X* is treated as a full parameter vector (independent +
            dependent); the independent part is extracted before evaluation.

        Returns
        -------
        np.ndarray
            1-D when *X* is 1-D; 2-D (n_individuals, n_meta_scores) otherwise.
        """
        X = np.array(X)
        X_2d = np.array(X, ndmin=2)

        if untransform:
            X_2d = np.array([self.untransform(x) for x in X_2d])

        if not get_dependent_values:
            X_2d = np.array([self.get_independent_values(x) for x in X_2d])

        if not self._meta_scores:
            return np.zeros((len(X_2d), 0))

        Y = self._evaluate_population(X_2d, self._meta_scores, parallelization_backend)
        Y_2d = Y.reshape(len(X_2d), -1)

        if X.ndim == 1:
            return Y_2d[0]
        return Y_2d

    def evaluate_multi_criteria_decision_functions(
        self, pareto_front: Any = None
    ) -> list:
        """Apply registered MCDFs to the Pareto front; return indices of selected individuals.

        Each registered function receives *pareto_front* (a ``Population``) and
        must return a sequence of indices selecting "best" individuals from it.
        When multiple functions are registered, results are concatenated (union).
        Returns an empty list when no functions are registered.

        Parameters
        ----------
        pareto_front : Population
            The current Pareto front.

        Returns
        -------
        list
            Indices into *pareto_front* for the selected individuals.
        """
        if not self._multi_criteria_decision_functions:
            return []
        selected: list = []
        for mcdf in self._multi_criteria_decision_functions:
            result = mcdf(pareto_front)
            if result is not None:
                selected.extend(result)
        return selected

    # ── Individual and population creation ────────────────────────────────────

    def create_individual(
        self,
        x: npt.ArrayLike,
        f: npt.ArrayLike | None = None,
        f_minimized: npt.ArrayLike | None = None,
        g: npt.ArrayLike | None = None,
        cv_nonlincon: npt.ArrayLike | None = None,
        m: npt.ArrayLike | None = None,
        m_minimized: npt.ArrayLike | None = None,
    ) -> Individual:
        """Create an ``Individual`` from a full parameter vector and metric values."""
        x = np.asarray(x, dtype=float)
        x_indep = self.get_independent_values(x)
        x_transformed = self.transform(x_indep)

        cv_bounds = self.evaluate_bounds(x, get_dependent_values=False)
        cv_lincon = self.evaluate_linear_constraints(x, get_dependent_values=False)
        cv_lineqcon = np.abs(
            self.evaluate_linear_equality_constraints(x, get_dependent_values=False)
        )

        return Individual(
            x=x,
            x_transformed=x_transformed,
            cv_bounds=cv_bounds,
            cv_lincon=cv_lincon,
            cv_lineqcon=cv_lineqcon,
            f=f,
            f_minimized=f_minimized,
            g=g,
            cv_nonlincon=cv_nonlincon,
            m=m,
            m_minimized=m_minimized,
            independent_variable_names=self.independent_variable_names,
            objective_labels=self.objective_labels,
            nonlinear_constraint_labels=self.nonlinear_constraint_labels,
            meta_score_labels=self.meta_score_labels,
            variable_names=self.variable_names,
        )

    def create_population(
        self,
        X: npt.ArrayLike,
        F: npt.ArrayLike | None = None,
        F_minimized: npt.ArrayLike | None = None,
        G: npt.ArrayLike | None = None,
        CV_nonlincon: npt.ArrayLike | None = None,
        M: npt.ArrayLike | None = None,
        M_minimized: npt.ArrayLike | None = None,
        untransform: bool = False,
        get_dependent_values: bool = False,
    ) -> Population:
        """Create a ``Population`` from arrays of parameter vectors and metric values."""
        X = np.array(X, ndmin=2)
        if untransform:
            X = np.array([self.untransform(x) for x in X])
        if get_dependent_values:
            X = np.array([self._resolve_full_vector(x) for x in X])

        n = len(X)

        def _to_rows(arr: npt.ArrayLike | None) -> list:
            if arr is None:
                return n * [None]
            return list(np.array(arr, ndmin=2))

        F_rows = _to_rows(F)
        F_min_rows = F_rows if F_minimized is None else _to_rows(F_minimized)
        G_rows = _to_rows(G)
        CV_rows = G_rows if CV_nonlincon is None else _to_rows(CV_nonlincon)
        M_rows = _to_rows(M)
        M_min_rows = M_rows if M_minimized is None else _to_rows(M_minimized)

        pop = Population()
        for x, f, f_min, g, cv, m, m_min in zip(
            X, F_rows, F_min_rows, G_rows, CV_rows, M_rows, M_min_rows
        ):
            ind = self.create_individual(
                x,
                f=f,
                f_minimized=f_min,
                g=g,
                cv_nonlincon=cv,
                m=m,
                m_minimized=m_min,
            )
            pop.add_individual(ind)

        return pop

    # ── String representations ────────────────────────────────────────────────

    def __str__(self) -> str:
        """Return the problem name."""
        return self.name

    def __repr__(self) -> str:
        """Return a short developer representation."""
        return (
            f"OptimizationProblem(name={self.name!r}, "
            f"n_variables={self.n_variables}, "
            f"n_independent_variables={self.n_independent_variables})"
        )
