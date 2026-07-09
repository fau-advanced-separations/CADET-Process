"""OptimizationProblem: optimization policy on top of ParameterSpace.

Covers the parameter side (evaluation objects, variables, bounds, linear
constraints, dependencies, transforms, initial value sampling) and the
evaluation side (objectives, nonlinear constraints, callbacks, evaluators).
"""

from __future__ import annotations

import inspect
import logging
import math
import re
import shutil
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError, log
from CADETProcess.dataStructure.deprecation import deprecated_alias
from CADETProcess.dataStructure.nested_dict import attribute_path_exists
from CADETProcess.evaluation_pipeline import EvaluationFailure, EvaluationPipeline
from CADETProcess.metric_space import Metric, MetricSpace
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
from CADETProcess.problem import Problem

if TYPE_CHECKING:
    from CADETProcess.parameter_space.sampling import SamplerBase

__all__ = ["OptimizationProblem"]


# ── Metric annotation ─────────────────────────────────────────────────────


class _CallbackRecord:
    """Scheduling record for a callback.

    Callbacks are not metrics: they produce files, not values.  The record
    holds the callable, its wiring (evaluation objects, evaluator chain),
    and scheduling policy only.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        evaluation_objects: list | None = None,
        evaluator_chain: list[str] | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        frequency: int = 1,
        callbacks_dir: Any = None,
        keep_progress: bool = False,
    ) -> None:
        self.func = func
        self.name = name
        self.evaluation_objects: list = list(evaluation_objects) if evaluation_objects else []
        self.evaluator_chain: list[str] = list(evaluator_chain) if evaluator_chain else []
        self.args = args if args else ()
        self.kwargs = kwargs if kwargs is not None else {}
        self.frequency = frequency
        self.callbacks_dir = callbacks_dir
        self.keep_progress = keep_progress
        # Per-call state (individual, evaluation_object, callbacks_dir) set by
        # evaluate_callbacks just before triggering the pipeline node; the
        # node reads it because these values cannot travel through the DAG.
        self.runtime: dict[str, Any] = {}

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


class _MetricRecord:
    """Execution record binding an objective/constraint callable to its declaration.

    The declaration (name, shape, labels) is the ``Metric`` registered in the
    problem's ``MetricSpace``; direction and bounds live on the ``Objective``
    or ``Constraint`` annotation there.  This record holds only what
    evaluation needs: the callable, its evaluator chain, evaluation objects,
    and fallback values.

    ``n_metrics`` is the per-evaluation-object entry count (what the callable
    returns); the declaration's ``n_metrics`` is the total across evaluation
    objects, exposed here as ``n_total_metrics``.
    """

    def __init__(
        self,
        func: Callable,
        metric: Metric,
        annotation: Any,
        n_per_object: int,
        bad_metrics: float | npt.ArrayLike | None = None,
        evaluation_objects: list | None = None,
        evaluator_chain: list[str] | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        self.func = func
        self.metric = metric
        self.annotation = annotation
        self.n_metrics = n_per_object
        if bad_metrics is None:
            self.bad_metrics = np.full(n_per_object, np.inf)
        elif np.isscalar(bad_metrics):
            self.bad_metrics = np.full(n_per_object, float(bad_metrics))
        else:
            self.bad_metrics = np.asarray(bad_metrics, dtype=float)
        self.evaluation_objects: list = list(evaluation_objects) if evaluation_objects else []
        self.evaluator_chain: list[str] = list(evaluator_chain) if evaluator_chain else []
        self.args = args if args else ()
        self.kwargs = kwargs if kwargs is not None else {}

    @property
    def name(self) -> str:
        """Name of the underlying metric declaration."""
        return self.metric.name

    @property
    def labels(self) -> list[str]:
        """Expanded labels from the declaration; order matches the data."""
        return self.metric.labels

    @property
    def n_total_metrics(self) -> int:
        """Total metric count across all evaluation objects."""
        return self.metric.n_metrics

    @property
    def minimize(self) -> bool:
        """Direction from the Objective annotation."""
        return self.annotation.minimize

    @property
    def bounds(self) -> list[float]:
        """Expanded bounds from the Constraint annotation."""
        return list(self.annotation.bounds)

    @property
    def comparison_operator(self) -> str:
        """Comparison operator from the Constraint annotation."""
        return self.annotation.comparison_operator

    def __str__(self) -> str:
        return self.name


# ── Helpers ────────────────────────────────────────────────────────────────


def _adapt_root_input(parameter_space: ParameterSpace, value: Any) -> Any:
    """Adapt a pipeline root input for user callables.

    Root nodes receive the evaluation object, or, in the
    zero-evaluation-object mode, the named assignment itself.  User callables
    on free-variable problems are written against the numeric vector, so a
    Mapping root input is converted to the physical x vector (independent
    parameters in registration order).  The conversion is optimizer policy;
    the pipeline itself stays purely named.

    Module-level on purpose: node closures are pickled per node by pipefunc
    (fresh cloudpickle pass, no shared memo), so they must not capture the
    ``OptimizationProblem``, which references the pipeline and would recurse.
    """
    if isinstance(value, Mapping):
        return np.array(
            [value[p.name] for p in parameter_space.independent_parameters],
            dtype=float,
        )
    return value


def _derive_name(func: Callable) -> str:
    """Derive a default node name for a callable.

    Metric, callback, and evaluator names are pipeline output names and must
    be valid Python identifiers.  Functions and methods use their ``__name__``
    (sanitized, so bare lambdas become ``_lambda_``); callable instances use
    their class name, since ``str(obj)`` typically embeds a memory address
    and would not be stable across runs.
    """
    if inspect.isfunction(func) or inspect.ismethod(func):
        name = func.__name__
    else:
        name = type(func).__name__
    return re.sub(r"\W|^(?=\d)", "_", name)


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


class OptimizationProblem(Problem):
    """Optimization policy over ParameterSpace, MetricSpace, and EvaluationPipeline.

    Decides what to optimize: which callables are objectives, which are
    constraints, which are callbacks, and how failures are handled.
    Delegates parameter semantics (bounds, linear constraints,
    normalization, dependency resolution) to ``ParameterSpace``, output
    declarations and annotations (direction, constraint bounds) to
    ``MetricSpace``, and execution to ``EvaluationPipeline``.

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
        self.logger = log.get_logger(name, level=log_level)

        # Disk cache when requested: use_diskcache=True OR an explicit directory.
        effective_cache_dir = cache_directory if use_diskcache else None

        parameter_space = ParameterSpace()
        super().__init__(
            parameter_space=parameter_space,
            metric_space=MetricSpace(),
            backend=EvaluationPipeline(parameter_space, cache_dir=effective_cache_dir),
            name=name,
        )
        self._params: dict[str, ParameterBase] = {}
        self._path_registry: dict[tuple, str] = {}  # (path, obj_id, index_repr) → var_name

        # Evaluator registry: callable → name, name → wrapped callable, ordered list
        self._evaluator_names: dict[Callable, str] = {}        # func → output_name
        self._evaluator_func_by_name: dict[str, Callable] = {}  # output_name → wrapped callable
        self._evaluator_registry: list[tuple[str, Callable]] = []  # ordered (name, func)

        self._objectives: list[_MetricRecord] = []
        self._nonlinear_constraints: list[_MetricRecord] = []
        self._callbacks: list[_CallbackRecord] = []
        self._meta_scores: list[_MetricRecord] = []
        self._multi_criteria_decision_functions: list = []

    # ── Evaluation objects ─────────────────────────────────────────────────────

    def add_evaluation_object(self, obj: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Register an evaluation object."""
        self._parameter_space.add_evaluation_object(obj)

    @property
    def evaluation_objects(self) -> list[Any]:
        """Registered evaluation objects, in insertion order."""
        return self._parameter_space.evaluation_objects

    @property
    def evaluation_objects_dict(self) -> dict[str, Any]:
        """Mapping of ``str(obj)`` → obj for all registered evaluation objects."""
        return {str(obj): obj for obj in self._parameter_space.evaluation_objects}

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
            eval_objs = list(self._parameter_space.evaluation_objects)
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
            self._parameter_space.add_parameter(param)
        elif pre_processing is not None:
            mapper = make_preprocessing_mapper(eval_objs, parameter_path, pre_processing)
            self._parameter_space.add_parameter(param, mapper=mapper)
        elif indices is not None:
            self._parameter_space.add_parameter(
                param, mapper=IndexedMapper(eval_objs, parameter_path, indices)
            )
        else:
            self._parameter_space.add_parameter(
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
            eval_objs = list(self._parameter_space.evaluation_objects)
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
            self._parameter_space.add_parameter(param)
        else:
            self._parameter_space.add_parameter(
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
        return self._parameter_space.parameters

    @property
    def variable_names(self) -> list[str]:
        """Names of all parameters in registration order."""
        return [p.name for p in self._parameter_space.parameters]

    @property
    def variables_dict(self) -> dict:
        """All optimization variables indexed by name."""
        return self._params

    @property
    def n_variables(self) -> int:
        """Total number of parameters (independent + derived)."""
        return len(self._parameter_space.parameters)

    @property
    def independent_variables(self) -> list[ParameterBase]:
        """Parameters that are not derived from other parameters."""
        return self._parameter_space.independent_parameters

    @property
    def independent_variable_names(self) -> list[str]:
        """Names of independent parameters."""
        return [p.name for p in self._parameter_space.independent_parameters]

    @property
    def n_independent_variables(self) -> int:
        """Number of independent (optimizer-facing) parameters."""
        return self._parameter_space.n_variables

    @property
    def dependent_variables(self) -> list[ParameterBase]:
        """Parameters computed from other parameters."""
        return self._parameter_space.dependent_parameters

    @property
    def dependent_variable_names(self) -> list[str]:
        """Names of derived parameters."""
        return [p.name for p in self._parameter_space.dependent_parameters]

    @property
    def n_dependent_variables(self) -> int:
        """Number of derived parameters."""
        return len(self._parameter_space.dependent_parameters)

    @property
    def continuous_variables(self) -> list[RangedParameter]:
        """Independent continuous (float) variables."""
        return self._parameter_space.continuous_parameters

    @property
    def n_continuous_variables(self) -> int:
        """Number of independent continuous variables."""
        return len(self._parameter_space.continuous_parameters)

    @property
    def integer_variables(self) -> list[RangedParameter]:
        """Independent integer variables."""
        return self._parameter_space.integer_parameters

    @property
    def n_integer_variables(self) -> int:
        """Number of independent integer variables."""
        return len(self._parameter_space.integer_parameters)

    @property
    def categorical_variables(self) -> list[ChoiceParameter]:
        """Independent categorical variables."""
        return self._parameter_space.categorical_parameters

    @property
    def n_categorical_variables(self) -> int:
        """Number of independent categorical variables."""
        return len(self._parameter_space.categorical_parameters)

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
            self._parameter_space.add_dependency(derived, ind_params, transform)
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
        all_vals = self._parameter_space._resolve_all_values(x)
        return np.array([all_vals[p.name] for p in self._parameter_space.parameters])

    def get_independent_values(self, x_all: npt.ArrayLike) -> np.ndarray:
        """Extract independent values from a full parameter vector."""
        x_all = np.asarray(x_all, dtype=float).ravel()
        ind_names = {p.name for p in self._parameter_space.independent_parameters}
        return np.array(
            [v for p, v in zip(self._parameter_space.parameters, x_all) if p.name in ind_names]
        )

    def set_variables(self, x: npt.ArrayLike) -> None:
        """Write *x* (independent values) into evaluation objects."""
        self._parameter_space.set_values(self._parameter_space.transformed_space.decode(x))

    def get_variable_value(self, name: str) -> Any:
        """Read the current value of variable *name* from its evaluation object.

        Returns ``None`` when the variable has no path (no evaluation object
        wired) or the mapper does not support read-back.

        Raises
        ------
        KeyError
            If no variable named *name* is registered.
        """
        return self._parameter_space.get_value(name)

    # ── Bounds ────────────────────────────────────────────────────────────────

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds for all variables (independent + dependent); ``-inf`` when unbounded."""
        return self._parameter_space.lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds for all variables (independent + dependent); ``+inf`` when unbounded."""
        return self._parameter_space.upper_bounds

    @property
    def lower_bounds_independent(self) -> np.ndarray:
        """Lower bounds for independent variables only; ``-inf`` when unbounded."""
        return self._parameter_space.lower_bounds_independent

    @property
    def upper_bounds_independent(self) -> np.ndarray:
        """Upper bounds for independent variables only; ``+inf`` when unbounded."""
        return self._parameter_space.upper_bounds_independent

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
        return self._parameter_space.evaluate_bounds(x, resolve_dependencies=get_dependent_values)

    def check_bounds(
        self, x: npt.ArrayLike, tol: float | npt.ArrayLike = 0.0
    ) -> bool:
        """Return True if all independent values satisfy their bounds."""
        return self._parameter_space.check_bounds(x, tol=tol, resolve_dependencies=True)

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
        self._parameter_space.add_linear_constraint(constraint)

    def remove_linear_constraint(self, index: int) -> None:
        """Remove the linear inequality constraint at *index*."""
        self._parameter_space._linear_constraints.pop(index)

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
        self._parameter_space.add_linear_equality_constraint(constraint)

    def remove_linear_equality_constraint(self, index: int) -> None:
        """Remove the linear equality constraint at *index*."""
        self._parameter_space._linear_equality_constraints.pop(index)

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """Registered linear inequality constraints."""
        return self._parameter_space.linear_constraints

    @property
    def n_linear_constraints(self) -> int:
        """Number of registered linear inequality constraints."""
        return len(self._parameter_space.linear_constraints)

    @property
    def linear_equality_constraints(self) -> list[LinearEqualityConstraint]:
        """Registered linear equality constraints."""
        return self._parameter_space.linear_equality_constraints

    @property
    def n_linear_equality_constraints(self) -> int:
        """Number of registered linear equality constraints."""
        return len(self._parameter_space.linear_equality_constraints)

    @property
    def A(self) -> np.ndarray:
        """Inequality constraint matrix over all parameters, shape (m, n_parameters)."""
        return self._parameter_space.A

    @property
    def b(self) -> np.ndarray:
        """Inequality constraint RHS, shape (m,)."""
        return self._parameter_space.b

    @property
    def Aeq(self) -> np.ndarray:
        """Equality constraint matrix over all parameters, shape (m, n_parameters)."""
        return self._parameter_space.A_eq

    @property
    def beq(self) -> np.ndarray:
        """Equality constraint RHS, shape (m,)."""
        return self._parameter_space.b_eq

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
        return self._parameter_space.evaluate_linear_constraints(
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
        if self._parameter_space.A.shape[0] == 0:
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
        return self._parameter_space.evaluate_linear_equality_constraints(
            x, resolve_dependencies=get_dependent_values
        )

    def check_linear_equality_constraints(
        self,
        x: npt.ArrayLike,
        tol: float = 1e-6,
        get_dependent_values: bool = True,
    ) -> bool:
        """Return True if *x* satisfies all equality constraints."""
        if self._parameter_space.A_eq.shape[0] == 0:
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
    def transformed_space(self) -> TransformedSpace:
        """Normalized optimizer view of the parameter space."""
        return self._parameter_space.transformed_space

    # ── Transform / normalization ─────────────────────────────────────────────

    def transform(self, x: npt.ArrayLike) -> np.ndarray:
        """Map independent values from physical to normalized coordinates."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return np.array([self._parameter_space.normalize(row) for row in x])
        return self._parameter_space.normalize(x.ravel())

    def untransform(self, x: npt.ArrayLike) -> np.ndarray:
        """Map independent values from normalized to physical coordinates."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return np.array([self._parameter_space.denormalize(row) for row in x])
        return self._parameter_space.denormalize(x.ravel())

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
        from CADETProcess.parameter_space.sampling import chebyshev_center

        assignment = chebyshev_center(self._parameter_space)
        center = np.array(
            [assignment[p.name] for p in self._parameter_space.independent_parameters]
        )
        if include_dependent_variables:
            return self.get_dependent_values(center)
        return center

    def create_initial_values(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None,
        burn_in: int = 100_000,
        include_dependent_variables: bool = False,
        sampler: Optional[SamplerBase] = None,
    ) -> np.ndarray:
        """Draw feasible initial values from the independent-variable polytope.

        Delegates to *sampler* (default: HopsySampler with pool_size=burn_in);
        encodes the resulting named assignments back to numeric vectors via
        TransformedSpace.encode.

        Parameters
        ----------
        sampler : SamplerBase, optional
            Sampling strategy, e.g. LatinHypercubeSampler or SobolSampler for
            box-bounded problems.  *burn_in* applies only to the default
            HopsySampler.

        Returns
        -------
        np.ndarray, shape (n_samples, n_variables or n_independent_variables)
        """
        from CADETProcess.parameter_space.sampling import HopsySampler

        if sampler is None:
            sampler = HopsySampler(pool_size=int(burn_in))
        assignments = sampler.sample(
            self._parameter_space,
            n_samples,
            seed=seed,
            include_dependent=False,
        )
        ts = self._parameter_space.transformed_space
        rows = []
        for a in assignments:
            x_ind = ts.encode(a)
            rows.append(
                self.get_dependent_values(x_ind) if include_dependent_variables else x_ind
            )
        return np.array(rows, ndmin=2)

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
            name = _derive_name(evaluator)

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
        return self._metric_space.objective_names

    @property
    def objective_labels(self) -> list[str]:
        """Flat list of metric labels across all objectives.

        Order matches the objective vector: expansion over evaluation objects
        is object-major, ``[obj1_a, obj1_b, obj2_a, obj2_b]``.
        """
        return self._metric_space.objective_labels

    @property
    def n_objectives(self) -> int:
        """Total number of objective metrics across all objectives and eval objects."""
        return self._metric_space.n_objectives

    def _build_metric(
        self,
        name: str,
        n_per_object: int,
        base_labels: list[str] | None,
        eval_objs: list,
    ) -> Metric:
        """Build the Metric declaration, expanding over evaluation objects.

        In a multi-object problem every object-bound metric carries an
        explicit ``evaluation_object`` dimension, even when bound to a single
        object: ``Problem.evaluate`` selects the metric's entries from the
        backend's per-object results by these coordinates.  Labels expand
        object-major, matching the flattening order in
        ``_evaluate_individual``.
        """
        if base_labels is not None and len(base_labels) != n_per_object:
            raise CADETProcessError(f"Expected {n_per_object} labels.")
        if base_labels is None:
            if n_per_object == 1:
                base_labels = [name]
            else:
                base_labels = [f"{name}_{i}" for i in range(n_per_object)]
        if eval_objs and len(self.evaluation_objects) > 1:
            obj_names = [str(obj) for obj in eval_objs]
            if n_per_object == 1:
                dims = ("evaluation_object",)
                coords = {"evaluation_object": obj_names}
            else:
                dims = ("evaluation_object", "entry")
                coords = {"evaluation_object": obj_names, "entry": base_labels}
            labels = [f"{obj}_{label}" for obj in obj_names for label in base_labels]
            return Metric(name, dims=dims, coords=coords, labels=labels)
        return Metric(name, n_metrics=n_per_object, labels=base_labels)

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
            name = _derive_name(objective)

        if name in self._metric_space.metrics_dict:
            raise CADETProcessError(
                f"Metric {name!r} is already registered. Objectives and "
                f"nonlinear constraints share one metric namespace; pass "
                f"name= to disambiguate."
            )
        self._check_metric_name(name)

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

        # Declare the metric and annotate it with the direction; raises on
        # duplicate names (named storage requires unique metrics).
        base_labels = labels if labels is not None else getattr(objective, "labels", None)
        metric = self._build_metric(name, n_objectives, base_labels, eval_objs)
        annotation = self._metric_space.add_objective(metric, minimize=minimize)

        # Register the objective itself as a pipeline node: the DAG owns the
        # computation, the annotation is a view over the output.
        requires_node = [evaluator_chain[-1]] if evaluator_chain else None
        self._backend.add_evaluator(
            self._make_metric_node(
                objective, n_objectives, args, kwargs,
                is_root=requires_node is None,
            ),
            output_name=name,
            requires=requires_node,
        )

        record = _MetricRecord(
            objective,
            metric,
            annotation,
            n_per_object=n_objectives,
            bad_metrics=bad_metrics,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            args=args,
            kwargs=kwargs if kwargs else None,
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
        return self._metric_space.constraint_names

    @property
    def nonlinear_constraint_labels(self) -> list[str]:
        """Flat list of labels across all nonlinear constraints.

        Order matches the constraint vector: expansion over evaluation
        objects is object-major.
        """
        return self._metric_space.constraint_labels

    @property
    def nonlinear_constraints_bounds(self) -> list[float]:
        """Flat list of per-metric bounds across all nonlinear constraints.

        Expanded across evaluation objects; length equals
        ``n_nonlinear_constraints``.
        """
        return list(self._metric_space.constraints_bounds)

    @property
    def n_nonlinear_constraints(self) -> int:
        """Total number of nonlinear constraint metrics."""
        return self._metric_space.n_constraints

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
            name = _derive_name(nonlincon)

        if name in self._metric_space.metrics_dict:
            raise CADETProcessError(
                f"Metric {name!r} is already registered. Objectives and "
                f"nonlinear constraints share one metric namespace; pass "
                f"name= to disambiguate."
            )
        self._check_metric_name(name)

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

        # Declare the metric and annotate it with operator and bounds; raises
        # on duplicate names (named storage requires unique metrics).  Bounds
        # tile object-major across evaluation objects, matching the labels.
        base_labels = labels if labels is not None else getattr(nonlincon, "labels", None)
        metric = self._build_metric(name, n_nonlinear_constraints, base_labels, eval_objs)
        bounds_total = bounds_list * max(len(eval_objs), 1)
        annotation = self._metric_space.add_constraint(
            metric, bound=bounds_total, comparison_operator=comparison_operator
        )

        # Register the constraint itself as a pipeline node.
        requires_node = [evaluator_chain[-1]] if evaluator_chain else None
        self._backend.add_evaluator(
            self._make_metric_node(
                nonlincon, n_nonlinear_constraints, args, kwargs,
                is_root=requires_node is None,
            ),
            output_name=name,
            requires=requires_node,
        )

        record = _MetricRecord(
            nonlincon,
            metric,
            annotation,
            n_per_object=n_nonlinear_constraints,
            bad_metrics=bad_metrics,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            args=args,
            kwargs=kwargs if kwargs else None,
        )
        self._nonlinear_constraints.append(record)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @property
    def callbacks(self) -> list[_CallbackRecord]:
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
            name = _derive_name(callback)

        if name in self.callback_names:
            warnings.warn("Callback with same name already exists.")
            raise CADETProcessError("Callback with same name already exists.")
        self._check_metric_name(name)

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

        record = _CallbackRecord(
            callback,
            name,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            args=args,
            kwargs=kwargs if kwargs else None,
            frequency=frequency,
            callbacks_dir=callbacks_dir,
            keep_progress=keep_progress,
        )

        # Register the callback itself as a pipeline node.  cache=False:
        # callbacks produce files, not values; repeated execution is the point.
        requires_node = [evaluator_chain[-1]] if evaluator_chain else None
        self._backend.add_evaluator(
            self._make_callback_node(record, is_root=requires_node is None),
            output_name=name,
            requires=requires_node,
            cache=False,
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
        """Total number of meta-score metrics across all eval objects."""
        return sum(ms.metric.n_metrics for ms in self._meta_scores)

    def add_meta_score(
        self,
        func: Callable,
        name: Optional[str] = None,
        n_meta_scores: int = 1,
        labels: Any = None,
        bad_metrics: Any = None,
        evaluation_objects: Any = -1,
        requires: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a meta-score function (post-objective ranking criterion).

        A meta score is a metric without direction annotation: it is
        declared on the ``MetricSpace`` like objectives and constraints,
        but carries no minimize/maximize or bound semantics.
        """
        if not callable(func):
            raise TypeError("Expected callable meta-score function.")

        if name is None:
            name = _derive_name(func)

        if name in self._metric_space.metrics_dict:
            raise CADETProcessError(
                f"Metric {name!r} is already registered. Objectives, "
                f"nonlinear constraints, and meta scores share one metric "
                f"namespace; pass name= to disambiguate."
            )
        self._check_metric_name(name)

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

        # Declare the metric without direction or constraint annotation.
        base_labels = labels if labels is not None else getattr(func, "labels", None)
        metric = self._build_metric(name, n_meta_scores, base_labels, eval_objs)
        self._metric_space.add_metric(metric)

        # Register the meta score itself as a pipeline node.
        requires_node = [evaluator_chain[-1]] if evaluator_chain else None
        self._backend.add_evaluator(
            self._make_metric_node(
                func, n_meta_scores, args, kwargs,
                is_root=requires_node is None,
            ),
            output_name=name,
            requires=requires_node,
        )

        record = _MetricRecord(
            func,
            metric,
            annotation=None,
            n_per_object=n_meta_scores,
            bad_metrics=bad_metrics,
            evaluation_objects=eval_objs,
            evaluator_chain=evaluator_chain,
            args=args,
            kwargs=kwargs if kwargs else None,
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

    def _make_metric_node(
        self,
        func: Callable,
        n_per_object: int,
        args: tuple,
        kwargs: dict,
        is_root: bool,
    ) -> Callable:
        """Build the pipeline node wrapping a metric callable.

        The node adapts the root input, bakes in fixed args/kwargs, coerces
        the result to a float array, and raises on length mismatch; the
        pipeline's failure-propagation wrapper turns that into an
        ``EvaluationFailure``.  Fallback substitution stays out of the node:
        ``bad_metrics`` is optimizer policy applied in post-processing.

        The closure captures the ``ParameterSpace``, never ``self``: node
        functions are pickled per node for parallel evaluation, and a
        reference to the problem would recurse through the pipeline.
        """
        space = self._parameter_space

        def metric_node(value: Any) -> np.ndarray:
            if is_root:
                value = _adapt_root_input(space, value)
            result = np.atleast_1d(
                np.asarray(func(value, *args, **kwargs), dtype=float)
            )
            if len(result) != n_per_object:
                raise CADETProcessError(
                    f"Expected {n_per_object} values, got {len(result)}."
                )
            return result

        return metric_node

    def _make_callback_node(self, record: _CallbackRecord, is_root: bool) -> Callable:
        """Build the pipeline node wrapping a callback callable.

        Runtime-only arguments (``individual``, ``evaluation_object``,
        ``callbacks_dir``) cannot travel through the DAG; the node reads them
        from ``record.runtime``, set by ``evaluate_callbacks`` per call.

        Captures the ``ParameterSpace`` and the record, never ``self`` (see
        ``_make_metric_node``).
        """
        space = self._parameter_space
        try:
            sig_params = set(inspect.signature(record.func).parameters)
        except (ValueError, TypeError):
            sig_params = set()

        def callback_node(value: Any) -> Any:
            if is_root:
                value = _adapt_root_input(space, value)
            kwargs = dict(record.kwargs)
            if "individual" in sig_params:
                kwargs["individual"] = record.runtime.get("individual")
            if "evaluation_object" in sig_params:
                kwargs["evaluation_object"] = record.runtime.get("evaluation_object")
            if "callbacks_dir" in sig_params:
                kwargs["callbacks_dir"] = record.runtime.get("callbacks_dir")
            return record.func(value, *record.args, **kwargs)

        return callback_node

    def _check_metric_name(self, name: str) -> None:
        """Reject names that cannot become pipeline nodes.

        Metric and evaluator names share the pipeline's output namespace.
        """
        if not name.isidentifier():
            raise CADETProcessError(
                f"Name {name!r} is not a valid Python identifier; pass name=."
            )
        if name in self._backend.output_names:
            raise CADETProcessError(
                f"Name {name!r} is already registered as a pipeline node; "
                f"metrics and evaluators share one namespace."
            )

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
            if ev_name not in self._backend._output_names:
                prev = (
                    [self._evaluator_names[req_list[i - 1]]] if i > 0 else None
                )
                func = self._evaluator_func_by_name[ev_name]
                if prev is None:
                    # Root evaluator: adapt a Mapping root input to x on
                    # free-variable problems.  Captures the space, not self.
                    def func(
                        value: Any,
                        _fn: Callable = func,
                        _space: ParameterSpace = self._parameter_space,
                    ) -> Any:
                        return _fn(_adapt_root_input(_space, value))
                self._backend.add_evaluator(
                    func,
                    output_name=ev_name,
                    requires=prev,
                )

    def _evaluate_individual(
        self,
        x: npt.ArrayLike,
        target_functions: list[_MetricRecord],
    ) -> np.ndarray:
        """Evaluate all *target_functions* for a single parameter vector.

        Thin optimizer-policy adapter over the inherited ``Problem.evaluate``:
        decodes the numeric vector, evaluates the requested metric nodes
        through the pipeline, substitutes each metric's ``bad_metrics`` for
        failed metric/object blocks, and flattens object-major.

        When evaluation fails wholesale (e.g. out-of-bounds x rejected by
        ``set_values``), all metrics return their ``bad_metrics`` fallback.
        """
        x = np.asarray(x, dtype=float).ravel()

        if not target_functions:
            return np.empty(0)

        def _bad_for(metric: _MetricRecord) -> np.ndarray:
            # Declaration-driven: total declared entries over per-object entries.
            return np.tile(metric.bad_metrics, metric.n_total_metrics // metric.n_metrics)

        names = [metric.name for metric in target_functions]
        try:
            results = self.evaluate(
                self._parameter_space.transformed_space.decode(x), targets=names
            )
        except CADETProcessError as e:
            self.logger.warning(
                "Evaluation failed at x=%s: %s. Returning bad metrics.", x, e
            )
            return np.concatenate([_bad_for(m) for m in target_functions])
        except Exception:
            self.logger.warning(
                "Unexpected error during evaluation at x=%s.", x, exc_info=True
            )
            return np.concatenate([_bad_for(m) for m in target_functions])

        rows = []
        for metric in target_functions:
            raw = results[metric.name]
            if isinstance(raw, EvaluationFailure):
                self.logger.warning(
                    "Metric '%s' failed at x=%s: %s.", metric.name, x, raw.reason
                )
                rows.append(_bad_for(metric))
            elif isinstance(raw, list):
                # Per-object passthrough: at least one object failed.
                blocks = []
                for value in raw:
                    if isinstance(value, EvaluationFailure):
                        self.logger.warning(
                            "Metric '%s' failed at x=%s: %s.",
                            metric.name, x, value.reason,
                        )
                        blocks.append(metric.bad_metrics)
                    else:
                        blocks.append(np.atleast_1d(np.asarray(value, dtype=float)))
                rows.append(np.concatenate(blocks))
            else:
                # Canonical (possibly multi-object) shape; ravel is object-major.
                rows.append(np.atleast_1d(np.asarray(raw, dtype=float)).ravel())

        return np.hstack(rows)

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
        for constraint in self._metric_space.constraints:
            factor = -1 if constraint.comparison_operator == "ge" else 1
            factors += constraint.n_metrics * [factor]

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
        return F * np.where(self._metric_space.minimize, 1.0, -1.0)

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
        eval_objs = self._parameter_space.evaluation_objects or []
        independent_names = {
            p.name for p in self._parameter_space.independent_parameters
        }
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

            cb_eval_objs = (
                cb.evaluation_objects if cb.evaluation_objects else eval_objs or [None]
            )
            for individual in population:
                assignment = {
                    name: value
                    for name, value in individual.X.items()
                    if name in independent_names
                }
                # The callback is a pipeline node (cache=False): triggering it
                # per evaluation object reuses chain results cached during
                # objective/constraint evaluation and never runs the side
                # effect for objects outside the callback's subset.
                for eval_obj in cb_eval_objs:
                    cb.runtime = {
                        "individual": individual,
                        "evaluation_object": eval_obj,
                        "callbacks_dir": _cb_dir,
                    }
                    try:
                        result = self._backend.evaluate(
                            assignment,
                            targets=[cb.name],
                            evaluation_objects=(
                                None if eval_obj is None else [eval_obj]
                            ),
                        )
                        value = result[cb.name]
                        if isinstance(value, EvaluationFailure):
                            _logger.warning(
                                f"Callback {cb.name!r} failed at iteration"
                                f" {current_iteration}: {value.reason}"
                            )
                    except Exception as exc:
                        _logger.warning(
                            f"Callback {cb.name!r} failed at iteration"
                            f" {current_iteration}: {exc}",
                            exc_info=True,
                        )
                    finally:
                        cb.runtime = {}

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

    # ── Population creation ───────────────────────────────────────────────────

    def _metric_columns(
        self,
        records: list[_MetricRecord],
        values: npt.ArrayLike,
        n: int,
        kind: str,
    ) -> dict[str, np.ndarray]:
        """Split a flat value matrix into named metric columns.

        *values* has one row per individual; each record consumes
        ``metric.n_metrics`` columns and is reshaped to the metric's
        declared shape.
        """
        values = np.asarray(values, dtype=float).reshape(n, -1)
        n_total = sum(record.metric.n_metrics for record in records)
        if values.shape[1] != n_total:
            raise CADETProcessError(
                f"Expected {n_total} {kind} values per individual, "
                f"got {values.shape[1]}."
            )
        columns: dict[str, np.ndarray] = {}
        offset = 0
        for record in records:
            metric = record.metric
            block = values[:, offset:offset + metric.n_metrics]
            columns[metric.name] = block.reshape(n, *metric.shape)
            offset += metric.n_metrics
        return columns

    def create_population(
        self,
        X: npt.ArrayLike,
        F: npt.ArrayLike | None = None,
        G: npt.ArrayLike | None = None,
        M: npt.ArrayLike | None = None,
        untransform: bool = False,
        get_dependent_values: bool = False,
    ) -> Population:
        """Create a columnar ``Population`` from parameter and metric arrays.

        Parameters
        ----------
        X : array-like
            Parameter vectors, one row per individual.  Full vectors by
            default; pass ``get_dependent_values=True`` for
            independent-only rows and ``untransform=True`` for normalized
            coordinates.
        F, G, M : array-like, optional
            Objective, nonlinear-constraint, and meta-score values in
            physical direction, one row per individual, flattened in
            registration order.
        """
        X = np.array(X, ndmin=2)
        if untransform:
            X = np.array([self.untransform(x) for x in X])
        if get_dependent_values:
            X = np.array([self._resolve_full_vector(x) for x in X])

        n = len(X)
        if X.shape[1] != self.n_variables:
            raise CADETProcessError(
                f"Expected {self.n_variables} parameter values per individual, "
                f"got {X.shape[1]}."
            )

        X_columns = {
            name: X[:, i] for i, name in enumerate(self.variable_names)
        }

        metrics: dict[str, np.ndarray] = {}
        if F is not None:
            metrics.update(self._metric_columns(self._objectives, F, n, "objective"))
        if G is not None:
            metrics.update(
                self._metric_columns(
                    self._nonlinear_constraints, G, n, "nonlinear constraint"
                )
            )
        if M is not None:
            metrics.update(
                self._metric_columns(self._meta_scores, M, n, "meta score")
            )

        return Population(
            X=X_columns,
            metrics=metrics,
            metric_space=self._metric_space,
            parameter_space=self._parameter_space,
        )

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
