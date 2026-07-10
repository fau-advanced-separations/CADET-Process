"""
=========================================
Problem (:mod:`CADETProcess.problem`)
=========================================

.. currentmodule:: CADETProcess.problem

``Problem`` is the general problem description, independent of any
optimizer: a ``ParameterSpace`` (input domain), a ``MetricSpace`` (output
declarations), and an ``EvaluationBackend`` that computes the declared
metrics for a named parameter assignment.  It is directly usable for
sensitivity analysis, design-space exploration, surrogate training, and
emulation; ``OptimizationProblem`` extends it with objective/constraint
policy.

Discipline rule: if a sampler or surrogate would not call it, it does not
go on ``Problem``.  Direction-aware evaluation, ``bad_metrics``
substitution, and callbacks belong to ``OptimizationProblem``.

.. autosummary::
    :toctree: generated/

    Problem
    EvaluationBackend

"""  # noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from CADETProcess.evaluation_pipeline import EvaluationFailure
from CADETProcess.metric_space import MetricSpace
from CADETProcess.parameter_space import ParameterSpace

__all__ = ["EvaluationBackend", "Problem"]


@runtime_checkable
class EvaluationBackend(Protocol):
    """Protocol for anything that evaluates a named parameter assignment.

    Satisfied by both ``EvaluationPipeline`` and surrogate models.  The
    returned mapping must contain every metric declared in the problem's
    ``MetricSpace``; additional keys (pipeline intermediates) are allowed.
    """

    def evaluate(
        self,
        assignment: Mapping[str, Any],
        targets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute named outputs for a named parameter assignment.

        Backends may ignore *targets* and compute everything;
        ``Problem.evaluate`` filters the result either way.
        """
        ...


class Problem:
    """General problem description: parameter space, metric space, backend.

    Concrete and directly usable; samplers and surrogates consume it
    without any optimizer policy.

    Parameters
    ----------
    parameter_space : ParameterSpace, optional
        Input domain.  A fresh empty space is created when omitted.
    metric_space : MetricSpace, optional
        Output declarations.  A fresh empty space is created when omitted.
    backend : EvaluationBackend, optional
        Computes the declared metrics for a named assignment.  ``evaluate``
        raises until a backend is set.
    name : str, optional
        Problem name; used in the string representation.

    Examples
    --------
    >>> problem = Problem(parameter_space, metric_space, backend=pipeline)
    >>> results = problem.evaluate({"length": 0.5})
    >>> surrogate_problem = problem.with_evaluator(surrogate)
    """

    def __init__(
        self,
        parameter_space: Optional[ParameterSpace] = None,
        metric_space: Optional[MetricSpace] = None,
        backend: Optional[EvaluationBackend] = None,
        name: Optional[str] = None,
    ) -> None:
        if parameter_space is None:
            parameter_space = ParameterSpace()
        elif not isinstance(parameter_space, ParameterSpace):
            raise TypeError(
                f"Expected ParameterSpace, got {type(parameter_space).__name__}."
            )
        if metric_space is None:
            metric_space = MetricSpace()
        elif not isinstance(metric_space, MetricSpace):
            raise TypeError(
                f"Expected MetricSpace, got {type(metric_space).__name__}."
            )
        if backend is not None and not isinstance(backend, EvaluationBackend):
            raise TypeError(
                f"Backend {type(backend).__name__} does not satisfy "
                f"EvaluationBackend: it has no evaluate method."
            )
        self._parameter_space = parameter_space
        self._metric_space = metric_space
        self._backend = backend
        self.name = name

    @property
    def parameter_space(self) -> ParameterSpace:
        """Input domain."""
        return self._parameter_space

    @property
    def metric_space(self) -> MetricSpace:
        """Output declarations and annotations."""
        return self._metric_space

    @property
    def backend(self) -> Optional[EvaluationBackend]:
        """Active evaluation backend; None until one is set."""
        return self._backend

    def with_evaluator(self, backend: EvaluationBackend) -> "Problem":
        """Return a new ``Problem`` with *backend* as evaluation backend.

        Non-mutating: the original problem keeps its backend and both share
        the same parameter and metric spaces.  Always returns a plain
        ``Problem``, never a subclass: a surrogate-backed problem carries no
        optimizer policy.
        """
        return Problem(
            parameter_space=self._parameter_space,
            metric_space=self._metric_space,
            backend=backend,
            name=self.name,
        )

    def evaluate(
        self,
        assignment: Mapping[str, Any],
        targets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the declared metrics for a named parameter assignment.

        Delegates to the backend, then validates each declared metric
        against its declared shape.  Backend outputs that are not declared
        in the metric space (pipeline intermediates) are dropped; declared
        metric names are always passed to the backend as its targets, so
        undeclared side-effect nodes (callbacks) never execute.
        ``EvaluationFailure`` values pass through unvalidated; substituting
        fallback values is optimizer policy and stays out of ``Problem``.

        A backend evaluating multiple evaluation objects returns per-object
        lists (the ``EvaluationPipeline`` convention).  Such values are
        reduced to the metric's declared shape here: the entries belonging
        to the metric's declared ``evaluation_object`` coordinates are
        selected and stacked object-major.  If any selected entry is an
        ``EvaluationFailure``, the selected per-object list passes through
        unvalidated instead.

        Parameters
        ----------
        assignment : Mapping
            Values for the independent parameters by name, in physical units.
        targets : list[str], optional
            Declared metric names to evaluate.  `None` evaluates all
            declared metrics.  A pipeline backend only executes the
            subgraph the requested metrics need.

        Returns
        -------
        dict[str, Any]
            Requested metrics in registration order; values are arrays in
            canonical shape, ``EvaluationFailure``, or a per-object list
            containing at least one ``EvaluationFailure``.

        Raises
        ------
        RuntimeError
            If no backend is set.
        ValueError
            If *targets* contains an undeclared name, the backend result
            misses a declared metric, or a value does not match its
            declared shape.
        """
        if self._backend is None:
            raise RuntimeError(
                "Problem has no evaluation backend.  Construct with backend=... "
                "or use with_evaluator."
            )
        metrics = self._metric_space.metrics
        if targets is not None:
            declared = {m.name for m in metrics}
            unknown = [t for t in targets if t not in declared]
            if unknown:
                raise ValueError(f"Unknown metric target(s): {unknown}")
            requested = set(targets)
            metrics = [m for m in metrics if m.name in requested]
        if not metrics:
            return {}
        raw = self._backend.evaluate(assignment, targets=[m.name for m in metrics])
        results: dict[str, Any] = {}
        for metric in metrics:
            if metric.name not in raw:
                raise ValueError(
                    f"Backend result is missing declared metric {metric.name!r}."
                )
            value = raw[metric.name]
            has_object_dim = (
                metric.coords is not None and "evaluation_object" in metric.coords
            )
            if isinstance(value, EvaluationFailure):
                results[metric.name] = value
            elif has_object_dim and isinstance(value, list):
                results[metric.name] = self._reduce_per_object(metric, value)
            elif (
                isinstance(value, list)
                and any(isinstance(v, EvaluationFailure) for v in value)
            ):
                # Per-object failures for a metric that declares no
                # evaluation_object dimension: the backend fanned out over
                # objects the metric does not know about.
                raise ValueError(
                    f"Metric {metric.name!r} declares no evaluation_object "
                    f"dimension but the backend returned per-object results.  "
                    f"This happens for metrics registered with "
                    f"evaluation_objects=None while evaluation objects exist; "
                    f"see the input-handling cleanup note in PROJECT.md."
                )
            else:
                results[metric.name] = metric.validate(value)
        return results

    def evaluate_batch(
        self,
        assignments: Sequence[Mapping[str, Any]],
        targets: list[str] | None = None,
        parallelization_backend: Any = None,
    ) -> list[dict[str, Any]]:
        """Evaluate the declared metrics for a batch of named assignments.

        Uniform batch entry point: optimizers, samplers, and surrogate
        trainers all dispatch populations through this method.  Each
        assignment is evaluated via ``evaluate``; a row whose evaluation
        raises yields an ``EvaluationFailure`` for every requested target
        instead of aborting the batch.  Substituting fallback values for
        failures remains caller policy.

        Parameters
        ----------
        assignments : Sequence[Mapping]
            One named parameter assignment per row, in physical units.
        targets : list[str], optional
            Declared metric names to evaluate.  `None` evaluates all
            declared metrics.
        parallelization_backend : ParallelizationBackendBase, optional
            When provided, rows are dispatched via ``backend.evaluate``.
            When None, evaluation is sequential.

        Returns
        -------
        list[dict[str, Any]]
            One result mapping per assignment, in input order.

        Raises
        ------
        RuntimeError
            If no backend is set.
        ValueError
            If *targets* contains an undeclared name.
        """
        if self._backend is None:
            raise RuntimeError(
                "Problem has no evaluation backend.  Construct with backend=... "
                "or use with_evaluator."
            )
        declared = [m.name for m in self._metric_space.metrics]
        if targets is not None:
            unknown = [t for t in targets if t not in declared]
            if unknown:
                raise ValueError(f"Unknown metric target(s): {unknown}")
        names = targets if targets is not None else declared

        def _evaluate_one(assignment: Mapping[str, Any]) -> dict[str, Any]:
            try:
                return self.evaluate(assignment, targets=targets)
            except Exception as e:
                return {
                    name: EvaluationFailure(stage=name, reason=str(e), exc=e)
                    for name in names
                }

        assignments = list(assignments)
        if parallelization_backend is None:
            return [_evaluate_one(assignment) for assignment in assignments]
        return list(parallelization_backend.evaluate(_evaluate_one, assignments))

    def _reduce_per_object(self, metric: Any, values: list[Any]) -> Any:
        """Reduce a per-object result list to the metric's declared shape.

        The list is indexed by the parameter space's registered evaluation
        objects; the metric's ``evaluation_object`` coordinates select the
        entries it covers.  Failures keep per-object granularity so the
        optimizer layer can substitute fallbacks per object.
        """
        coords = metric.coords["evaluation_object"]
        obj_names = [str(obj) for obj in self._parameter_space.evaluation_objects]
        missing = [c for c in coords if c not in obj_names]
        if missing:
            raise ValueError(
                f"Metric {metric.name!r}: declared evaluation object(s) "
                f"{missing} are not registered."
            )
        selected = [values[obj_names.index(c)] for c in coords]
        if any(isinstance(v, EvaluationFailure) for v in selected):
            return selected
        flat = np.concatenate([
            np.atleast_1d(np.asarray(v, dtype=float)) for v in selected
        ])
        return metric.validate(flat.reshape(metric.shape))

    def __str__(self) -> str:
        """Return the problem name, falling back to the class name."""
        return self.name if self.name is not None else type(self).__name__

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"n_parameters={self._parameter_space.n_parameters}, "
            f"n_metrics={self._metric_space.n_metrics}, "
            f"backend={type(self._backend).__name__ if self._backend else None})"
        )
