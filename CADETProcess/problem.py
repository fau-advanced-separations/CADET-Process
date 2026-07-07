"""Problem: parameter space, metric space, and evaluation backend.

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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Protocol, runtime_checkable

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

    def evaluate(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        """Compute named outputs for a named parameter assignment."""
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

    def evaluate(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate the declared metrics for a named parameter assignment.

        Delegates to the backend, then validates each declared metric
        against its declared shape.  Backend outputs that are not declared
        in the metric space (pipeline intermediates) are dropped.
        ``EvaluationFailure`` values pass through unvalidated; substituting
        fallback values is optimizer policy and stays out of ``Problem``.

        Returns
        -------
        dict[str, Any]
            Declared metrics in registration order; values are arrays in
            canonical shape or ``EvaluationFailure``.

        Raises
        ------
        RuntimeError
            If no backend is set.
        ValueError
            If the backend result misses a declared metric or a value does
            not match its declared shape.
        """
        if self._backend is None:
            raise RuntimeError(
                "Problem has no evaluation backend.  Construct with backend=... "
                "or use with_evaluator."
            )
        raw = self._backend.evaluate(assignment)
        results: dict[str, Any] = {}
        for metric in self._metric_space.metrics:
            if metric.name not in raw:
                raise ValueError(
                    f"Backend result is missing declared metric {metric.name!r}."
                )
            value = raw[metric.name]
            if isinstance(value, EvaluationFailure):
                results[metric.name] = value
            else:
                results[metric.name] = metric.validate(value)
        return results

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
