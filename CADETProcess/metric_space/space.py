"""MetricSpace: output declarations and their problem-level annotations.

``MetricSpace`` collects ``Metric`` declarations and owns the annotations
that give them meaning in a problem context: optimization direction
(``Objective``), operator/bound constraints (``Constraint``), and output
normalization.  Direction and normalization are properties of the
registration, not of the metric itself: the same metric can be an
objective in one problem and a plain output in another.  A metric
without a direction annotation is just an output.

Only nonlinear constraints live here; linear constraints on parameters
stay in ``ParameterSpace``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from CADETProcess.metric_space.metric import Metric
from CADETProcess.parameter_space.normalize import NormalizerBase

__all__ = ["MetricSpace", "Objective", "Constraint"]


class Objective:
    """Direction annotation on a metric.

    Parameters
    ----------
    metric : Metric
        The annotated metric.
    minimize : bool
        Optimization direction; True minimizes, False maximizes.
    """

    def __init__(self, metric: Metric, minimize: bool = True) -> None:
        self.metric = metric
        self.minimize = bool(minimize)

    @property
    def name(self) -> str:
        """Name of the annotated metric."""
        return self.metric.name

    @property
    def labels(self) -> list[str]:
        """Labels of the annotated metric, one per scalar entry."""
        return self.metric.labels

    @property
    def n_metrics(self) -> int:
        """Number of scalar entries of the annotated metric."""
        return self.metric.n_metrics

    def __repr__(self) -> str:
        """Return a readable representation."""
        return f"Objective(metric={self.metric!r}, minimize={self.minimize})"


class Constraint:
    """Operator/bound annotation on a metric; canonical form is ``g <= 0``.

    Parameters
    ----------
    metric : Metric
        The annotated metric.
    bound : float or array-like
        Constraint bound.  A scalar applies to every entry; an array must
        match the metric's ``n_metrics``.
    comparison_operator : {"le", "ge"}
        Direction of the comparison: ``value <= bound`` or ``value >= bound``.
    """

    def __init__(
        self,
        metric: Metric,
        bound: float | npt.ArrayLike = 0.0,
        comparison_operator: Literal["le", "ge"] = "le",
    ) -> None:
        if comparison_operator not in ("le", "ge"):
            raise ValueError(
                f"Constraint on {metric.name!r}: comparison_operator must be "
                f"'le' or 'ge', got {comparison_operator!r}."
            )
        self.metric = metric
        self.comparison_operator = comparison_operator
        try:
            self.bounds = np.broadcast_to(
                np.asarray(bound, dtype=float), (metric.n_metrics,)
            ).copy()
        except ValueError:
            raise ValueError(
                f"Constraint on {metric.name!r}: expected scalar bound or "
                f"{metric.n_metrics} bounds, got {np.shape(bound)}."
            ) from None

    @property
    def name(self) -> str:
        """Name of the annotated metric."""
        return self.metric.name

    @property
    def labels(self) -> list[str]:
        """Labels of the annotated metric, one per scalar entry."""
        return self.metric.labels

    @property
    def n_metrics(self) -> int:
        """Number of scalar entries of the annotated metric."""
        return self.metric.n_metrics

    def violation(self, values: Any) -> np.ndarray:
        """Signed violation in canonical ``g <= 0`` form.

        Positive entries violate the constraint.  Returns a 1-D array of
        length ``n_metrics``.
        """
        arr = self.metric.validate(values).reshape(-1)
        if self.comparison_operator == "le":
            return arr - self.bounds
        return self.bounds - arr

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"Constraint(metric={self.metric!r}, bounds={self.bounds!r}, "
            f"comparison_operator={self.comparison_operator!r})"
        )


class MetricSpace:
    """Collection of metric declarations with problem-level annotations.

    Parallel to ``ParameterSpace`` on the output side: owns direction,
    normalization, and constraint annotations.  Purely declarative; it
    performs no evaluation and holds no reference to an evaluation backend.

    Typical usage::

        metric_space = MetricSpace()
        metric_space.add_objective(Metric("yield", n_metrics=2), minimize=False)
        metric_space.add_constraint("purity", bound=0.95, comparison_operator="ge")
    """

    def __init__(self) -> None:
        self._metrics: list[Metric] = []
        self._objectives: list[Objective] = []
        self._constraints: list[Constraint] = []
        self._normalizers: dict[str, NormalizerBase] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def add_metric(
        self,
        metric: Metric | str,
        normalizer: Optional[NormalizerBase] = None,
    ) -> Metric:
        """Register a metric declaration without annotations.

        Parameters
        ----------
        metric : Metric or str
            The declaration to register.  A string is shorthand for a scalar
            ``Metric(name)``.
        normalizer : NormalizerBase, optional
            Output normalizer for this metric.

        Returns
        -------
        Metric
            The registered metric.

        Raises
        ------
        ValueError
            If a metric with the same name is already registered.
        """
        if isinstance(metric, str):
            metric = Metric(metric)
        if metric.name in self.metrics_dict:
            raise ValueError(f"Metric {metric.name!r} is already registered.")
        self._metrics.append(metric)
        if normalizer is not None:
            self._normalizers[metric.name] = normalizer
        return metric

    def _resolve_metric(self, metric: Metric | str) -> Metric:
        """Return the registered metric, registering new instances on the fly."""
        if isinstance(metric, str):
            try:
                return self.metrics_dict[metric]
            except KeyError:
                raise ValueError(f"Unknown metric {metric!r}.") from None
        registered = self.metrics_dict.get(metric.name)
        if registered is None:
            return self.add_metric(metric)
        if registered is not metric:
            raise ValueError(
                f"A different metric named {metric.name!r} is already registered."
            )
        return registered

    def add_objective(
        self,
        metric: Metric | str,
        minimize: bool = True,
    ) -> Objective:
        """Annotate a metric with an optimization direction.

        Parameters
        ----------
        metric : Metric or str
            A new metric (registered on the fly) or the name of a registered
            one.
        minimize : bool
            Optimization direction; True minimizes, False maximizes.

        Returns
        -------
        Objective
            The direction annotation.

        Raises
        ------
        ValueError
            If the metric already has a direction annotation.
        """
        resolved = self._resolve_metric(metric)
        if any(objective.metric is resolved for objective in self._objectives):
            raise ValueError(
                f"Metric {resolved.name!r} already has a direction annotation."
            )
        objective = Objective(resolved, minimize=minimize)
        self._objectives.append(objective)
        return objective

    def add_constraint(
        self,
        metric: Metric | str,
        bound: float | npt.ArrayLike = 0.0,
        comparison_operator: Literal["le", "ge"] = "le",
    ) -> Constraint:
        """Annotate a metric with an operator/bound constraint.

        Parameters
        ----------
        metric : Metric or str
            A new metric (registered on the fly) or the name of a registered
            one.
        bound : float or array-like
            Constraint bound; scalar or one per metric entry.
        comparison_operator : {"le", "ge"}
            Direction of the comparison.

        Returns
        -------
        Constraint
            The constraint annotation.
        """
        resolved = self._resolve_metric(metric)
        constraint = Constraint(
            resolved, bound=bound, comparison_operator=comparison_operator
        )
        self._constraints.append(constraint)
        return constraint

    def set_normalizer(self, metric: Metric | str, normalizer: NormalizerBase) -> None:
        """Attach an output normalizer to a registered metric."""
        resolved = self._resolve_metric(metric)
        self._normalizers[resolved.name] = normalizer

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> list[Metric]:
        """All registered metrics."""
        return list(self._metrics)

    @property
    def metrics_dict(self) -> dict[str, Metric]:
        """Registered metrics keyed by name."""
        return {metric.name: metric for metric in self._metrics}

    @property
    def metric_names(self) -> list[str]:
        """Names of all registered metrics."""
        return [metric.name for metric in self._metrics]

    @property
    def n_metrics(self) -> int:
        """Total number of scalar entries across all registered metrics."""
        return sum(metric.n_metrics for metric in self._metrics)

    @property
    def labels(self) -> list[str]:
        """Flattened labels across all registered metrics."""
        return [label for metric in self._metrics for label in metric.labels]

    # ── Objectives ────────────────────────────────────────────────────────────

    @property
    def objectives(self) -> list[Objective]:
        """All direction annotations."""
        return list(self._objectives)

    @property
    def objective_names(self) -> list[str]:
        """Names of all metrics with a direction annotation."""
        return [objective.name for objective in self._objectives]

    @property
    def objective_labels(self) -> list[str]:
        """Flattened labels across all objectives."""
        return [
            label for objective in self._objectives for label in objective.labels
        ]

    @property
    def n_objectives(self) -> int:
        """Total number of scalar objective entries."""
        return sum(objective.n_metrics for objective in self._objectives)

    @property
    def minimize(self) -> np.ndarray:
        """Direction per scalar objective entry; True minimizes."""
        return np.array(
            [
                objective.minimize
                for objective in self._objectives
                for _ in range(objective.n_metrics)
            ],
            dtype=bool,
        )

    # ── Constraints ───────────────────────────────────────────────────────────

    @property
    def constraints(self) -> list[Constraint]:
        """All constraint annotations."""
        return list(self._constraints)

    @property
    def constraint_names(self) -> list[str]:
        """Names of all metrics with a constraint annotation."""
        return [constraint.name for constraint in self._constraints]

    @property
    def constraint_labels(self) -> list[str]:
        """Flattened labels across all constraints."""
        return [
            label for constraint in self._constraints for label in constraint.labels
        ]

    @property
    def n_constraints(self) -> int:
        """Total number of scalar constraint entries."""
        return sum(constraint.n_metrics for constraint in self._constraints)

    @property
    def constraints_bounds(self) -> np.ndarray:
        """Flattened bounds across all constraints."""
        if not self._constraints:
            return np.empty(0)
        return np.concatenate(
            [constraint.bounds for constraint in self._constraints]
        )

    # ── Validation and normalization ──────────────────────────────────────────

    def validate(self, results: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate evaluation results against the declared metrics.

        Every registered metric must be present with its declared shape.
        Extra keys (pipeline intermediates) are ignored.

        Returns
        -------
        dict[str, np.ndarray]
            Declared metrics in canonical shape, in registration order.

        Raises
        ------
        ValueError
            If a declared metric is missing or has the wrong shape.
        """
        missing = [name for name in self.metric_names if name not in results]
        if missing:
            raise ValueError(f"Missing metrics in results: {missing}.")
        return {
            metric.name: metric.validate(results[metric.name])
            for metric in self._metrics
        }

    def normalize(self, results: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate results and map metric values to normalized space.

        Metrics without a registered normalizer pass through unchanged.
        """
        validated = self.validate(results)
        return {
            name: (
                values
                if (normalizer := self._normalizers.get(name)) is None
                else np.asarray(normalizer.normalize(values))
            )
            for name, values in validated.items()
        }

    def denormalize(self, results: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate results and map metric values back to physical space.

        Metrics without a registered normalizer pass through unchanged.
        """
        validated = self.validate(results)
        return {
            name: (
                values
                if (normalizer := self._normalizers.get(name)) is None
                else np.asarray(normalizer.denormalize(values))
            )
            for name, values in validated.items()
        }

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"MetricSpace(n_metrics={self.n_metrics}, "
            f"n_objectives={self.n_objectives}, "
            f"n_constraints={self.n_constraints})"
        )
