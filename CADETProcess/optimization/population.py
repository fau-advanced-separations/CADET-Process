"""Columnar named storage for evaluated parameter points.

``Population`` is a universal, optimizer-agnostic record of evaluated
points: a scipy trace, an Ax experiment history, a GA generation, and a
surrogate training corpus are all the same type.  Parameters and metrics
are stored as named columns (``dict[str, np.ndarray]``); everything else
(objective matrices, constraint violations, feasibility, dominance) is
derived from the columns plus the annotations in ``MetricSpace`` and
``ParameterSpace``.

A single evaluated point is a one-row ``Population``; indexing with a
scalar returns an ``IndividualView``, a lightweight row lens that owns no
data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from addict import Dict

from CADETProcess import CADETProcessError, plotting
from CADETProcess.metric_space import Metric, MetricSpace
from CADETProcess.parameter_space import ParameterSpace

__all__ = ["IndividualView", "Population", "ParetoFront"]

_METADATA_KEYS = ("timestamp", "evaluation_time")


def _as_column(values: npt.ArrayLike) -> np.ndarray:
    """Coerce a parameter column to a 1-D array, numeric when possible."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            pass
    return arr


def _decode(value: Any) -> Any:
    """Decode bytes (from HDF5 round-trips) to str, recursively for arrays."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    return value


def _hash_row(row: npt.ArrayLike) -> str:
    """Deterministic sha256 hex digest of one parameter row's values.

    Numeric rows hash the raw float64 bytes; object-dtype rows (categorical
    parameters) hash a stable text encoding instead, since ``ndarray.tobytes``
    on an object array serializes pointers, not values.
    """
    row = np.asarray(row)
    if row.dtype.kind in "fiub":
        payload = row.astype(np.float64).tobytes()
    else:
        payload = repr(row.tolist()).encode()
    return hashlib.sha256(payload).hexdigest()


class IndividualView:
    """Row lens over canonical ``Population`` storage.

    Deliberately minimal: convenience indexing only.  It owns no data and
    implements no optimization semantics; anything that vectorizes
    (dominance, feasibility, statistics) stays on ``Population``.
    """

    __slots__ = ("_population", "_idx")

    def __init__(self, population: "Population", idx: int) -> None:
        self._population = population
        self._idx = int(idx)

    @property
    def X(self) -> dict[str, Any]:
        """Named parameter values of this row."""
        return {
            name: column[self._idx]
            for name, column in self._population.X.items()
        }

    @property
    def metrics(self) -> dict[str, np.ndarray]:
        """Named metric values of this row, in declared shape."""
        return {
            name: values[self._idx]
            for name, values in self._population.metrics.items()
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata values of this row; empty when the population has none."""
        metadata = self._population.metadata
        if metadata is None:
            return {}
        return {name: values[self._idx] for name, values in metadata.items()}

    @property
    def id(self) -> str:
        """Content-derived id: sha256 digest of this row's parameter values.

        Identical parameter values always produce the same id, including
        across repeated evaluations in different generations, which is what
        lets callback output files be traced back to a row in the results
        table.
        """
        return _hash_row(self._population.x[self._idx])

    @property
    def id_short(self) -> str:
        """First seven characters of :attr:`id`, for filenames and display."""
        return self.id[0:7]

    def as_record(self) -> dict[str, Any]:
        """Return this row as a ``{"X": ..., "metrics": ..., "metadata": ...}`` record."""
        record: dict[str, Any] = {"X": self.X, "metrics": self.metrics}
        metadata = self.metadata
        if metadata:
            record["metadata"] = metadata
        return record

    def as_population(self) -> "Population":
        """Return this row as a one-row ``Population``."""
        return self._population[[self._idx]]

    def __repr__(self) -> str:
        """Return a readable representation."""
        return f"IndividualView(X={self.X!r})"


class Population:
    """Columnar record of evaluated parameter points.

    Parameters
    ----------
    X : Mapping[str, array-like]
        Named parameter columns, one ``(n,)`` array per parameter.
        Categorical parameters are held as string/object columns.
    metrics : Mapping[str, array-like], optional
        Named metric columns; each value has shape ``(n, *metric_shape)``
        with the shape declared on the ``Metric`` in *metric_space*.
    metadata : Mapping[str, array-like], optional
        Universal bookkeeping only: ``timestamp`` and ``evaluation_time``.
        Optimizer context (generation, ranks) belongs in
        ``OptimizationResults``.
    metric_space : MetricSpace
        Output declarations and annotations; required.  Dominance,
        constraint violations, and shape validation read from it.
    parameter_space : ParameterSpace, optional
        Input domain.  Semantically required; ``None`` is an
        interoperability escape hatch for raw arrays.  Operations that
        need it (``X_num``, bound/linear-constraint violations) raise
        when it is absent.

    Notes
    -----
    Population is immutable: combination is explicit via
    ``Population.concat``.  Deduplication uses exact value matching on the
    stored parameter columns; there is no row identity beyond the values.
    """

    def __init__(
        self,
        X: Mapping[str, npt.ArrayLike],
        metrics: Optional[Mapping[str, npt.ArrayLike]] = None,
        metadata: Optional[Mapping[str, npt.ArrayLike]] = None,
        *,
        metric_space: MetricSpace,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> None:
        if not isinstance(metric_space, MetricSpace):
            raise TypeError(
                f"metric_space is required; got {type(metric_space).__name__}."
            )
        if parameter_space is not None and not isinstance(
            parameter_space, ParameterSpace
        ):
            raise TypeError(
                f"Expected ParameterSpace, got {type(parameter_space).__name__}."
            )
        self._metric_space = metric_space
        self._parameter_space = parameter_space
        self._init_storage(X, metrics, metadata)

    def _init_storage(
        self,
        X: Mapping[str, npt.ArrayLike],
        metrics: Optional[Mapping[str, npt.ArrayLike]],
        metadata: Optional[Mapping[str, npt.ArrayLike]],
    ) -> None:
        """Validate and freeze the columnar storage."""
        X_cols: dict[str, np.ndarray] = {}
        n: Optional[int] = None
        for name, values in dict(X).items():
            col = _as_column(values)
            if n is None:
                n = len(col)
            elif len(col) != n:
                raise ValueError(
                    f"Parameter column {name!r} has length {len(col)}, "
                    f"expected {n}."
                )
            col.flags.writeable = False
            X_cols[str(name)] = col

        declared = self._metric_space.metrics_dict
        metric_cols: dict[str, np.ndarray] = {}
        for name, values in dict(metrics or {}).items():
            if name not in declared:
                raise ValueError(f"Metric {name!r} is not declared in metric_space.")
            metric = declared[name]
            arr = np.asarray(values, dtype=float)
            if n is None:
                n = arr.shape[0] if arr.ndim > 0 else 0
            expected = (n, *metric.shape)
            if arr.shape != expected:
                # A scalar metric may arrive as (n, 1); canonicalize.
                if metric.shape == () and arr.shape == (n, 1):
                    arr = arr.reshape(n)
                else:
                    raise ValueError(
                        f"Metric {name!r}: expected shape {expected}, "
                        f"got {arr.shape}."
                    )
            arr.flags.writeable = False
            metric_cols[name] = arr

        meta_cols: Optional[dict[str, np.ndarray]] = None
        if metadata:
            meta_cols = {}
            for name, values in dict(metadata).items():
                if name not in _METADATA_KEYS:
                    raise ValueError(
                        f"Unknown metadata key {name!r}; allowed: {_METADATA_KEYS}. "
                        "Optimizer context belongs in OptimizationResults."
                    )
                arr = np.asarray(values)
                if n is None:
                    n = len(arr)
                if arr.shape != (n,):
                    raise ValueError(
                        f"Metadata {name!r}: expected shape ({n},), got {arr.shape}."
                    )
                arr.flags.writeable = False
                meta_cols[name] = arr

        self._X = X_cols
        self._metrics = metric_cols
        self._metadata = meta_cols
        self._n = n if n is not None else 0

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def empty(
        cls,
        *,
        metric_space: MetricSpace,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> "Population":
        """Return an empty population (concat/accumulation seed)."""
        return cls(
            X={}, metric_space=metric_space, parameter_space=parameter_space
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        metric_space: MetricSpace,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> "Population":
        """Build a population from per-row records.

        Each record is a mapping with keys ``"X"`` (required),
        ``"metrics"``, and ``"metadata"``; all records must provide the
        same keys and the same column names.
        """
        records = list(records)
        if not records:
            return cls.empty(
                metric_space=metric_space, parameter_space=parameter_space
            )

        x_names = list(records[0].get("X", {}))
        metric_names = list(records[0].get("metrics", {}) or {})
        metadata_names = list(records[0].get("metadata", {}) or {})

        X = {name: [] for name in x_names}
        metrics: dict[str, list] = {name: [] for name in metric_names}
        metadata: dict[str, list] = {name: [] for name in metadata_names}
        declared = metric_space.metrics_dict
        for record in records:
            record_X = record.get("X", {})
            if list(record_X) != x_names:
                raise ValueError("All records must share the same parameter names.")
            for name in x_names:
                X[name].append(record_X[name])
            record_metrics = record.get("metrics", {}) or {}
            if list(record_metrics) != metric_names:
                raise ValueError("All records must share the same metric names.")
            for name in metric_names:
                if name not in declared:
                    raise ValueError(
                        f"Metric {name!r} is not declared in metric_space."
                    )
                metrics[name].append(declared[name].validate(record_metrics[name]))
            record_metadata = record.get("metadata", {}) or {}
            for name in metadata_names:
                metadata[name].append(record_metadata[name])

        return cls(
            X=X,
            metrics={name: np.stack(vals) for name, vals in metrics.items()},
            metadata={name: np.asarray(vals) for name, vals in metadata.items()}
            if metadata
            else None,
            metric_space=metric_space,
            parameter_space=parameter_space,
        )

    @classmethod
    def from_sample(
        cls,
        X: Mapping[str, Any],
        metrics: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        metric_space: MetricSpace,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> "Population":
        """Build a one-row population from a single named sample."""
        record: dict[str, Any] = {"X": dict(X)}
        if metrics is not None:
            record["metrics"] = dict(metrics)
        if metadata is not None:
            record["metadata"] = dict(metadata)
        return cls.from_records(
            [record], metric_space=metric_space, parameter_space=parameter_space
        )

    @classmethod
    def concat(cls, populations: Iterable["Population"]) -> "Population":
        """Concatenate populations into a new one.

        All non-empty populations must share parameter columns, metric
        columns, and a compatible metric space (same metric names and
        shapes).  Empty populations are ignored.
        """
        populations = list(populations)
        if not populations:
            raise ValueError("Expected at least one population.")
        non_empty = [pop for pop in populations if len(pop) > 0]
        if not non_empty:
            return populations[0]

        first = non_empty[0]
        names_first = first._metric_space.metric_names
        for pop in non_empty[1:]:
            if list(pop._X) != list(first._X):
                raise CADETProcessError("Parameter columns do not match.")
            if list(pop._metrics) != list(first._metrics):
                raise CADETProcessError("Metric columns do not match.")
            if pop._metric_space.metric_names != names_first:
                raise CADETProcessError("Metric spaces are not compatible.")

        X = {
            name: np.concatenate([pop._X[name] for pop in non_empty])
            for name in first._X
        }
        metrics = {
            name: np.concatenate([pop._metrics[name] for pop in non_empty])
            for name in first._metrics
        }
        has_metadata = [pop._metadata for pop in non_empty]
        if all(md is not None for md in has_metadata):
            keys = list(has_metadata[0])
            metadata = {
                key: np.concatenate([md[key] for md in has_metadata])
                for key in keys
                if all(key in md for md in has_metadata)
            }
        else:
            metadata = None

        return cls(
            X=X,
            metrics=metrics,
            metadata=metadata,
            metric_space=first._metric_space,
            parameter_space=first._parameter_space,
        )

    # ── Storage access ────────────────────────────────────────────────────────

    @property
    def X(self) -> dict[str, np.ndarray]:
        """Named parameter columns (read-only arrays)."""
        return dict(self._X)

    @property
    def metrics(self) -> dict[str, np.ndarray]:
        """Named metric columns (read-only arrays)."""
        return dict(self._metrics)

    @property
    def metadata(self) -> Optional[dict[str, np.ndarray]]:
        """Metadata columns, or None."""
        return dict(self._metadata) if self._metadata is not None else None

    @property
    def metric_space(self) -> MetricSpace:
        """Output declarations and annotations."""
        return self._metric_space

    @property
    def parameter_space(self) -> Optional[ParameterSpace]:
        """Input domain; None when constructed from raw arrays."""
        return self._parameter_space

    @property
    def variable_names(self) -> list[str]:
        """Names of the stored parameter columns."""
        return list(self._X)

    @property
    def n_individuals(self) -> int:
        """Number of rows."""
        return self._n

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._n

    def __iter__(self) -> Iterator[IndividualView]:
        """Iterate over row views."""
        return (IndividualView(self, i) for i in range(self._n))

    def __getitem__(
        self, key: int | slice | Iterable[int] | npt.ArrayLike
    ) -> "IndividualView | Population":
        """Scalar index → ``IndividualView``; slice/list/mask → ``Population``."""
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += self._n
            if not 0 <= idx < self._n:
                raise IndexError(f"Index {key} out of range for {self._n} rows.")
            return IndividualView(self, idx)

        if isinstance(key, slice):
            indices = np.arange(self._n)[key]
        else:
            indices = np.asarray(key)
            if indices.dtype == bool:
                if len(indices) != self._n:
                    raise IndexError("Boolean mask length does not match.")
                indices = np.flatnonzero(indices)
            else:
                indices = indices.astype(int)

        return type(self)._sliced(self, indices)

    @classmethod
    def _sliced(cls, source: "Population", indices: np.ndarray) -> "Population":
        """Return a new plain ``Population`` with the selected rows."""
        return Population(
            X={name: col[indices] for name, col in source._X.items()},
            metrics={
                name: values[indices] for name, values in source._metrics.items()
            },
            metadata=(
                {name: vals[indices] for name, vals in source._metadata.items()}
                if source._metadata is not None
                else None
            ),
            metric_space=source._metric_space,
            parameter_space=source._parameter_space,
        )

    # ── Row lookup ────────────────────────────────────────────────────────────

    def _match_mask(self, x: Mapping[str, Any] | npt.ArrayLike) -> np.ndarray:
        """Boolean mask of rows exactly matching *x* on all parameter columns."""
        if isinstance(x, Mapping):
            assignment = dict(x)
        else:
            values = list(np.asarray(x, dtype=object).reshape(-1))
            if len(values) != len(self._X):
                raise ValueError(
                    f"Expected {len(self._X)} values, got {len(values)}."
                )
            assignment = dict(zip(self._X, values))

        mask = np.ones(self._n, dtype=bool)
        for name, value in assignment.items():
            if name not in self._X:
                raise KeyError(f"Unknown parameter {name!r}.")
            mask &= self._X[name] == value
        return mask

    def index_of(self, x: Mapping[str, Any] | npt.ArrayLike) -> int:
        """Return the first row index whose parameter values equal *x* exactly.

        *x* is a named assignment or a vector in column order.  Matching is
        exact: values that round-trip through normalize/denormalize are
        intentionally not treated as duplicates.
        """
        matches = np.flatnonzero(self._match_mask(x))
        if len(matches) == 0:
            raise KeyError(f"No row matches {x!r}.")
        return int(matches[0])

    def __contains__(self, x: Mapping[str, Any] | npt.ArrayLike) -> bool:
        """Return True when a row exactly matches *x*."""
        try:
            return bool(self._match_mask(x).any())
        except (KeyError, ValueError):
            return False

    def _rows_equal(self, i: int, j: int) -> bool:
        """Return True when rows *i* and *j* have exactly equal parameter values."""
        return all(column[i] == column[j] for column in self._X.values())

    def drop_duplicates(self) -> "Population":
        """Return a population with exact-duplicate parameter rows removed.

        The first occurrence of each unique parameter row is kept.
        """
        seen: set = set()
        keep = []
        columns = list(self._X.values())
        for i in range(self._n):
            key = tuple(column[i] for column in columns)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        if len(keep) == self._n:
            return self
        return self[np.asarray(keep, dtype=int)]

    # ── Projections: parameters ───────────────────────────────────────────────

    def _aligned_names(self) -> list[str]:
        """Parameter names aligned with ``parameter_space`` order when present."""
        if self._parameter_space is None:
            return list(self._X)
        names = [p.name for p in self._parameter_space.parameters]
        missing = [name for name in names if name not in self._X]
        if missing:
            raise CADETProcessError(
                f"Population is missing parameter columns {missing!r}."
            )
        return names

    @property
    def x(self) -> np.ndarray:
        """2-D projection of the parameter columns, ``(n, n_parameters)``.

        Column order follows ``parameter_space`` when present, otherwise
        insertion order.  The dtype is float when all columns are numeric,
        object otherwise.
        """
        names = self._aligned_names()
        if not names:
            return np.empty((self._n, 0))
        columns = [self._X[name] for name in names]
        if all(col.dtype.kind in "fiub" for col in columns):
            return np.column_stack([col.astype(float) for col in columns])
        return np.column_stack([col.astype(object) for col in columns])

    @property
    def ids(self) -> list[str]:
        """Content-derived id per row: sha256 digest of the parameter values.

        Rows with identical parameter values, in this or any other
        population, get the same id.
        """
        x = self.x
        return [_hash_row(x[i]) for i in range(self._n)]

    def _require_parameter_space(self) -> ParameterSpace:
        if self._parameter_space is None:
            raise CADETProcessError(
                "This operation requires a parameter_space."
            )
        return self._parameter_space

    @property
    def _independent_names(self) -> list[str]:
        space = self._require_parameter_space()
        return [p.name for p in space.independent_parameters]

    @property
    def X_num(self) -> np.ndarray:
        """Numeric matrix of the independent numeric parameters, physical units.

        The projection matches ``TransformedSpace.encode``: dependent and
        categorical parameters have no column.
        """
        space = self._require_parameter_space()
        transformed = space.transformed_space
        rows = [
            transformed.encode({name: self._X[name][i] for name in self._X})
            for i in range(self._n)
        ]
        n_num = len(rows[0]) if rows else 0
        return np.array(rows, dtype=float).reshape(self._n, n_num)

    @property
    def x_independent(self) -> np.ndarray:
        """Values of the independent parameters, ``(n, n_independent)``."""
        names = self._independent_names
        if not names:
            return np.empty((self._n, 0))
        return np.column_stack([self._X[name] for name in names])

    @property
    def x_transformed(self) -> np.ndarray:
        """Independent values in normalized coordinates, ``(n, n_independent)``."""
        space = self._require_parameter_space()
        x_ind = self.x_independent.astype(float)
        return np.array([space.normalize(row) for row in x_ind]).reshape(
            self._n, x_ind.shape[1]
        )

    @property
    def independent_variable_names(self) -> list[str]:
        """Names of the independent parameters."""
        return self._independent_names

    # ── Projections: metrics ──────────────────────────────────────────────────

    def _flat_values(self, metrics: list[Metric]) -> np.ndarray:
        """Row-major flattening of the given metric columns, ``(n, k)``."""
        if not metrics:
            return np.empty((self._n, 0))
        missing = [m.name for m in metrics if m.name not in self._metrics]
        if missing:
            raise CADETProcessError(
                f"Population has no values for metrics {missing!r}."
            )
        return np.hstack(
            [self._metrics[m.name].reshape(self._n, -1) for m in metrics]
        )

    @property
    def _plain_metrics_list(self) -> list[Metric]:
        """Declared metrics without direction or constraint annotation."""
        space = self._metric_space
        annotated = set(space.objective_names) | set(space.constraint_names)
        return [m for m in space.metrics if m.name not in annotated]

    @property
    def plain_metric_labels(self) -> list[str]:
        """Flattened labels of the unannotated metrics."""
        return [
            label for metric in self._plain_metrics_list for label in metric.labels
        ]

    @property
    def plain_metrics(self) -> np.ndarray:
        """Values of the unannotated metrics, ``(n, k)``; k may be 0."""
        return self._flat_values(self._plain_metrics_list)

    @property
    def f(self) -> np.ndarray:
        """Objective values in physical direction, ``(n, n_objectives)``."""
        return self._flat_values(
            [objective.metric for objective in self._metric_space.objectives]
        )

    @property
    def _minimization_factors(self) -> np.ndarray:
        return np.where(self._metric_space.minimize, 1.0, -1.0)

    @property
    def f_minimized(self) -> np.ndarray:
        """Objective values with maximization objectives negated."""
        return self.f * self._minimization_factors

    @property
    def f_min(self) -> np.ndarray:
        """Per-objective minimum."""
        return np.min(self.f, axis=0)

    @property
    def f_max(self) -> np.ndarray:
        """Per-objective maximum."""
        return np.max(self.f, axis=0)

    @property
    def f_avg(self) -> np.ndarray:
        """Per-objective average, ignoring non-finite entries."""
        return np.mean(np.ma.masked_invalid(self.f), axis=0)

    @property
    def f_best(self) -> np.ndarray:
        """Per-objective best value, respecting direction."""
        f_best = np.min(self.f_minimized, axis=0)
        return self._minimization_factors * f_best

    @property
    def f_best_indices(self) -> np.ndarray:
        """Row indices of the per-objective best values."""
        return np.argmin(self.f_minimized, axis=0)

    @property
    def g(self) -> np.ndarray:
        """Raw values of the constraint-annotated metrics, ``(n, k)``."""
        return self._flat_values(
            [constraint.metric for constraint in self._metric_space.constraints]
        )

    @property
    def g_min(self) -> np.ndarray:
        """Per-constraint minimum."""
        return np.min(self.g, axis=0)

    @property
    def g_max(self) -> np.ndarray:
        """Per-constraint maximum."""
        return np.max(self.g, axis=0)

    @property
    def g_avg(self) -> np.ndarray:
        """Per-constraint average, ignoring non-finite entries."""
        return np.mean(np.ma.masked_invalid(self.g), axis=0)

    @property
    def g_best(self) -> np.ndarray:
        """Constraint value at the row minimizing each violation column."""
        indices = np.argmin(self.cv_nonlincon, axis=0)
        g = self.g
        return np.array([g[row, i] for i, row in enumerate(indices)])

    @property
    def cv_nonlincon(self) -> np.ndarray:
        """Signed nonlinear constraint violations (positive = violated)."""
        constraints = self._metric_space.constraints
        if not constraints:
            return np.empty((self._n, 0))
        columns = []
        for constraint in constraints:
            values = self._flat_values([constraint.metric])
            if constraint.comparison_operator == "le":
                columns.append(values - constraint.bounds)
            else:
                columns.append(constraint.bounds - values)
        return np.hstack(columns)

    @property
    def cv_nonlincon_min(self) -> np.ndarray:
        """Per-constraint minimum violation."""
        return np.min(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_max(self) -> np.ndarray:
        """Per-constraint maximum violation."""
        return np.max(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_avg(self) -> np.ndarray:
        """Per-constraint average violation, ignoring non-finite entries."""
        return np.mean(np.ma.masked_invalid(self.cv_nonlincon), axis=0)

    # ── Projections: parameter-side violations ────────────────────────────────

    @property
    def cv_bounds(self) -> np.ndarray:
        """Bound violations ``[lb - x, x - ub]`` per row (positive = violated)."""
        space = self._require_parameter_space()
        x = self.x.astype(float)
        lb = space.lower_bounds
        ub = space.upper_bounds
        return np.hstack([lb - x, x - ub])

    @property
    def cv_lincon(self) -> np.ndarray:
        """Linear inequality constraint violations ``A @ x - b`` per row."""
        space = self._require_parameter_space()
        A, b = space.A, space.b
        if A.shape[0] == 0:
            return np.empty((self._n, 0))
        return self.x.astype(float) @ A.T - b

    @property
    def cv_lineqcon(self) -> np.ndarray:
        """Absolute linear equality constraint residuals per row."""
        space = self._require_parameter_space()
        A_eq, b_eq = space.A_eq, space.b_eq
        if A_eq.shape[0] == 0:
            return np.empty((self._n, 0))
        return np.abs(self.x.astype(float) @ A_eq.T - b_eq)

    @property
    def cv(self) -> np.ndarray:
        """All constraint violations combined, ``(n, k)``."""
        parts = []
        if self._parameter_space is not None:
            parts += [self.cv_bounds, self.cv_lincon, self.cv_lineqcon]
        parts.append(self.cv_nonlincon)
        return np.hstack(parts)

    # ── Feasibility ───────────────────────────────────────────────────────────

    def is_feasible(
        self,
        cv_bounds_tol: float = 0.0,
        cv_lincon_tol: float = 0.0,
        cv_lineqcon_tol: float = 0.0,
        cv_nonlincon_tol: float = 0.0,
    ) -> np.ndarray:
        """Boolean mask of rows satisfying all constraints within tolerances.

        Bound and linear-constraint checks require a ``parameter_space``
        and are skipped without one.
        """
        mask = np.ones(self._n, dtype=bool)
        if self._parameter_space is not None:
            mask &= np.all(self.cv_bounds <= cv_bounds_tol, axis=1)
            cv_lincon = self.cv_lincon
            if cv_lincon.shape[1] > 0:
                mask &= np.all(cv_lincon <= cv_lincon_tol, axis=1)
            cv_lineqcon = self.cv_lineqcon
            if cv_lineqcon.shape[1] > 0:
                mask &= np.all(cv_lineqcon <= cv_lineqcon_tol, axis=1)
        cv_nonlincon = self.cv_nonlincon
        if cv_nonlincon.shape[1] > 0:
            mask &= np.all(cv_nonlincon <= cv_nonlincon_tol, axis=1)
        return mask

    @property
    def feasible(self) -> "Population":
        """Rows satisfying all constraints (zero tolerance)."""
        return self[self.is_feasible()]

    @property
    def infeasible(self) -> "Population":
        """Rows violating at least one constraint (zero tolerance)."""
        return self[~self.is_feasible()]

    # ── Dominance and similarity ──────────────────────────────────────────────

    def dominates(
        self,
        i: int,
        j: int,
        feasible: Optional[np.ndarray] = None,
    ) -> bool:
        """Return True when row *i* dominates row *j*.

        Directions are read from ``metric_space``.  A feasible row
        dominates an infeasible one; two infeasible rows compare on their
        combined constraint violations.

        Parameters
        ----------
        i, j : int
            Row indices.
        feasible : np.ndarray, optional
            Precomputed feasibility mask (e.g. with optimizer tolerances).
            Computed with zero tolerance when omitted.
        """
        if feasible is None:
            feasible = self.is_feasible()
        if feasible[i] and not feasible[j]:
            return True
        if not feasible[i] and feasible[j]:
            return False
        if not feasible[i] and not feasible[j]:
            cv = self.cv
            return bool(
                np.all(cv[i] <= cv[j]) and np.any(cv[i] < cv[j])
            )
        f_minimized = self.f_minimized
        return bool(
            np.all(f_minimized[i] <= f_minimized[j])
            and np.any(f_minimized[i] < f_minimized[j])
        )

    def is_similar(self, i: int, j: int, tol: float = 1e-1) -> bool:
        """Return True when rows *i* and *j* are similar within relative *tol*.

        Numeric parameter columns and all metric values are compared with
        ``np.allclose``; non-numeric (categorical) columns must be equal.
        """
        if not tol:
            return False
        for column in self._X.values():
            if column.dtype.kind in "fiub":
                if not np.allclose(
                    float(column[i]), float(column[j]), rtol=tol
                ):
                    return False
            elif column[i] != column[j]:
                return False
        for values in self._metrics.values():
            if not np.allclose(values[i], values[j], rtol=tol):
                return False
        return True

    def drop_similar(self, tol: float) -> "Population":
        """Return a population with similar rows removed.

        Rows carrying a per-objective best value are always kept.
        """
        if not tol or self._n == 0:
            return self
        f = self.f
        f_best = self.f_best
        keep = np.ones(self._n, dtype=bool)
        for i in range(self._n):
            if not keep[i]:
                continue
            for j in range(self._n):
                if j == i or not keep[j]:
                    continue
                if self.is_similar(i, j, tol):
                    if np.any(f[j] == f_best):
                        continue
                    keep[j] = False
        if keep.all():
            return self
        return self[keep]

    # ── Plotting ──────────────────────────────────────────────────────────────

    @plotting.figure_utils
    def plot_objectives(
        self,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        autoscale: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        ax: npt.NDArray[plt.Axes] | None = None,
        setup_figure_kwargs: Optional[dict] = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot each metric against each parameter.

        Parameters
        ----------
        include_meta : bool, default=True
            If True, include unannotated metrics in the plot.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        autoscale : bool, default=True
            If True, automatically adjust the scaling of the axes.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        n_x = len(self.variable_names)
        if n_x == 0:
            raise CADETProcessError("Cannot plot without parameter columns.")

        labels = list(self._metric_space.objective_labels)
        if include_meta:
            labels += self.plain_metric_labels
        m = len(labels)

        if ax is None:
            fig, axs = plotting.setup_figure(
                **(setup_figure_kwargs or {}),
                nrows=m,
                ncols=n_x,
                aspect=1,
                squeeze=False,
            )
        else:
            axs = ax
            fig = axs[0, 0].get_figure()

        variables = self.variable_names
        feasible = self.feasible
        infeasible = self.infeasible
        x_feas = feasible.x
        x_infeas = infeasible.x

        def _values(pop: "Population") -> np.ndarray:
            if len(pop) == 0:
                return np.empty((0, m))
            if include_meta:
                return np.hstack((pop.f, pop.plain_metrics))
            return pop.f

        values_feas = _values(feasible)
        values_infeas = _values(infeasible)

        for i_var, var in enumerate(variables):
            if len(feasible) > 0:
                x_var_feas = x_feas[:, i_var]
            if len(infeasible) > 0:
                x_var_infeas = x_infeas[:, i_var]

            for i_metric, label in enumerate(labels):
                ax_ij = axs[i_metric, i_var]

                # Plot feasible/infeasible points
                if len(feasible) > 0:
                    v_metric_feas = values_feas[:, i_metric]
                    ax_ij.scatter(x_var_feas, v_metric_feas, alpha=0.5, color=color_feas)
                if len(infeasible) > 0 and plot_infeasible:
                    v_metric_infeas = values_infeas[:, i_metric]
                    ax_ij.scatter(x_var_infeas, v_metric_infeas, alpha=0.5, color=color_infeas)

                # Set axis labels and limits
                points = np.vstack([col.get_offsets() for col in ax_ij.collections])
                x_all = points[:, 0].astype(float)
                v_all = points[:, 1].astype(float)

                ax_ij.set_xlabel(var)
                ax_ij.set_ylabel(label)
                ax_ij.set_xlim(np.nanmin(x_all), np.nanmax(x_all))

                if autoscale and np.min(x_all) > 0:
                    if np.max(x_all) / np.min(x_all[x_all > 0]) > 100.0:
                        ax_ij.set_xscale("log")

                # Replace inf with nan
                mask = np.isfinite(v_all)
                v_all = v_all[mask]

                if len(v_all) == 0:
                    ax_ij.autoscale()
                    continue

                # Scale axis
                y_min = np.nanmin(v_all)
                y_max = np.nanmax(v_all)
                if y_min != y_max:
                    if autoscale and np.min(v_all) > 0:
                        if np.max(v_all) / np.min(v_all[v_all > 0]) > 100.0:
                            ax_ij.set_yscale("log")

                ax_ij.autoscale()

        return fig, axs

    @plotting.figure_utils
    def plot_pareto(
        self,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        *args: Any,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot pairwise metric values.

        Parameters
        ----------
        include_meta : bool, default=True
            If True, include unannotated metrics in the plot.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        *args : Any
            Additional positional arguments passed to `plot_pairwise`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `plot_pairwise`.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        labels = list(self._metric_space.objective_labels)
        if include_meta:
            labels += self.plain_metric_labels
        m = len(labels)

        feasible = self.feasible
        infeasible = self.infeasible

        def _values(pop: "Population") -> np.ndarray:
            if len(pop) == 0:
                return np.empty((0, m))
            if include_meta:
                return np.hstack((pop.f, pop.plain_metrics))
            return pop.f

        values_feas = _values(feasible)
        values_infeas = _values(infeasible)

        if len(feasible) > 0:
            fig, ax = plot_pairwise(
                values_feas,
                labels,
                color=color_feas,
                *args,
                ax=ax,
                setup_figure_kwargs=setup_figure_kwargs,
                tight_layout=False,
                **kwargs,
            )
        if plot_infeasible and len(infeasible) > 0:
            fig, ax = plot_pairwise(
                values_infeas,
                labels,
                color=color_infeas,
                *args,
                ax=ax,
                tight_layout=False,
                **({"update_layout": False, **kwargs})
            )

        return fig, ax

    @plotting.figure_utils
    def plot_pairwise(
        self,
        use_transformed: bool = False,
        plot_infeasible: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        *args: Any,
        ax: Optional[npt.NDArray[plt.Axes]] = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Create a pairwise parameter plot.

        Parameters
        ----------
        use_transformed : bool, optional
            If True, use the independent variables in normalized coordinates.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        *args : Any
            Additional positional arguments passed to `plot_pairwise`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `plot_pairwise`.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        feasible = self.feasible
        infeasible = self.infeasible

        if use_transformed:
            x_feas = feasible.x_transformed
            x_infeas = infeasible.x_transformed
            labels = self.independent_variable_names
        else:
            x_feas = feasible.x
            x_infeas = infeasible.x
            labels = self.variable_names

        fig, ax = plot_pairwise(
            x_feas,
            labels,
            color=color_feas,
            *args,
            ax=ax,
            tight_layout=False,
            setup_figure_kwargs=setup_figure_kwargs,
            **kwargs,
        )

        if plot_infeasible and len(infeasible) > 0:
            fig, ax = plot_pairwise(
                x_infeas,
                labels,
                color=color_infeas,
                *args,
                ax=ax,
                tight_layout=False,
                **{"update_layout": False, **kwargs}
            )

        return fig, ax

    # ── Serialization ─────────────────────────────────────────────────────────

    def _metric_space_spec(self) -> Dict:
        """Serializable description of the metric space."""
        spec = Dict()
        for i, metric in enumerate(self._metric_space.metrics):
            entry = Dict()
            entry.name = metric.name
            entry.n_metrics = metric.n_metrics
            entry.labels = list(metric.labels)
            if metric.dims is not None:
                entry.dims = list(metric.dims)
                entry.coords = {
                    dim: [str(c) for c in coords]
                    for dim, coords in metric.coords.items()
                }
            spec.metrics[str(i)] = entry
        for i, objective in enumerate(self._metric_space.objectives):
            spec.objectives[str(i)] = Dict(
                name=objective.name, minimize=int(objective.minimize)
            )
        for i, constraint in enumerate(self._metric_space.constraints):
            spec.constraints[str(i)] = Dict(
                name=constraint.name,
                bounds=np.asarray(constraint.bounds),
                comparison_operator=constraint.comparison_operator,
            )
        return spec

    @staticmethod
    def _metric_space_from_spec(spec: Mapping[str, Any]) -> MetricSpace:
        """Rebuild a ``MetricSpace`` from its serialized description."""
        space = MetricSpace()
        metrics = spec.get("metrics", {})
        for i in sorted(metrics, key=int):
            entry = metrics[i]
            name = _decode(entry["name"])
            labels = [_decode(label) for label in entry.get("labels", [])] or None
            dims = entry.get("dims")
            if dims is not None:
                dims = tuple(_decode(d) for d in dims)
                coords = {
                    _decode(dim): [_decode(c) for c in coords]
                    for dim, coords in entry["coords"].items()
                }
                space.add_metric(
                    Metric(name, dims=dims, coords=coords, labels=labels)
                )
            else:
                space.add_metric(
                    Metric(name, n_metrics=int(entry["n_metrics"]), labels=labels)
                )
        objectives = spec.get("objectives", {})
        for i in sorted(objectives, key=int):
            entry = objectives[i]
            space.add_objective(
                _decode(entry["name"]), minimize=bool(entry["minimize"])
            )
        constraints = spec.get("constraints", {})
        for i in sorted(constraints, key=int):
            entry = constraints[i]
            space.add_constraint(
                _decode(entry["name"]),
                bound=np.asarray(entry["bounds"], dtype=float),
                comparison_operator=_decode(entry["comparison_operator"]),
            )
        return space

    def to_dict(self) -> Dict:
        """Convert the population to a serializable dictionary."""
        data = Dict()
        for name, column in self._X.items():
            if column.dtype == object:
                column = column.astype(str)
            data.X[name] = np.asarray(column)
        for name, values in self._metrics.items():
            data.metrics[name] = np.asarray(values)
        if self._metadata is not None:
            for name, values in self._metadata.items():
                data.metadata[name] = np.asarray(values)
        data.metric_space = self._metric_space_spec()
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        metric_space: Optional[MetricSpace] = None,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> "Population":
        """Create a population from its dictionary representation.

        Parameters
        ----------
        data : dict
            Serialized population.
        metric_space : MetricSpace, optional
            Existing metric space to attach; rebuilt from the serialized
            spec when omitted.
        parameter_space : ParameterSpace, optional
            Parameter space to attach; serialized data carries none.
        """
        if "individuals" in data:
            raise CADETProcessError(
                "Checkpoint predates columnar Population storage and cannot "
                "be loaded."
            )
        if metric_space is None:
            metric_space = cls._metric_space_from_spec(data.get("metric_space", {}))
        X = {
            _decode(name): _decode(np.asarray(column))
            for name, column in data.get("X", {}).items()
        }
        metrics = {
            _decode(name): np.asarray(values)
            for name, values in data.get("metrics", {}).items()
        }
        metadata = {
            _decode(name): np.asarray(values)
            for name, values in data.get("metadata", {}).items()
        } or None
        return cls(
            X=X,
            metrics=metrics,
            metadata=metadata,
            metric_space=metric_space,
            parameter_space=parameter_space,
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"{type(self).__name__}(n_individuals={self._n}, "
            f"parameters={list(self._X)!r}, metrics={list(self._metrics)!r})"
        )


class ParetoFront(Population):
    """Mutable accumulator of non-dominated rows.

    ``ParetoFront`` is the one deliberately mutable holder in the columnar
    design: updates replace the internal storage wholesale, so any plain
    ``Population`` obtained from it stays immutable.

    Parameters
    ----------
    similarity_tol : float, optional
        Tolerance for removing near-duplicate front members.
    cv_bounds_tol, cv_lincon_tol, cv_lineqcon_tol, cv_nonlincon_tol : float
        Feasibility tolerances used when classifying candidate rows.
    metric_space : MetricSpace
        Output declarations; required.
    parameter_space : ParameterSpace, optional
        Input domain.
    """

    def __init__(
        self,
        similarity_tol: float = 1e-1,
        *,
        metric_space: MetricSpace,
        parameter_space: Optional[ParameterSpace] = None,
        cv_bounds_tol: float = 0.0,
        cv_lincon_tol: float = 0.0,
        cv_lineqcon_tol: float = 0.0,
        cv_nonlincon_tol: float = 0.0,
    ) -> None:
        self.similarity_tol = similarity_tol
        self._feasibility_tols = {
            "cv_bounds_tol": cv_bounds_tol,
            "cv_lincon_tol": cv_lincon_tol,
            "cv_lineqcon_tol": cv_lineqcon_tol,
            "cv_nonlincon_tol": cv_nonlincon_tol,
        }
        super().__init__(
            X={}, metric_space=metric_space, parameter_space=parameter_space
        )

    def _set_data(self, population: Population) -> None:
        """Replace the internal storage with the rows of *population*."""
        self._init_storage(
            population.X, population.metrics, population.metadata
        )

    def merge(self, other: Population) -> None:
        """Merge rows of *other*, dropping exact duplicates."""
        if not isinstance(other, Population):
            raise TypeError("Expected Population")
        if len(other) == 0:
            return
        if len(self) == 0:
            combined = other.drop_duplicates()
        else:
            combined = Population.concat(
                [Population._sliced(self, np.arange(len(self))), other]
            ).drop_duplicates()
        self._set_data(combined)

    def update_population(
        self, population: Population
    ) -> tuple[Population, bool]:
        """Update the front with a new population.

        Parameters
        ----------
        population : Population
            Candidate rows.

        Returns
        -------
        tuple[Population, bool]
            New members added to the front, and whether the update was a
            significant improvement.
        """
        n_front = len(self)
        if n_front == 0:
            work = population
            front_indices: list[int] = []
            candidates = list(range(len(population)))
        else:
            current = Population._sliced(self, np.arange(n_front))
            work = Population.concat([current, population])
            front_indices = list(range(n_front))
            candidates = list(range(n_front, n_front + len(population)))

        feasible = work.is_feasible(**self._feasibility_tols)

        selected = list(front_indices)
        new_members: list[int] = []
        significant: list[bool] = []

        for c in candidates:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove: list[int] = []

            if not feasible[c]:
                continue

            # An exact re-evaluation of a front member is never a new point,
            # independent of similarity_tol.
            if any(work._rows_equal(c, i) for i in selected):
                continue

            for i in selected:
                # Do not add if is dominated
                if not dominates_one and work.dominates(i, c, feasible=feasible):
                    is_dominated = True
                    break

                # Remove existing if infeasible
                elif not feasible[i]:
                    dominates_one = True
                    to_remove.append(i)
                    significant.append(True)

                # Remove existing if new dominates
                elif work.dominates(c, i, feasible=feasible):
                    dominates_one = True
                    to_remove.append(i)
                    if not work.is_similar(c, i, self.similarity_tol):
                        significant.append(True)

                # Ignore similar individuals
                elif work.is_similar(c, i, self.similarity_tol):
                    has_twin = True
                    break

            selected = [i for i in selected if i not in to_remove]

            if not is_dominated:
                if len(selected) == 0:
                    significant.append(True)
                if not has_twin:
                    significant.append(True)

                selected.append(c)
                new_members.append(c)

        if len(selected) == 0:
            # Fall back to the least infeasible candidates.
            offset = n_front
            for cv in (
                population.cv_bounds if population.parameter_space else None,
                population.cv_lincon if population.parameter_space else None,
                population.cv_lineqcon if population.parameter_space else None,
                population.cv_nonlincon,
            ):
                if cv is None or cv.shape[1] == 0:
                    continue
                for index in np.argmin(cv, axis=0):
                    candidate = offset + int(index)
                    if candidate not in selected:
                        selected.append(candidate)
        elif len(selected) > 1:
            selected = [i for i in selected if feasible[i]] or selected

        result = work[np.asarray(sorted(selected), dtype=int)]
        if self.similarity_tol:
            result = result.drop_similar(self.similarity_tol)
        self._set_data(result)

        new = work[np.asarray(new_members, dtype=int)]
        return new, any(significant)

    def remove_infeasible(self) -> None:
        """Remove infeasible rows from the front."""
        mask = self.is_feasible(**self._feasibility_tols)
        self._set_data(self[mask])

    def remove_dominated(self) -> None:
        """Remove dominated rows from the front."""
        feasible = self.is_feasible(**self._feasibility_tols)
        keep = np.ones(len(self), dtype=bool)
        for i in range(len(self)):
            if not keep[i]:
                continue
            for j in range(len(self)):
                if i == j or not keep[j]:
                    continue
                if self.dominates(i, j, feasible=feasible):
                    keep[j] = False
        self._set_data(self[keep])

    def remove_similar(self) -> None:
        """Remove similar rows from the front."""
        self._set_data(self.drop_similar(self.similarity_tol))

    def to_dict(self) -> Dict:
        """Convert the front to a dictionary."""
        data = super().to_dict()
        if self.similarity_tol:
            data.similarity_tol = self.similarity_tol
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        metric_space: Optional[MetricSpace] = None,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> "ParetoFront":
        """Create a ParetoFront from its dictionary representation."""
        if metric_space is None:
            metric_space = cls._metric_space_from_spec(data.get("metric_space", {}))
        front = cls(
            similarity_tol=data.get("similarity_tol") or 0,
            metric_space=metric_space,
            parameter_space=parameter_space,
        )
        front._set_data(
            Population.from_dict(
                data, metric_space=metric_space, parameter_space=parameter_space
            )
        )
        return front


def _determine_scaling(
    population: npt.ArrayLike,
    threshold: float = 100.0
) -> list[bool]:
    """
    Determine whether to use log scaling for each variable in a population.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    threshold : float, default=100.0
        Threshold for the data range to trigger log scaling.

    Returns
    -------
    list[bool]
        List of flags indicating whether to use log scaling for each variable.
    """
    n_variables = population.shape[1]
    scaling = []
    for i in range(n_variables):
        min_i, max_i = population[:, i].min(), population[:, i].max()
        scaling.append(min_i > 0 and (max_i / min_i) > threshold)
    return scaling


def _setup_pairwise_axes(
    population: npt.ArrayLike,
    variable_names: list[str] | None,
    autoscale: bool = True,
    update_layout: bool = True,
    ax: npt.NDArray[plt.Axes] | None = None,
    setup_figure_kwargs: dict | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """
    Set up a figure and axes for pairwise plots.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array-like structure with shape (n_samples, n_variables).
    variable_names : list[str], optional
        List of variable names. If None, default names are assigned.
    autoscale : bool, default=True
        If True, automatically determine log scaling for each variable.
    update_layout : bool, default=True
        If True, update layout (labels, ticks, etc.).
    ax : npt.NDArray[plt.Axes] | None, default=None
        Optional array of Matplotlib axes.
    setup_figure_kwargs : dict | None, default=None
        Additional figure setup options.

    Returns
    -------
    tuple
        A tuple containing:
        - plt.Figure: The Matplotlib Figure object.
        - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplot grid.
        - list[bool] : A list of flags indicating whether to use log scaling for
          each variable.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    n_variables = population.shape[1]

    # Determine scaling
    scaling = _determine_scaling(population) if autoscale else [False] * n_variables

    # Create or reuse axes
    if ax is None:
        fig, axs = plotting.setup_figure(
            nrows=n_variables,
            ncols=n_variables,
            sharex="col",
            sharey="row",
            squeeze=False,
            **{"aspect": 1.0, **(setup_figure_kwargs or {})},
        )
    else:
        axs = ax
        fig = axs[0, 0].get_figure()

    if axs.shape != (n_variables, n_variables):
        raise ValueError(
            "Inconsistent shape for provided axs. "
            f"Expected {(n_variables, n_variables)}, got {axs.shape}."
        )

    if update_layout:
        _update_layout(axs, population, variable_names, scaling)

    return fig, axs, scaling


def _update_layout(
    axs: npt.NDArray[plt.Axes],
    population: npt.ArrayLike,
    variable_names: list[str] | None,
    scaling: list[bool],
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """
    Set up a figure and axes for pairwise plots.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes] | None, default=None
        Array of Matplotlib axes.
    population : npt.ArrayLike
        2D array-like structure with shape (n_samples, n_variables).
    variable_names : list[str] | None
        List of variable names. If None, default names are assigned.
    scaling: list[bool]
        List of flags indicating whether to use log scaling for each variable.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    n_variables = population.shape[1]
    variable_names = variable_names or [f"$x_{{{i}}}$" for i in range(n_variables)]

    # Rows i
    for i in range(n_variables):
        scale_i = scaling[i]
        # Columns j
        for j in range(n_variables):
            scale_j = scaling[j]
            ax_ij = axs[i, j]

            # Apply log scale if needed
            if scale_j:
                if ax_ij.get_xscale() != "log":
                    ax_ij.set_xscale("log")
            else:
                ax_ij.ticklabel_format(axis="x", useMathText=True, scilimits=(-3, 3))

            if scale_i:
                if ax_ij.get_yscale() != "log":
                    ax_ij.set_yscale("log")
            else:
                ax_ij.ticklabel_format(axis="y", useMathText=True, scilimits=(-3, 3))

            # Ticks should only be visible on the first column ...
            if j == 0:
                ax_ij.yaxis.set_tick_params(labelleft=True)
            else:
                ax_ij.yaxis.set_tick_params(labelleft=False)
            # ... and last row
            if i == n_variables - 1:
                ax_ij.xaxis.set_tick_params(labelbottom=True)
            else:
                ax_ij.xaxis.set_tick_params(labelbottom=False)

            # Set axis labels on the edges
            if i == n_variables - 1:
                ax_ij.set_xlabel(variable_names[j])
            if j == 0:
                ax_ij.set_ylabel(variable_names[i])


def _plot_pairwise_histogram(
    axs: npt.NDArray[plt.Axes],
    data: npt.ArrayLike,
    color: str,
    n_bins: int = 20,
) -> None:
    """
    Plot histograms on the diagonal of a pairwise plot.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes]
        2D array of Matplotlib axes.
    data : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    color : str
        Color for the histograms.
    n_bins : int, default=20
        Number of bins for the histograms.
    """
    n_variables = axs.shape[0]

    for i in range(n_variables):
        ax = axs[i, i]

        x = data[:, i][np.isfinite(data[:, i])]

        if len(x) == 0:
            continue

        if not hasattr(ax, "_pairwise_bins"):
            ax_hist = ax.twinx()
            ax_hist.set_yticks([])

            lo, hi = x.min(), x.max()

            if ax.get_xscale() == "log":
                if lo <= 0:
                    raise ValueError("Log-scaled histogram requires positive data.")
                bins = np.geomspace(lo, hi, n_bins + 1)
            else:
                bins = np.linspace(lo, hi, n_bins + 1)

            ax._pairwise_bins = bins
            ax._pairwise_hist_ax = ax_hist
        else:
            bins = ax._pairwise_bins
            ax_hist = ax._pairwise_hist_ax

        ax_hist.hist(
            x,
            bins=bins,
            alpha=0.7,
            color=color,
            edgecolor="black",
            align="mid",
        )


def _plot_pairwise_scatter(
    axs: npt.NDArray[plt.Axes],
    data: npt.ArrayLike,
    color: str,
) -> None:
    """
    Plot scatter plots for non-diagonal elements of a pairwise plot.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes]
        2D array of Matplotlib axes.
    data : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    color : str
        Color for the scatter points.
    """
    n_variables = axs.shape[0]
    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:
                continue  # Skip diagonal

            ax = axs[i, j]
            ax.scatter(
                data[:, j], data[:, i],
                alpha=0.5, color=color
            )


@plotting.figure_utils
def plot_pairwise(
    population: npt.ArrayLike,
    variable_names: list[str] | None = None,
    color: str = "blue",
    n_bins: int = 20,
    autoscale: bool = True,
    plot_scatter: bool = True,
    plot_histogram: bool = True,
    update_layout: bool = True,
    ax: npt.NDArray[plt.Axes] | None = None,
    setup_figure_kwargs: dict | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Create a pairwise scatter plot for all variables of a population.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array-like structure containing numerical variables with shape
        (n_samples, n_variables)
    variable_names : list of str, optional
        list of variable names corresponding to columns in the data.
        If None, default names will be assigned.
    color : str
        Color for markers. Default is "tab10".
    n_bins : int, default=20
        Number of bins for histogram plots.
    autoscale : bool, default=True
        If True, automatically adjust the scaling of the axes.
    plot_scatter : bool, optional, default=True
        If True, add scatter plots.
    plot_histogram : bool, optional, default=True
        If True, add histogram plots.
    update_layout : bool, optional, default=True
        If True, update layout.
    ax : np.ndarray[plt.Axes] | None, default=None
        Optional array of Matplotlib axs.
        If not provided, a new figure is created.
    setup_figure_kwargs : dict | None, default=None
        Additional options to setup the figure.

    Returns
    -------
    tuple
        A tuple containing:
        - plt.Figure: The Matplotlib Figure object.
        - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplot grid.

    Raises
    ------
    ValueError
        If data does not contain 2D data.
        If the provided axes array does not have the correct shape.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    fig, ax, scaling = _setup_pairwise_axes(
        population,
        variable_names,
        autoscale,
        update_layout,
        ax,
        setup_figure_kwargs
    )

    # Plot histograms and scatter plots
    if plot_histogram:
        _plot_pairwise_histogram(
            ax,
            population,
            color=color,
        )
    if plot_scatter:
        _plot_pairwise_scatter(
            ax,
            population,
            color=color,
        )
    if update_layout:
        _update_layout(
            ax,
            population,
            variable_names,
            scaling,
        )

    return fig, ax
