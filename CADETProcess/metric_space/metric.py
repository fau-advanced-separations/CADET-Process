"""Metric declarations for MetricSpace.

``Metric`` describes a single named output of an evaluation backend: its
identity, declared shape, and labels.  It carries no direction, bounds, or
normalization; those are annotations owned by ``MetricSpace``.  Shapes are
declared, never inferred from the first evaluation: inferred shapes cause
hard-to-diagnose failures when batch size, feed composition, or failed
evaluations change the returned shape.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

__all__ = ["Metric"]


class Metric:
    """Declaration of a named output.

    Parameters
    ----------
    name : str
        Unique name identifying this metric within a ``MetricSpace``.
    n_metrics : int, optional
        Number of scalar entries.  Derived from *dims*/*coords* when named
        dimensions are declared; defaults to 1 (scalar).
    dims : tuple of str, optional
        Named dimensions, e.g. ``("component",)``.  Requires *coords*.
    coords : dict, optional
        Coordinate values per dimension, e.g. ``{"component": ["A", "B"]}``.
        Every entry in *dims* must have coordinates.
    labels : list of str, optional
        One label per scalar entry.  Defaults to *name* for scalars, to
        ``{name}_{coord}`` for one-dimensional named metrics, and to
        ``{name}_{i}`` otherwise.
    """

    def __init__(
        self,
        name: str,
        n_metrics: Optional[int] = None,
        dims: Optional[tuple[str, ...]] = None,
        coords: Optional[dict[str, list[Any]]] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")
        self.name = name

        if dims is not None:
            if coords is None:
                raise ValueError(f"Metric {name!r}: dims requires coords.")
            dims = tuple(dims)
            missing = [d for d in dims if d not in coords]
            if missing:
                raise ValueError(
                    f"Metric {name!r}: missing coordinates for dimensions {missing}."
                )
            self.dims: Optional[tuple[str, ...]] = dims
            self.coords: Optional[dict[str, list[Any]]] = {
                d: list(coords[d]) for d in dims
            }
            shape = tuple(len(self.coords[d]) for d in dims)
            derived = math.prod(shape)
            if n_metrics is not None and n_metrics != derived:
                raise ValueError(
                    f"Metric {name!r}: n_metrics={n_metrics} contradicts shape "
                    f"{shape} declared via dims/coords."
                )
            self._shape = shape
            self.n_metrics = derived
        else:
            if coords is not None:
                raise ValueError(f"Metric {name!r}: coords requires dims.")
            self.dims = None
            self.coords = None
            n = 1 if n_metrics is None else int(n_metrics)
            if n < 1:
                raise ValueError(f"Metric {name!r}: n_metrics must be >= 1.")
            self.n_metrics = n
            self._shape = () if n == 1 else (n,)

        self.labels = labels

    @property
    def shape(self) -> tuple[int, ...]:
        """Declared value shape; ``()`` for scalar metrics."""
        return self._shape

    @property
    def labels(self) -> list[str]:
        """One label per scalar entry."""
        if self._labels is not None:
            return list(self._labels)
        if self.n_metrics == 1:
            return [self.name]
        if self.dims is not None and len(self.dims) == 1:
            return [f"{self.name}_{c}" for c in self.coords[self.dims[0]]]
        return [f"{self.name}_{i}" for i in range(self.n_metrics)]

    @labels.setter
    def labels(self, value: Optional[list[str]]) -> None:
        if value is not None and len(value) != self.n_metrics:
            raise ValueError(
                f"Metric {self.name!r}: expected {self.n_metrics} labels, "
                f"got {len(value)}."
            )
        self._labels = list(value) if value is not None else None

    def validate(self, value: Any) -> np.ndarray:
        """Validate *value* against the declared shape.

        Returns the value as ``np.ndarray`` with the declared shape.  A scalar
        metric additionally accepts a length-1 vector, which is canonicalized
        to a 0-d array.

        Raises
        ------
        ValueError
            If the value's shape does not match the declaration.
        """
        arr = np.asarray(value)
        if arr.shape == self._shape:
            return arr
        if self._shape == () and arr.shape == (1,):
            return arr.reshape(())
        raise ValueError(
            f"Metric {self.name!r}: expected shape {self._shape}, got {arr.shape}."
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        if self.dims is not None:
            return f"Metric(name={self.name!r}, dims={self.dims!r})"
        if self.n_metrics != 1:
            return f"Metric(name={self.name!r}, n_metrics={self.n_metrics})"
        return f"Metric(name={self.name!r})"
