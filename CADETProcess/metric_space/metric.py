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
        Number of scalar entries.  Derived from *coords* when named
        dimensions are declared; defaults to 1 (scalar).
    coords : dict, optional
        Coordinate values per named dimension, e.g.
        ``{"component": ["A", "B"]}``.  The keys are the dimensions and their
        order is the order of the axes.
    labels : list of str, optional
        One label per scalar entry.  Defaults to *name* for scalars, to
        ``{name}_{coord}`` for one-dimensional named metrics, and to
        ``{name}_{i}`` otherwise.
    """

    def __init__(
        self,
        name: str,
        n_metrics: Optional[int] = None,
        coords: Optional[dict[str, list[Any]]] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")
        self.name = name
        self._declare(n_metrics, coords, labels)

    def _declare(
        self,
        n_metrics: Optional[int],
        coords: Optional[dict[str, list[Any]]],
        labels: Optional[list[str]],
    ) -> None:
        """Set shape and labels from a declaration, validating consistency."""
        name = self.name
        if coords is not None:
            self.coords: Optional[dict[str, list[Any]]] = {
                d: list(values) for d, values in coords.items()
            }
            shape = tuple(len(values) for values in self.coords.values())
            derived = math.prod(shape)
            if n_metrics is not None and n_metrics != derived:
                raise ValueError(
                    f"Metric {name!r}: n_metrics={n_metrics} contradicts shape "
                    f"{shape} declared via coords."
                )
            self._shape = shape
            self.n_metrics = derived
        else:
            self.coords = None
            n = 1 if n_metrics is None else int(n_metrics)
            if n < 1:
                raise ValueError(f"Metric {name!r}: n_metrics must be >= 1.")
            self.n_metrics = n
            self._shape = () if n == 1 else (n,)

        self.labels = labels

    def redeclare(self, other: Metric, *, allow_resize: bool = False) -> None:
        """Adopt *other*'s dimensions, keeping identity.

        Mutates in place because the declaration is shared by reference with
        every annotation holding it. Resizing is opt-in during problem setup;
        the caller must update any entry-wise annotations such as bounds.

        Parameters
        ----------
        other : Metric
            Replacement declaration with the same name.
        allow_resize : bool, optional
            Permit a changed entry count, default False.

        Raises
        ------
        ValueError
            If *other* has a different name, or changes the number of entries
            without *allow_resize*.
        """
        if other.name != self.name:
            raise ValueError(
                f"Metric {self.name!r}: cannot redeclare as {other.name!r}."
            )
        if other.n_metrics != self.n_metrics and not allow_resize:
            raise ValueError(
                f"Metric {self.name!r}: redeclaration changes the entry count "
                f"from {self.n_metrics} to {other.n_metrics}."
            )
        self._declare(other.n_metrics, other.coords, other._labels)

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
        if self.coords is not None and len(self.coords) == 1:
            (values,) = self.coords.values()
            return [f"{self.name}_{c}" for c in values]
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

    @property
    def dims(self) -> Optional[tuple[str, ...]]:
        """Names of the dimensions, in axis order; ``None`` without coordinates."""
        return None if self.coords is None else tuple(self.coords)

    def __repr__(self) -> str:
        """Return a readable representation."""
        if self.dims is not None:
            return f"Metric(name={self.name!r}, dims={self.dims!r})"
        if self.n_metrics != 1:
            return f"Metric(name={self.name!r}, n_metrics={self.n_metrics})"
        return f"Metric(name={self.name!r})"
