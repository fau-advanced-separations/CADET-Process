"""Path-based and callable parameter mappers.

Provides the primitive that writes a parameter value into an evaluation object.
``DotPathMapper`` traverses a dot-separated attribute/key path; ``IndexedMapper``
extends this with a read-patch-write cycle for array elements; ``CallableMapper``
delegates to a user-supplied function for anything else.  All mappers broadcast
over a list of evaluation objects.

The path segment parser is centralized here so that every mapper that accepts a
string path uses identical traversal and parsing semantics.

Path syntax
-----------
``"column.length"``
    Dot-separated attribute / dict-key traversal.
``"film_diffusion[2]"``
    Attribute traversal followed by an integer array index.
``"film_diffusion[1:3]"``
    Attribute traversal followed by a slice.
``"units[0].column.length"``
    Intermediate index followed by further attribute traversal.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.exceptions import VisibleDeprecationWarning

__all__ = [
    "parse_path",
    "ParameterMapperBase",
    "DotPathMapper",
    "IndexedMapper",
    "CallableMapper",
    "make_preprocessing_mapper",
]

# Matches the bracket expression at the very end of a dot-segment: "name[expr]"
_BRACKET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]$")


def _parse_index_expr(expr: str) -> int | slice:
    """Parse an index expression string into an ``int`` or ``slice``.

    Parameters
    ----------
    expr : str
        Content inside ``[...]``, e.g. ``"2"``, ``"1:3"``, ``":"``

    Returns
    -------
    int or slice
    """
    expr = expr.strip()
    if ":" in expr:
        if expr.count(":") > 2:
            raise ValueError(f"Malformed slice expression {expr!r}: too many colons.")
        parts = expr.split(":", 2)
        try:
            start = int(parts[0].strip()) if parts[0].strip() else None
            stop = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else None
            step = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
        except ValueError as exc:
            raise ValueError(f"Malformed slice expression {expr!r}.") from exc
        return slice(start, stop, step)
    try:
        return int(expr)
    except ValueError as exc:
        raise ValueError(f"Malformed index expression {expr!r}.") from exc


def parse_path(path: str) -> tuple[str | int | slice, ...]:
    """Split a path string into typed segments.

    Dot notation separates attribute / dict-key hops.  Bracket notation at the
    end of a segment specifies an array index or slice.

    Parameters
    ----------
    path : str
        Path string, e.g. ``"column.length"``, ``"film_diffusion[2]"``,
        ``"column.film_diffusion[1:3]"``, or ``"units[0].column.length"``.

    Returns
    -------
    tuple[str | int | slice, ...]
        Non-empty tuple of segments.  String segments are attribute names or
        dict keys; ``int`` and ``slice`` segments are array indices.

    Raises
    ------
    ValueError
        If *path* is empty, contains an empty attribute segment, or has
        malformed bracket notation.

    Examples
    --------
    >>> parse_path("column.length")
    ('column', 'length')
    >>> parse_path("film_diffusion[2]")
    ('film_diffusion', 2)
    >>> parse_path("column.film_diffusion[1:3]")
    ('column', 'film_diffusion', slice(1, 3, None))
    >>> parse_path("units[0].column.length")
    ('units', 0, 'column', 'length')
    >>> parse_path("feed")
    ('feed',)
    """
    if not path:
        raise ValueError("Path must not be empty.")

    segments: list[str | int | slice] = []
    for dot_seg in path.split("."):
        if not dot_seg:
            raise ValueError(f"Path {path!r} contains an empty segment.")
        m = _BRACKET_RE.match(dot_seg)
        if m:
            segments.append(m.group(1))
            segments.append(_parse_index_expr(m.group(2)))
        else:
            # No bracket — must be a plain identifier (no stray ']' etc.)
            if "[" in dot_seg or "]" in dot_seg:
                raise ValueError(
                    f"Malformed bracket notation in segment {dot_seg!r} of path {path!r}."
                )
            segments.append(dot_seg)

    return tuple(segments)


def _traverse(root: Any, segments: Sequence[str | int | slice]) -> tuple[Any, str | int | slice]:
    """Walk *segments* starting from *root*, returning ``(parent, leaf_segment)``.

    Intermediate string segments are resolved by dict lookup (for ``Mapping``
    nodes) or attribute access (for everything else).  Intermediate int / slice
    segments are resolved via ``__getitem__``.  The leaf segment is returned
    unevaluated so the caller decides how to write.

    Raises
    ------
    KeyError
        If a dict-hop key is missing.
    AttributeError
        If an attribute-hop name is missing.
    IndexError
        If an integer-hop index is out of range.
    """
    cur = root
    for seg in segments[:-1]:
        if isinstance(seg, (int, slice)):
            cur = cur[seg]
        elif isinstance(cur, Mapping):
            cur = cur[seg]
        else:
            cur = getattr(cur, seg)
    return cur, segments[-1]


class ParameterMapperBase(ABC):
    """Base class: broadcasts a write over a list of evaluation objects."""

    def __init__(self, evaluation_objects: Sequence[Any]) -> None:
        self.evaluation_objects = list(evaluation_objects)

    def set_value(self, value: Any) -> None:
        """Write *value* into every object in ``evaluation_objects``."""
        for obj in self.evaluation_objects:
            self._set_value(obj, value)

    @abstractmethod
    def _set_value(self, obj: Any, value: Any) -> None:
        """Write *value* into a single evaluation object."""

    def get_value(self) -> Any:
        """Read the current value from the first evaluation object.

        Returns ``None`` for mappers that do not support read-back
        (e.g. ``CallableMapper``).
        """
        if not self.evaluation_objects:
            return None
        return self._get_value(self.evaluation_objects[0])

    def _get_value(self, obj: Any) -> Any:  # noqa: ARG002
        """Read from a single evaluation object.  Override to support read-back."""
        return None


class DotPathMapper(ParameterMapperBase):
    """Write a value to the leaf referenced by a dot-separated path.

    Traverses dicts and objects in any combination; missing hops raise.
    The leaf segment must be a string (attribute name or dict key).  For
    array element writes use ``IndexedMapper``.

    Parameters
    ----------
    evaluation_objects : sequence
        Objects to write into.
    path : str
        Dot-separated path to the target attribute or dict key.  Must not
        contain bracket index notation (e.g. ``"film_diffusion[2]"``).

    Raises
    ------
    ValueError
        If *path* ends with a bracket index.  Use ``IndexedMapper`` instead.
    """

    def __init__(self, evaluation_objects: Sequence[Any], path: str) -> None:
        super().__init__(evaluation_objects)
        self.path = path
        self._segments = parse_path(path)
        if isinstance(self._segments[-1], (int, slice)):
            raise ValueError(
                f"Path {path!r} ends with an array index. "
                "Use IndexedMapper for indexed array writes."
            )

    def _set_value(self, obj: Any, value: Any) -> None:
        parent, leaf = _traverse(obj, self._segments)
        if isinstance(parent, Mapping):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)

    def _get_value(self, obj: Any) -> Any:
        parent, leaf = _traverse(obj, self._segments)
        if isinstance(parent, Mapping):
            return parent[leaf]
        return getattr(parent, leaf)


class IndexedMapper(ParameterMapperBase):
    """Read-patch-write mapper for individual array elements or slices.

    Traverses to the target array via *path*, then patches the entry at
    *index* with the new scalar value and writes the whole array back.
    This read-patch-write cycle is necessary because array-valued parameters
    are stored as plain values (not mutable proxies) on most model objects.

    Supports two equivalent forms::

        IndexedMapper(objs, path="film_diffusion[2]")
        IndexedMapper(objs, path="film_diffusion", index=2)

    Multi-dimensional parameters (e.g. 2-D reaction exponent arrays or
    polynomial concentration profiles) are supported via tuple indices::

        IndexedMapper(objs, path="exponents_fwd", index=(0, 1))
        IndexedMapper(objs, path="c", index=np.s_[0, :])

    Genuinely inhomogeneous (ragged) arrays raise ``NotImplementedError``.
    Use a ``CallableMapper`` for those.

    Parameters
    ----------
    evaluation_objects : sequence
        Objects to write into.
    path : str
        Dot-separated path to the target array attribute.  May embed a
        scalar index as bracket notation: ``"film_diffusion[2]"``.
    index : int, slice, or tuple of (int | slice), optional
        Index into the array.  Required when *path* does not embed an index;
        must be omitted when *path* already contains one.  Pass a tuple for
        multi-dimensional access, e.g. ``(0, 1)`` or ``np.s_[0, :]``.
    """

    def __init__(
        self,
        evaluation_objects: Sequence[Any],
        path: str,
        index: int | slice | tuple | None = None,
    ) -> None:
        super().__init__(evaluation_objects)
        segments = parse_path(path)

        if isinstance(segments[-1], (int, slice)):
            if index is not None:
                raise ValueError(
                    "Cannot specify both embedded index notation in path and "
                    "an explicit index parameter."
                )
            self._attr_segments = segments[:-1]
            self._index: int | slice | tuple = segments[-1]
        else:
            if index is None:
                raise ValueError(
                    "IndexedMapper requires an index.  Either embed it in the "
                    "path (e.g. 'film_diffusion[2]') or pass index= explicitly."
                )
            self._attr_segments = segments
            self._index = index

        if not self._attr_segments:
            raise ValueError("IndexedMapper requires a non-empty attribute path.")

        self.path = ".".join(str(s) for s in self._attr_segments)
        self.index = self._index

    def _set_value(self, obj: Any, value: Any) -> None:
        parent, leaf = _traverse(obj, self._attr_segments)
        if isinstance(parent, Mapping):
            current = parent[leaf]
        else:
            current = getattr(parent, leaf)

        new_value = self._patch(current, value)

        if isinstance(parent, Mapping):
            parent[leaf] = new_value
        else:
            setattr(parent, leaf, new_value)

    def _get_value(self, obj: Any) -> Any:
        parent, leaf = _traverse(obj, self._attr_segments)
        if isinstance(parent, Mapping):
            arr = parent[leaf]
        else:
            arr = getattr(parent, leaf)
        return np.asarray(arr)[self._index]

    def _patch(self, current: Any, value: Any) -> Any:
        """Return a copy of *current* with ``self._index`` set to *value*."""
        was_list = isinstance(current, list)
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            try:
                arr = np.array(current, copy=True)
            except (ValueError, VisibleDeprecationWarning):
                raise NotImplementedError(
                    "Inhomogeneous (ragged) arrays are not supported by "
                    "IndexedMapper.  Use CallableMapper instead."
                )
        if arr.dtype == object:
            raise NotImplementedError(
                "Object-type arrays are not supported by IndexedMapper."
            )
        arr[self._index] = value
        return arr.tolist() if was_list else arr


class CallableMapper(ParameterMapperBase):
    """Write a value by delegating to a user-supplied function.

    The callable receives ``(evaluation_object, value)`` and is responsible
    for the actual write.  This is the escape hatch for any target that
    cannot be expressed as a dot-path.

    Parameters
    ----------
    evaluation_objects : sequence
        Objects to write into.
    fn : Callable[[Any, Any], None]
        Function called as ``fn(obj, value)`` for each evaluation object.
    """

    def __init__(
        self, evaluation_objects: Sequence[Any], fn: Callable[[Any, Any], None]
    ) -> None:
        super().__init__(evaluation_objects)
        self.fn = fn

    def _set_value(self, obj: Any, value: Any) -> None:
        self.fn(obj, value)


def make_preprocessing_mapper(
    evaluation_objects: Sequence[Any],
    path: str,
    pre_processing: Callable[[Any], Any],
) -> CallableMapper:
    """Return a CallableMapper that applies *pre_processing* before writing via *path*.

    Useful when the value must be transformed (e.g. unit conversion, clipping)
    before it is written to the attribute at *path*.

    Parameters
    ----------
    evaluation_objects : sequence
        Objects to write into.
    path : str
        Dot-separated attribute path on each evaluation object.
    pre_processing : callable
        Applied to the raw value before the write: ``obj.attr = pre_processing(v)``.
    """
    *parts, attr = path.split(".")

    def _write(obj: Any, v: Any) -> None:
        target = obj
        for part in parts:
            target = getattr(target, part)
        setattr(target, attr, pre_processing(v))

    return CallableMapper(evaluation_objects, _write)
