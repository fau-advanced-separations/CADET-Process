"""Parameter classes for ParameterSpace.

``ParameterBase`` and its subclasses describe a single optimizable quantity:
its identity, domain, validation rules, and optional normalization.  They
carry no reference to evaluation objects and perform no writes.  The space
wires mappers and calls them; parameters only validate.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal, Optional

import numpy as np

from CADETProcess.parameter_space.normalize import (
    AutoNormalizer,
    LinearNormalizer,
    LogNormalizer,
    NormalizerBase,
    NullNormalizer,
)

__all__ = [
    "ParameterBase",
    "RangedParameter",
    "ChoiceParameter",
]


class ParameterBase:
    """Base class for an optimizable parameter.

    Subclasses override ``validate`` to enforce type and range constraints.

    Parameters
    ----------
    name : str
        Unique name used to identify this parameter within a ``ParameterSpace``.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def validate(self, value: Any) -> Any:
        """Validate *value* against the parameter's constraints.

        Returns *value* unchanged (or canonicalized by subclasses).
        Raises ``TypeError`` or ``ValueError`` when *value* is invalid.
        The default implementation accepts any value.
        """
        return value

    def __repr__(self) -> str:
        """Return a readable representation."""
        return f"{type(self).__name__}(name={self.name!r})"


class RangedParameter(ParameterBase):
    """Scalar parameter bounded by a lower and upper limit.

    Parameters
    ----------
    name : str
        Unique parameter name.
    parameter_type : {int, float}
        Scalar domain.  ``int`` accepts integral values only (no booleans);
        ``float`` accepts any real-valued scalar, including integers and NumPy
        real scalars.
    lb : float
        Lower bound (inclusive).  Defaults to ``-inf``.
    ub : float
        Upper bound (inclusive).  Defaults to ``+inf``.
    normalization : {"auto", "linear", "log"} or None
        Normalization scheme.  ``None`` means no normalization (identity).
        Requires finite bounds when set.
    significant_digits : int or None
        When set, values are rounded to this many significant digits by
        ``ParameterSpace.set_values`` before being written to the evaluation
        object.
    """

    def __init__(
        self,
        name: str,
        parameter_type: type[int] | type[float] = float,
        lb: float = -math.inf,
        ub: float = math.inf,
        normalization: Literal["auto", "linear", "log"] | None = None,
        significant_digits: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        if parameter_type not in (int, float):
            raise TypeError(
                f"Parameter {name!r}: parameter_type must be int or float, "
                f"got {parameter_type!r}."
            )
        if lb >= ub:
            raise ValueError(
                f"Parameter {name!r}: lower bound must be < upper bound, "
                f"got lb={lb}, ub={ub}."
            )
        self.parameter_type = parameter_type
        self.lb = lb
        self.ub = ub
        self.normalization = normalization
        self.significant_digits = significant_digits
        self.normalizer: NormalizerBase = self._build_normalizer()

    def _build_normalizer(self) -> NormalizerBase:
        if self.normalization is None:
            return NullNormalizer(lb_input=self.lb, ub_input=self.ub)
        if np.isinf(self.lb) or np.isinf(self.ub):
            raise ValueError(
                f"Parameter {self.name!r}: normalization requires finite bounds."
            )
        if self.normalization == "linear":
            return LinearNormalizer(lb_input=self.lb, ub_input=self.ub)
        if self.normalization == "log":
            return LogNormalizer(lb_input=self.lb, ub_input=self.ub)
        if self.normalization == "auto":
            return AutoNormalizer(lb_input=self.lb, ub_input=self.ub)
        raise ValueError(
            f"Parameter {self.name!r}: unknown normalization {self.normalization!r}."
        )

    def validate(self, value: Any) -> Any:
        """Validate type and bounds.

        Returns *value* unchanged.

        Raises
        ------
        TypeError
            If *value* is not numerically compatible with ``parameter_type``
            (accepts NumPy scalar types via the ``numbers`` abstract base
            classes; rejects ``bool`` when ``parameter_type`` is ``int``).
        ValueError
            If *value* lies outside ``[lb, ub]``.
        """
        if self.parameter_type is int:
            ok = isinstance(value, numbers.Integral) and not isinstance(value, bool)
        else:
            ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if not ok:
            raise TypeError(
                f"Parameter {self.name!r}: expected {self.parameter_type.__name__}, "
                f"got {type(value).__name__}."
            )
        if not self.lb <= value <= self.ub:
            raise ValueError(
                f"Parameter {self.name!r}: value {value} outside [{self.lb}, {self.ub}]."
            )
        return self.parameter_type(value)

    def normalize(self, value: float) -> float:
        """Map *value* from ``[lb, ub]`` to the normalized space."""
        return self.normalizer.normalize(value)

    def denormalize(self, value: float) -> float:
        """Map *value* from the normalized space back to ``[lb, ub]``."""
        return self.normalizer.denormalize(value)

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"RangedParameter(name={self.name!r}, "
            f"parameter_type={self.parameter_type.__name__}, "
            f"lb={self.lb}, ub={self.ub})"
        )


class ChoiceParameter(ParameterBase):
    """Parameter constrained to a finite set of allowed values.

    Parameters
    ----------
    name : str
        Unique parameter name.
    valid_values : list
        Allowed choices.  Any value not in this list is rejected.
    """

    def __init__(self, name: str, valid_values: list[Any]) -> None:
        super().__init__(name)
        if not valid_values:
            raise ValueError(
                f"Parameter {name!r}: valid_values must not be empty."
            )
        self.valid_values = list(valid_values)

    def validate(self, value: Any) -> Any:
        """Raise ``ValueError`` when *value* is not in ``valid_values``.

        Returns *value* unchanged.
        """
        if value not in self.valid_values:
            raise ValueError(
                f"Parameter {self.name!r}: {value!r} is not a valid choice; "
                f"must be one of {self.valid_values!r}."
            )
        return value

    def __repr__(self) -> str:
        """Return a readable representation."""
        return f"ChoiceParameter(name={self.name!r}, valid_values={self.valid_values!r})"
