"""Linear constraints for ParameterSpace.

Both classes represent affine constraints over a subset of parameters.
They hold coefficients and references to the parameters they constrain;
the space assembles the full matrix form for consumption by optimizers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from CADETProcess.parameter_space.parameters import ParameterBase

__all__ = [
    "LinearConstraint",
    "LinearEqualityConstraint",
]


def _normalize_lhs(
    parameters: list[ParameterBase],
    lhs: Union[float, list[float]],
    label: str,
) -> list[float]:
    """Expand a scalar lhs to a per-parameter list and validate length."""
    if np.isscalar(lhs):
        return [float(lhs)] * len(parameters)
    lhs = list(lhs)
    if len(lhs) != len(parameters):
        raise ValueError(
            f"{label}: lhs has {len(lhs)} coefficients but "
            f"{len(parameters)} parameters were given."
        )
    return [float(c) for c in lhs]


class LinearConstraint:
    """Linear inequality constraint: ``lhs · x <= b``.

    Parameters
    ----------
    parameters : ParameterBase or sequence of ParameterBase
        Parameters involved in the constraint.  Validated as numeric
        (``RangedParameter``) when registered with ``ParameterSpace``.
    lhs : float or list[float]
        Coefficients.  A scalar is broadcast to all parameters.
    b : float
        Right-hand side.
    """

    def __init__(
        self,
        parameters: Union[ParameterBase, list[ParameterBase]],
        lhs: Union[float, list[float]] = 1.0,
        b: float = 0.0,
    ) -> None:
        if isinstance(parameters, ParameterBase):
            parameters = [parameters]
        elif not isinstance(parameters, Sequence):
            raise TypeError(
                f"parameters must be a ParameterBase or a sequence thereof, "
                f"got {type(parameters).__name__}."
            )
        else:
            parameters = list(parameters)
        if len({p.name for p in parameters}) != len(parameters):
            raise ValueError("Duplicate parameters in constraint.")
        self.parameters = parameters
        self.lhs = _normalize_lhs(parameters, lhs, "LinearConstraint")
        self.b = float(b)

    def __repr__(self) -> str:
        """Return a readable representation."""
        names = [p.name for p in self.parameters]
        return f"LinearConstraint(parameters={names}, lhs={self.lhs}, b={self.b})"


class LinearEqualityConstraint:
    """Linear equality constraint: ``lhs · x = b``.

    Parameters
    ----------
    parameters : ParameterBase or sequence of ParameterBase
        Parameters involved in the constraint.  Validated as numeric
        (``RangedParameter``) when registered with ``ParameterSpace``.
    lhs : float or list[float]
        Coefficients.  A scalar is broadcast to all parameters.
    b : float
        Right-hand side.
    eps : float
        Relaxation tolerance passed to external solvers (e.g. hopsy).
        Not enforced by ``ParameterSpace`` itself; ``A_eq x = b_eq`` is
        treated as exact equality during matrix assembly.
    """

    def __init__(
        self,
        parameters: Union[ParameterBase, list[ParameterBase]],
        lhs: Union[float, list[float]] = 1.0,
        b: float = 0.0,
        eps: float = 0.0,
    ) -> None:
        if isinstance(parameters, ParameterBase):
            parameters = [parameters]
        elif not isinstance(parameters, Sequence):
            raise TypeError(
                f"parameters must be a ParameterBase or a sequence thereof, "
                f"got {type(parameters).__name__}."
            )
        else:
            parameters = list(parameters)
        if len({p.name for p in parameters}) != len(parameters):
            raise ValueError("Duplicate parameters in constraint.")
        self.parameters = parameters
        self.lhs = _normalize_lhs(parameters, lhs, "LinearEqualityConstraint")
        self.b = float(b)
        self.eps = float(eps)

    def __repr__(self) -> str:
        """Return a readable representation."""
        names = [p.name for p in self.parameters]
        return (
            f"LinearEqualityConstraint(parameters={names}, lhs={self.lhs}, "
            f"b={self.b}, eps={self.eps})"
        )
