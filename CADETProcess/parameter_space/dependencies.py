"""Parameter dependency declarations for ParameterSpace.

A ``ParameterDependency`` makes one parameter a deterministic function of
one or more others.  The space resolves dependent parameters in topological
order before writing values into evaluation objects.
"""

from __future__ import annotations

from collections.abc import Callable

from CADETProcess.parameter_space.parameters import ParameterBase

__all__ = ["ParameterDependency"]


class ParameterDependency:
    """Declare that one parameter depends on others via a transform.

    The optimizer sees only independent parameters; ``ParameterSpace.set_values``
    resolves all dependent parameters before writing.

    Parameters
    ----------
    dependent_parameter : ParameterBase
        The parameter whose value is computed from others.
    independent_parameters : tuple[ParameterBase, ...]
        Parameters whose values are passed to *transform*, in order.
    transform : Callable
        Called as ``transform(*independent_values)``; must return the value
        for *dependent_parameter*.
    """

    def __init__(
        self,
        dependent_parameter: ParameterBase,
        independent_parameters: list[ParameterBase],
        transform: Callable,
    ) -> None:
        self.dependent_parameter = dependent_parameter
        self.independent_parameters = tuple(independent_parameters)
        self.transform = transform

    def __repr__(self) -> str:
        """Return a readable representation."""
        inputs = [p.name for p in self.independent_parameters]
        return (
            f"ParameterDependency("
            f"dependent={self.dependent_parameter.name!r}, "
            f"inputs={inputs!r})"
        )
