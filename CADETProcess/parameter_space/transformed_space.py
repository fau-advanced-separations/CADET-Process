"""TransformedSpace: optimizer-facing view of a ParameterSpace in normalized coordinates.

``TransformedSpace`` wraps a ``ParameterSpace`` and presents bounds and constraint
matrices in normalized coordinates.  It has no write method: the single write path
is ``ParameterSpace.set_values``, and vector-coordinate callers compose explicitly,
``space.set_values(ts.decode(space.denormalize(x)))``.  Three spaces are in play:

* **Optimizer space** — the ``n_variables`` independent parameters in normalized
  coordinates; the domain the optimizer directly controls.
* **Physical space** — the full parameter set in original units, owned by
  ``ParameterSpace``.
* **Dependency manifold** — the surface defined by dependency transforms; not
  representable as a linear constraint and resolved only at write time.

Coordinate convention
---------------------
Normalization is per-parameter.  ``RangedParameter`` instances with an active
normalizer (``normalization is not None``) and finite bounds are mapped to ``[0, 1]``.
Parameters without normalization pass through the identity chart (``NullNormalizer``)
and retain their physical-unit value.  The optimizer space is therefore a
**mixed-coordinate product**: some axes are unit-scaled, others are in physical units.
There is no canonical norm or distance in optimizer space; Euclidean geometry does not
apply across axes.

Linear constraint transformation
---------------------------------
Constraints are defined over the optimizer basis (independent parameters only).
Dependent parameters are excluded: lifting constraints through the dependency embedding
is not defined in the linear constraint formalism.  Feasibility of dependent parameters
is enforced after embedding in ``ParameterSpace.set_values`` via parameter validation.

The affine transform ``x_phys = lb + span * x_norm`` (where ``span = ub - lb``) holds
only for parameters with an active affine normalizer.  Parameters without normalization
pass through unchanged.  ``TransformedSpace`` raises at constraint assembly time when a
constrained parameter uses a non-affine normalizer (e.g. ``LogNormalizer``); use
``ParameterSpace`` directly with physical-unit coordinates in that case.

As a result, constraint satisfaction in optimizer space is not equivalent to physical
feasibility when nonlinear normalizers or dependency transforms are involved.  Sampling
must be followed by post-hoc validation via ``ParameterSpace.set_values``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from CADETProcess.parameter_space.constraints import (
    LinearConstraint,
    LinearEqualityConstraint,
)
from CADETProcess.parameter_space.parameters import ParameterBase, RangedParameter
from CADETProcess.parameter_space.space import ParameterSpace

__all__ = ["TransformedSpace"]


class TransformedSpace:
    """Optimizer-facing view of a ``ParameterSpace`` in normalized coordinates.

    All bounds and constraint matrices are expressed in the normalized
    coordinate system.  The underlying ``ParameterSpace`` is the source of
    truth; ``TransformedSpace`` derives everything from it lazily.  Every
    method is a pure function: writing goes through ``ParameterSpace``
    exclusively.

    Parameters
    ----------
    space : ParameterSpace
        The physical-unit parameter space this view wraps.

    Examples
    --------
    ::

        space = ParameterSpace()
        space.add_evaluation_object(process)
        space.add_parameter(
            RangedParameter("length", float, lb=0.1, ub=1.0, normalization="linear"),
            path="column.length",
        )
        ts = TransformedSpace(space)
        # 0.5 normalized → 0.55 physical
        space.set_values(ts.decode(space.denormalize([0.5])))
    """

    def __init__(self, space: ParameterSpace) -> None:
        self._space = space

    @property
    def space(self) -> ParameterSpace:
        """The underlying ``ParameterSpace``."""
        return self._space

    # ── Delegation ────────────────────────────────────────────────────────────

    @property
    def evaluation_objects(self) -> list[Any]:
        """Registered evaluation objects (delegates to the underlying space)."""
        return self._space.evaluation_objects

    @property
    def parameters(self) -> list[ParameterBase]:
        """All registered parameters (delegates to the underlying space)."""
        return self._space.parameters

    @property
    def independent_parameters(self) -> list[ParameterBase]:
        """Independent (optimizer-facing) parameters."""
        return self._space.independent_parameters

    @property
    def n_variables(self) -> int:
        """Number of independent (optimizer-facing) variables."""
        return self._space.n_variables

    # ── Normalized bounds ─────────────────────────────────────────────────────

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds in the optimizer coordinate system.

        Each value is the per-parameter normalization of the physical lower bound.
        Parameters with an active normalizer and finite bounds return 0; parameters
        without normalization pass through the identity chart and retain their
        physical-unit value.  The result is a mixed-coordinate vector: axes are not
        globally comparable.
        """
        return self._space.normalize(self._space.lower_bounds_independent)

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds in the optimizer coordinate system.

        Each value is the per-parameter normalization of the physical upper bound.
        Parameters with an active normalizer and finite bounds return 1; parameters
        without normalization pass through the identity chart and retain their
        physical-unit value.  The result is a mixed-coordinate vector: axes are not
        globally comparable.
        """
        return self._space.normalize(self._space.upper_bounds_independent)

    # ── Vectorization ─────────────────────────────────────────────────────────

    @property
    def _numeric_independent_parameters(self) -> list[RangedParameter]:
        """Independent numeric parameters, in registration order."""
        return [
            p for p in self._space.independent_parameters
            if isinstance(p, RangedParameter)
        ]

    def encode(self, assignment: Mapping[str, Any]) -> np.ndarray:
        """Project a named assignment onto the numeric parameter vector.

        The vector spans the independent numeric parameters in registration
        order.  Dependent and categorical parameters have no vector position;
        their entries are dropped (``encode`` is a lossy projection).

        Parameters
        ----------
        assignment : Mapping
            Named values in physical units.  Must contain every independent
            numeric parameter; registered dependent or categorical names are
            ignored.

        Returns
        -------
        np.ndarray
            Values of the independent numeric parameters, in physical units.

        Raises
        ------
        ValueError
            If the assignment contains unknown names or misses an independent
            numeric parameter.
        """
        known = {p.name for p in self._space.parameters}
        unknown = [name for name in assignment if name not in known]
        if unknown:
            raise ValueError(f"Unknown parameter names: {unknown!r}.")
        numeric = self._numeric_independent_parameters
        missing = [p.name for p in numeric if p.name not in assignment]
        if missing:
            raise ValueError(
                f"Assignment misses independent numeric parameters: {missing!r}."
            )
        return np.array([float(assignment[p.name]) for p in numeric])

    def decode(
        self,
        x_num: npt.ArrayLike,
        categorical_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Embed a numeric parameter vector into a named assignment.

        The returned assignment carries canonical Python types: ``int`` for
        integer parameters (rounded), ``float`` for continuous ones, plus the
        caller-supplied categorical values.  When the space contains
        categorical parameters, the assignment cannot be reconstructed from
        the vector alone; *categorical_values* must supply every categorical
        parameter.

        Parameters
        ----------
        x_num : array-like
            Values of the independent numeric parameters in **physical** units,
            in registration order.
        categorical_values : Mapping, optional
            Values for the categorical parameters.  Required when the space
            contains categorical parameters; must cover exactly those.

        Returns
        -------
        dict
            Named physical assignment of the independent parameters, ordered
            by registration.

        Raises
        ------
        ValueError
            If the vector length does not match the number of independent
            numeric parameters, if *categorical_values* contains unknown names,
            or if it misses a categorical parameter (including the case where
            the space has categorical parameters and *categorical_values* is
            None).
        """
        numeric = self._numeric_independent_parameters
        x = np.asarray(x_num, dtype=float).ravel()
        if x.size != len(numeric):
            raise ValueError(
                f"Expected {len(numeric)} numeric values, got {x.size}."
            )
        categorical = self._space.categorical_parameters
        categorical_names = {p.name for p in categorical}
        if categorical_values is None:
            categorical_values = {}
        unknown = [n for n in categorical_values if n not in categorical_names]
        if unknown:
            raise ValueError(f"Unknown categorical parameter names: {unknown!r}.")
        missing = [p.name for p in categorical if p.name not in categorical_values]
        if missing:
            raise ValueError(
                f"Values for categorical parameters {missing!r} are required; "
                "they have no position in the numeric vector."
            )
        numeric_values = {
            p.name: int(np.round(v)) if p.parameter_type is int else float(v)
            for p, v in zip(numeric, x)
        }
        return {
            p.name: (
                categorical_values[p.name]
                if p.name in categorical_names
                else numeric_values[p.name]
            )
            for p in self._space.independent_parameters
        }

    # ── Constraint matrices ───────────────────────────────────────────────────

    def _assemble_matrices(
        self,
        constraints: list[LinearConstraint] | list[LinearEqualityConstraint],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A, b)`` in normalized coordinates.

        Parameters
        ----------
        constraints : list
            List of ``LinearConstraint`` or ``LinearEqualityConstraint`` objects.

        Returns
        -------
        A : np.ndarray, shape (n_constraints, n_variables)
        b : np.ndarray, shape (n_constraints,)

        Raises
        ------
        ValueError
            If any constrained parameter uses a non-linear normalizer.
        """
        n = self._space.n_variables
        n_c = len(constraints)

        if n_c == 0:
            return np.zeros((0, n)), np.zeros(0)

        param_index = {
            p.name: i for i, p in enumerate(self._space.independent_parameters)
        }
        lb_phys = self._space.lower_bounds_independent
        ub_phys = self._space.upper_bounds_independent

        A = np.zeros((n_c, n))
        b = np.empty(n_c)

        for row_idx, constraint in enumerate(constraints):
            b_adj = 0.0
            for p, coeff in zip(constraint.parameters, constraint.lhs):
                if p.name not in param_index:
                    # Dependent parameters are not part of the optimizer basis;
                    # feasibility is enforced after embedding in set_values.
                    continue
                i = param_index[p.name]
                p_lb = lb_phys[i]
                p_ub = ub_phys[i]
                if (
                    isinstance(p, RangedParameter)
                    and np.isfinite(p_lb)
                    and np.isfinite(p_ub)
                    and p.normalization is not None
                ):
                    if not p.normalizer.is_linear:
                        raise ValueError(
                            f"Parameter {p.name!r} uses a non-affine normalizer. "
                            "Linear constraints involving non-affinely-normalized "
                            "parameters are not linearly representable in normalized "
                            "coordinates and cannot be assembled by TransformedSpace. "
                            "Use ParameterSpace directly with physical-unit coordinates."
                        )
                    span = p_ub - p_lb
                    A[row_idx, i] += coeff * span
                    b_adj += coeff * p_lb
                else:
                    A[row_idx, i] += coeff
            b[row_idx] = constraint.b - b_adj

        return A, b

    @property
    def A(self) -> np.ndarray:
        """Inequality constraint matrix in normalized coordinates, shape ``(m, n)``."""
        return self._assemble_matrices(self._space.linear_constraints)[0]

    @property
    def b(self) -> np.ndarray:
        """Inequality constraint RHS in normalized coordinates, shape ``(m,)``."""
        return self._assemble_matrices(self._space.linear_constraints)[1]

    @property
    def A_eq(self) -> np.ndarray:
        """Equality constraint matrix in normalized coordinates, shape ``(m, n)``."""
        return self._assemble_matrices(self._space.linear_equality_constraints)[0]

    @property
    def b_eq(self) -> np.ndarray:
        """Equality constraint RHS in normalized coordinates, shape ``(m,)``."""
        return self._assemble_matrices(self._space.linear_equality_constraints)[1]

    # ── Validate ──────────────────────────────────────────────────────────────

    def get_dependent_values(self, x: npt.ArrayLike) -> np.ndarray:
        """Expand normalized independent values to the full physical parameter vector.

        Parameters
        ----------
        x : array-like
            Values for the ``n_variables`` independent parameters in **normalized**
            coordinates.

        Returns
        -------
        np.ndarray
            Full physical parameter vector of length ``n_parameters``.
        """
        return self._space.get_dependent_values(x, denormalize=True)

    def check_bounds(self, x: npt.ArrayLike, tol: float | npt.ArrayLike = 0.0) -> bool:
        """Return True when *x* (in normalized coordinates) satisfies physical bounds.

        Parameters
        ----------
        x : array-like
            Values for the independent parameters in **normalized** coordinates.
        tol : float or array-like
            Per-variable tolerance added to each bound before checking.
        """
        x_phys = self._space.denormalize(x)
        return self._space.check_bounds(x_phys, tol=tol, resolve_dependencies=True)

    def validate_x(
        self,
        x: npt.ArrayLike,
        tol: float = 0.0,
        tol_eq: float = 1e-6,
    ) -> bool | np.ndarray:
        """Return True if *x* (normalized) satisfies bounds and all linear constraints.

        Parameters
        ----------
        x : array-like
            Normalized independent parameter vector, shape ``(n_variables,)`` for a
            single point or ``(m, n_variables)`` for a population.
        tol : float
            Tolerance applied to bounds and inequality constraints (inclusive).
        tol_eq : float
            Tolerance applied to equality constraints.

        Returns
        -------
        bool or np.ndarray of bool
            Scalar for a single point; 1-D boolean array for a population.
        """
        x_arr = np.asarray(x, dtype=float)
        population = x_arr.ndim == 2
        rows = x_arr if population else x_arr[np.newaxis, :]
        results = [
            self._space.validate_x(self.get_dependent_values(row), tol=tol, tol_eq=tol_eq)
            for row in rows
        ]
        if population:
            return np.array(results)
        return results[0]

    # ── Misc ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a readable representation."""
        return f"TransformedSpace({self._space!r})"
