"""
Sampler strategies for ParameterSpace.

SamplerBase defines the postprocessing contract (significant-digits snap, integer
rounding via decode, categorical merge, dependency resolution, validation).
Concrete backends implement _candidates to produce numeric candidate rows.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import hopsy
import numpy as np

from CADETProcess.numerics import round_to_significant_digits
from CADETProcess.parameter_space.parameters import RangedParameter

if TYPE_CHECKING:
    from CADETProcess.parameter_space.space import ParameterSpace


class _LogSpaceModel:
    """hopsy log-space model for parameters with nonlinear normalizers (e.g. LogNormalizer).

    Injects a Jacobian correction so hopsy draws are uniform in transformed space.
    """

    def __init__(self, log_indices: list[int]) -> None:
        self.log_space_indices = log_indices

    def compute_negative_log_likelihood(self, x: np.ndarray) -> float:
        return float(np.sum(np.log(x[self.log_space_indices])))


class SamplerBase(ABC):
    """Abstract sampler strategy.

    Concrete subclasses implement _candidates; this base class handles
    postprocessing: significant-digits snap, integer rounding (via decode),
    categorical merge, dependency resolution, and validation.
    """

    def sample(
        self,
        space: ParameterSpace,
        n: int,
        seed: Optional[int] = None,
        include_dependent: bool = False,
    ) -> list[dict[str, Any]]:
        """Draw n feasible samples as named assignments."""
        import random as _random

        if seed is None:
            seed = _random.randint(0, 255)

        independent = space.independent_parameters
        numeric = [p for p in independent if isinstance(p, RangedParameter)]
        categorical = space.categorical_parameters

        if numeric and any(np.isinf(p.lb) or np.isinf(p.ub) for p in numeric):
            raise ValueError(
                "Cannot sample a space with unbounded parameters. "
                "Narrow lb/ub on all parameters before sampling."
            )

        candidates = self._candidates(space, seed)  # shape (pool_size, n_numeric)
        pool_size = candidates.shape[0]

        ts = space.transformed_space
        rng = np.random.default_rng(seed)
        results = []
        counter = 0

        while len(results) < n:
            if counter >= pool_size:
                raise ValueError(
                    f"Could not find {n} feasible samples after exhausting the "
                    f"{pool_size} candidates. "
                    "Increase pool_size or relax dependent-parameter constraints."
                )
            idx = int(rng.integers(0, pool_size))
            counter += 1

            categorical_values = {
                c.name: c.valid_values[int(rng.integers(len(c.valid_values)))]
                for c in categorical
            } or None
            assignment = ts.decode(candidates[idx], categorical_values)

            for p in numeric:
                if p.significant_digits is not None:
                    assignment[p.name] = float(
                        round_to_significant_digits(assignment[p.name], p.significant_digits)
                    )

            try:
                all_values = space.resolve(assignment)
                for p in space._parameters:
                    p.validate(all_values[p.name])
            except (TypeError, ValueError):
                continue

            results.append(all_values if include_dependent else assignment)

        return results

    @abstractmethod
    def _candidates(self, space: ParameterSpace, seed: int) -> np.ndarray:
        """Return float array of shape (pool_size, n_numeric_independent)."""


class HopsySampler(SamplerBase):
    """Uniform polytope sampler using hopsy (Hit-and-Run MCMC).

    The only correct choice when linear inequality or equality constraints are
    present. Uses a log-space model for parameters with nonlinear normalizers.
    """

    def __init__(self, pool_size: int = 100_000) -> None:
        self.pool_size = pool_size

    def _candidates(self, space: ParameterSpace, seed: int) -> np.ndarray:
        independent = space.independent_parameters
        numeric = [p for p in independent if isinstance(p, RangedParameter)]
        numeric_idx = [
            i for i, p in enumerate(independent) if isinstance(p, RangedParameter)
        ]

        if not numeric:
            return np.zeros((self.pool_size, 0))

        log_indices = [
            i for i, p in enumerate(numeric) if not p.normalizer.is_linear
        ]
        model = _LogSpaceModel(log_indices) if log_indices else None

        lb_num = np.array([p.lb for p in numeric], dtype=float)
        ub_num = np.array([p.ub for p in numeric], dtype=float)
        problem = hopsy.Problem(space.A_independent[:, numeric_idx], space.b, model)
        problem = hopsy.add_box_constraints(problem, lb_num, ub_num, simplify=False)
        if space._linear_equality_constraints:
            problem = hopsy.add_equality_constraints(
                problem, space.A_eq_independent[:, numeric_idx], space.b_eq
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = hopsy.round(problem, simplify=False)
            mc = hopsy.MarkovChain(
                problem, proposal=hopsy.UniformCoordinateHitAndRunProposal
            )
            rng_hopsy = hopsy.RandomNumberGenerator(seed=seed)
            _, states = hopsy.sample(
                mc, rng_hopsy, n_samples=self.pool_size, thinning=2
            )
        return states[0]  # shape (pool_size, n_numeric)


def chebyshev_center(space: ParameterSpace) -> np.ndarray:
    """Compute the Chebyshev center of the independent-variable polytope.

    Returns an independent-variable float vector in physical units.
    """
    from packaging.version import Version

    problem = hopsy.Problem(space.A_independent, space.b)
    problem = hopsy.add_box_constraints(
        problem,
        space.lower_bounds_independent,
        space.upper_bounds_independent,
        simplify=False,
    )
    if space._linear_equality_constraints:
        problem = hopsy.add_equality_constraints(
            problem, space.A_eq_independent, space.b_eq
        )
    center = hopsy.compute_chebyshev_center(problem, original_space=True)
    if Version(hopsy.__version__.strip('"')) < Version("1.7.0b"):
        center = center[:, 0]
    return center
