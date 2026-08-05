"""
Sampler strategies for ParameterSpace.

SamplerBase defines the postprocessing contract (significant-digits snap, integer
rounding via decode, categorical merge, dependency resolution, validation, and
rejection of candidates violating linear inequality constraints on the resolved
values).
Concrete backends implement _candidates to produce numeric candidate rows,
optionally with unit-interval columns for stratified categorical coverage, and
declare via _sequential_candidates whether the row order carries structure.
"""

from __future__ import annotations

import math
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

    The polytope handed to hopsy lives in unit-box coordinates (see
    ``HopsySampler._candidates``), so the callback first recovers the physical
    value ``x = lb + scale * u``.  The correction is a property of the point,
    not of the coordinates naming it: the constant scale factor cancels in the
    normalization, so the target density is the one that would be obtained
    sampling ``x`` directly.
    """

    def __init__(
        self, log_indices: list[int], lb: np.ndarray, scale: np.ndarray
    ) -> None:
        self.log_space_indices = log_indices
        self.lb = lb
        self.scale = scale

    def compute_negative_log_likelihood(self, u: np.ndarray) -> float:
        x = self.lb + self.scale * u
        return float(np.sum(np.log(x[self.log_space_indices])))


class SamplerBase(ABC):
    """Abstract sampler strategy.

    Concrete subclasses implement _candidates; this base class handles
    postprocessing: significant-digits snap, integer rounding (via decode),
    categorical merge, dependency resolution, validation, and rejection of
    candidates that violate linear inequality constraints on the resolved
    values (constraints referencing dependent parameters cannot be part of
    the candidate polytope, and the snap and integer rounding perturb
    candidates after polytope enforcement).
    """

    #: When True, candidates are consumed in row order because the order
    #: carries structure (QMC designs).  When False, candidates are consumed
    #: in random order without replacement (MCMC pools, where sequential
    #: consumption would surface chain autocorrelation).
    _sequential_candidates = False

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
            # 32-bit range: a narrow default (e.g. 0..255) makes unseeded
            # calls collide and silently repeat "random" populations
            seed = _random.randint(0, 2**32 - 1)

        independent = space.independent_parameters
        numeric = [p for p in independent if isinstance(p, RangedParameter)]
        categorical = space.categorical_parameters

        if numeric and any(np.isinf(p.lb) or np.isinf(p.ub) for p in numeric):
            raise ValueError(
                "Cannot sample a space with unbounded parameters. "
                "Narrow lb/ub on all parameters before sampling."
            )

        if n <= 0:
            return []

        candidates = self._candidates(space, n, seed)
        pool_size = candidates.shape[0]
        n_numeric = len(numeric)
        # Backends may append one unit-interval column per categorical
        # parameter (QMC designs stratify categories); otherwise categories
        # are drawn uniformly at random per candidate.
        stratified_categoricals = (
            bool(categorical)
            and candidates.shape[1] == n_numeric + len(categorical)
        )

        ts = space.transformed_space
        rng = np.random.default_rng(seed)

        # All inequality constraints are re-checked on the resolved values:
        # constraints referencing dependent parameters cannot be expressed in
        # the independent-variable polytope, and the significant-digits snap
        # and integer rounding perturb candidates after polytope enforcement.
        inequality_constraints = space._linear_constraints

        order = (
            np.arange(pool_size)
            if self._sequential_candidates
            else rng.permutation(pool_size)
        )
        results = []

        for idx in order:
            row = candidates[int(idx)]
            if stratified_categoricals:
                categorical_values = {
                    c.name: c.valid_values[
                        min(int(u * len(c.valid_values)), len(c.valid_values) - 1)
                    ]
                    for c, u in zip(categorical, row[n_numeric:])
                }
            else:
                categorical_values = {
                    c.name: c.valid_values[int(rng.integers(len(c.valid_values)))]
                    for c in categorical
                } or None
            assignment = ts.decode(row[:n_numeric], categorical_values)

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

            if any(
                sum(coeff * all_values[p.name] for p, coeff in zip(c.parameters, c.lhs))
                > c.b
                for c in inequality_constraints
            ):
                continue

            results.append(all_values if include_dependent else assignment)
            if len(results) == n:
                break

        if len(results) < n:
            raise ValueError(
                f"Only {len(results)} of {n} requested samples are feasible within "
                f"the {pool_size} candidates. "
                "Increase pool_size or relax dependent-parameter constraints."
            )

        return results

    @abstractmethod
    def _candidates(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        """Return float array of shape (pool_size, n_numeric_independent).

        *n* is the number of feasible samples requested; backends whose
        candidate count must match the request (QMC designs) size the pool
        from it, pool-based backends may ignore it.

        Backends that stratify categorical parameters append one column per
        categorical parameter (in ``space.categorical_parameters`` order)
        holding unit-interval values; SamplerBase maps each value *u* to
        category index ``floor(u * n_categories)``.
        """


class HopsySampler(SamplerBase):
    """Uniform polytope sampler using hopsy (Hit-and-Run MCMC).

    The only correct choice when linear inequality or equality constraints are
    present. Uses a log-space model for parameters with nonlinear normalizers.

    The polytope is built in unit-box coordinates and the draws are mapped back
    to physical units, so bounds of any magnitude sample correctly; hopsy's
    rounding step otherwise rejects narrow-in-absolute-units spaces (e.g.
    ``ub=1e-9``) as degenerate.

    Inequality constraints referencing dependent parameters are excluded from
    the polytope (a relaxation); SamplerBase enforces them by rejection.
    Equality constraints referencing dependent parameters raise, because an
    equality on continuous values cannot be met by rejection sampling.
    """

    def __init__(self, pool_size: int = 100_000) -> None:
        self.pool_size = pool_size

    def _candidates(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:  # noqa: ARG002
        independent = space.independent_parameters
        numeric = [p for p in independent if isinstance(p, RangedParameter)]
        numeric_idx = [
            i for i, p in enumerate(independent) if isinstance(p, RangedParameter)
        ]

        if not numeric:
            return np.zeros((self.pool_size, 0))

        ind_names = {p.name for p in independent}
        if any(
            any(p.name not in ind_names for p in c.parameters)
            for c in space._linear_equality_constraints
        ):
            raise ValueError(
                "Linear equality constraints referencing dependent parameters "
                "cannot be enforced by sampling; express the constraint in "
                "independent parameters instead."
            )
        independent_rows = np.array(
            [
                all(p.name in ind_names for p in c.parameters)
                for c in space._linear_constraints
            ],
            dtype=bool,
        )

        lb_num = np.array([p.lb for p in numeric], dtype=float)
        ub_num = np.array([p.ub for p in numeric], dtype=float)

        # Sample in unit-box coordinates u = (x - lb) / (ub - lb) rather than in
        # physical units.  PolyRound rejects any polytope narrower than an
        # absolute threshold (thresh / accepted_tol_violation = 1e-9) as
        # degenerate, so a legitimate but small-scale bound (ub=1e-9) fails to
        # round.  The map is a shift and a per-parameter scale, so linear
        # constraints stay linear and transform exactly; it is deliberately
        # independent of each parameter's declared normalizer, since a
        # nonlinear one would bend constraints into curves the polytope cannot
        # represent.  Draws are mapped back to physical units before returning.
        width = ub_num - lb_num
        scale = np.where(width > 0, width, 1.0)
        # 1.0 for a normal parameter, 0.0 for a degenerate one (lb == ub),
        # which keeps the point-polytope a point instead of widening it.
        ub_unit = width / scale

        log_indices = [
            i for i, p in enumerate(numeric) if not p.normalizer.is_linear
        ]
        model = (
            _LogSpaceModel(log_indices, lb_num, scale) if log_indices else None
        )

        A = space.A_independent[independent_rows][:, numeric_idx]
        problem = hopsy.Problem(
            A * scale,
            space.b[independent_rows] - A @ lb_num,
            model,
        )
        problem = hopsy.add_box_constraints(
            problem, np.zeros_like(lb_num), ub_unit, simplify=False
        )
        if space._linear_equality_constraints:
            A_eq = space.A_eq_independent[:, numeric_idx]
            problem = hopsy.add_equality_constraints(
                problem, A_eq * scale, space.b_eq - A_eq @ lb_num
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
        # back to physical units; shape (pool_size, n_numeric)
        return lb_num + states[0] * scale


class LatinHypercubeSampler(SamplerBase):
    """Latin Hypercube sampler via scipy.stats.qmc.

    Generates exactly *n* points, so the one-point-per-stratum property holds
    for the returned set; a subset of a larger design would not be a Latin
    Hypercube. Categorical parameters are stratified as extra design
    dimensions, giving balanced category counts. Box-bounded spaces only:
    raises if linear constraints are present, and any candidate rejected
    during postprocessing exhausts the pool (a filtered design loses
    stratification). Use HopsySampler for spaces that require rejection.
    """

    _sequential_candidates = True

    def _candidates(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        return _qmc_candidates(space, n, seed, "lhs")


class SobolSampler(SamplerBase):
    """Sobol sequence sampler via scipy.stats.qmc.

    Generates the first ``2**ceil(log2(n))`` points of a scrambled Sobol
    sequence and consumes them in order; prefixes of the sequence retain low
    discrepancy, so a few postprocessing rejections are tolerated. Categorical
    parameters are covered as extra sequence dimensions, giving balanced
    category counts over full ``2**m`` blocks. Box-bounded spaces only: raises
    if linear constraints are present. Use HopsySampler when linear
    constraints are present.
    """

    _sequential_candidates = True

    def _candidates(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        return _qmc_candidates(space, n, seed, "sobol")


def _qmc_candidates(
    space: ParameterSpace,
    n: int,
    seed: int,
    kind: str,
) -> np.ndarray:
    """Generate QMC candidates in physical units (box bounds only).

    Categorical parameters are included as extra unit-interval design
    dimensions (one column per categorical parameter, appended after the
    numeric columns), so category coverage inherits the design's uniformity
    instead of being drawn independently at random.
    """
    from scipy.stats.qmc import LatinHypercube, Sobol

    independent = space.independent_parameters
    numeric = [p for p in independent if isinstance(p, RangedParameter)]
    categorical = space.categorical_parameters

    if space._linear_constraints or space._linear_equality_constraints:
        raise ValueError(
            f"{kind.upper()} sampler requires a box-bounded space with no linear "
            "constraints. Use HopsySampler when linear constraints are present."
        )

    d = len(numeric) + len(categorical)
    if d == 0:
        return np.zeros((n, 0))

    if kind == "lhs":
        engine = LatinHypercube(d=d, seed=seed)
        unit = engine.random(n=n)
    else:
        engine = Sobol(d=d, seed=seed, scramble=True)
        unit = engine.random_base2(m=max(0, math.ceil(math.log2(n))))

    lb = np.array([p.lb for p in numeric], dtype=float)
    ub = np.array([p.ub for p in numeric], dtype=float)
    scaled = unit.copy()
    scaled[:, : len(numeric)] = lb + unit[:, : len(numeric)] * (ub - lb)
    return scaled


def chebyshev_center(space: ParameterSpace) -> dict[str, float | int]:
    """Compute the Chebyshev center of the independent-variable polytope.

    Returns an independent-variable named assignment in physical units,
    typed and snapped the same way as ``SamplerBase.sample()`` (``int`` for
    integer parameters, significant-digits rounding where declared).

    Linear inequality constraints referencing dependent parameters cannot be
    part of the polytope (dependencies are arbitrary callables); they are
    dropped with a warning and the center of the relaxed polytope is verified
    against them a posteriori.

    Raises
    ------
    ValueError
        If the space contains categorical parameters (a center is undefined
        for them), if any independent parameter is unbounded, if a linear
        equality constraint references a dependent parameter (a dropped
        equality can never survive a point check), or if the relaxed center
        violates a dropped inequality constraint.
    """
    from packaging.version import Version

    if space.categorical_parameters:
        raise ValueError(
            "The Chebyshev center is undefined for spaces with categorical "
            "parameters. Compute it per category or remove the categorical "
            "parameters."
        )

    if any(
        np.isinf(p.lb) or np.isinf(p.ub) for p in space.independent_parameters
    ):
        raise ValueError(
            "Cannot compute the Chebyshev center of an unbounded space. "
            "Narrow lb/ub on all parameters first."
        )

    ind_names = {p.name for p in space.independent_parameters}
    if any(
        any(p.name not in ind_names for p in c.parameters)
        for c in space._linear_equality_constraints
    ):
        raise ValueError(
            "Linear equality constraints referencing dependent parameters "
            "cannot be included in the Chebyshev polytope; express the "
            "constraint in independent parameters instead."
        )
    keep = np.array(
        [
            all(p.name in ind_names for p in c.parameters)
            for c in space._linear_constraints
        ],
        dtype=bool,
    )
    dropped = [c for c, k in zip(space._linear_constraints, keep) if not k]
    if dropped:
        warnings.warn(
            f"{len(dropped)} linear constraint(s) reference dependent "
            "parameters and cannot be included in the Chebyshev polytope; "
            "computing the center of the relaxed polytope and verifying it "
            "against the full constraints."
        )

    problem = hopsy.Problem(space.A_independent[keep], space.b[keep])
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

    assignment = space.transformed_space.decode(center)
    for p in space.independent_parameters:
        if p.significant_digits is not None:
            assignment[p.name] = float(
                round_to_significant_digits(assignment[p.name], p.significant_digits)
            )

    if dropped:
        all_values = space.resolve(assignment)
        violated = [
            c for c in dropped
            if sum(coeff * all_values[p.name] for p, coeff in zip(c.parameters, c.lhs))
            > c.b
        ]
        if violated:
            raise ValueError(
                "The center of the relaxed polytope violates "
                f"{len(violated)} linear constraint(s) referencing dependent "
                "parameters. Express the constraint in independent parameters, "
                "or use create_initial_values / sampling for a feasible "
                "starting point."
            )

    return assignment
