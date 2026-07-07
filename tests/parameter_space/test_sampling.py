from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.parameter_space import (
    ChoiceParameter,
    HopsySampler,
    LatinHypercubeSampler,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterSpace,
    RangedParameter,
    SobolSampler,
    chebyshev_center,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class Column:
    length: float = 0.1
    diameter: float = 0.01
    volume: float = 0.0


@pytest.fixture
def column():
    return Column()


BURN_IN = 2000  # small for fast tests


def _space_1d(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("length", float, lb=1.0, ub=10.0), path="length")
    return space


def _space_2d(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("length", float, lb=0.0, ub=5.0), path="length")
    space.add_parameter(RangedParameter("diameter", float, lb=0.0, ub=2.0), path="diameter")
    return space


# ── physical constraint matrices ──────────────────────────────────────────────


def test_A_b_no_constraints(column):
    space = _space_1d(column)
    assert space.A.shape == (0, 1)       # (n_constraints, n_parameters)
    assert space.b.shape == (0,)


def test_A_b_with_constraint(column):
    space = _space_2d(column)
    p1, p2 = space.independent_parameters
    space.add_linear_constraint(LinearConstraint([p1, p2], lhs=[1.0, 1.0], b=4.0))
    np.testing.assert_allclose(space.A, [[1.0, 1.0]])
    np.testing.assert_allclose(space.b, [4.0])


def test_A_eq_b_eq_no_constraints(column):
    space = _space_1d(column)
    assert space.A_eq.shape == (0, 1)
    assert space.b_eq.shape == (0,)


def test_A_includes_derived_parameter_column(column):
    # A has n_parameters columns (independent + derived)
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 2)
    space.add_linear_constraint(LinearConstraint([a, b], lhs=[1.0, 1.0], b=8.0))
    assert space.A.shape == (1, 2)         # both params
    assert space.A_independent.shape == (1, 1)  # only 'a'
    np.testing.assert_allclose(space.A, [[1.0, 1.0]])
    np.testing.assert_allclose(space.A_independent, [[1.0]])


# ── sample: assignments ───────────────────────────────────────────────────────


def test_sample_returns_independent_assignments(column):
    space = _space_1d(column)
    samples = space.sample(5, seed=0, pool_size=BURN_IN)
    assert len(samples) == 5
    assert all(list(s) == ["length"] for s in samples)


def test_sample_include_dependent_returns_full_assignments(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 2)
    samples = space.sample(3, seed=0, pool_size=BURN_IN, include_dependent=True)
    assert all(list(s) == ["a", "b"] for s in samples)


def test_sample_assignments_ordered_by_registration(column):
    space = _space_2d(column)
    samples = space.sample(10, seed=1, pool_size=BURN_IN)
    assert all(list(s) == ["length", "diameter"] for s in samples)


# ── sample: values within bounds ─────────────────────────────────────────────


def test_sample_values_within_bounds(column):
    space = _space_2d(column)
    samples = space.sample(20, seed=2, pool_size=BURN_IN)
    assert all(0.0 <= s["length"] <= 5.0 for s in samples)
    assert all(0.0 <= s["diameter"] <= 2.0 for s in samples)


def test_sample_1d_within_bounds(column):
    space = _space_1d(column)
    samples = space.sample(10, seed=3, pool_size=BURN_IN)
    assert all(1.0 <= s["length"] <= 10.0 for s in samples)


# ── sample: derived parameters ────────────────────────────────────────────────


def test_sample_derived_value_correct(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 2)
    samples = space.sample(5, seed=4, pool_size=BURN_IN, include_dependent=True)
    np.testing.assert_allclose(
        [s["b"] for s in samples], [s["a"] * 2 for s in samples]
    )


def test_sample_derived_infeasible_filtered(column):
    # b = a * 3, but b.ub = 6, so only a < 2 is accepted
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=6.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 3)
    samples = space.sample(5, seed=5, pool_size=BURN_IN, include_dependent=False)
    # all independent values must satisfy a * 3 <= 6 → a <= 2
    assert all(s["a"] <= 2.0 + 1e-9 for s in samples)


# ── sample: typed parameters ──────────────────────────────────────────────────


def test_sample_integer_values_are_whole_numbers(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("n", int, lb=1, ub=100), path="length")
    samples = space.sample(10, seed=6, pool_size=BURN_IN)
    assert all(type(s["n"]) is int for s in samples)
    assert all(1 <= s["n"] <= 100 for s in samples)


def test_sample_categorical_draws_from_valid_values(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=5.0), path="length"
    )
    space.add_parameter(ChoiceParameter("mode", ["gradient", "isocratic"]))
    samples = space.sample(20, seed=7, pool_size=BURN_IN)
    assert all(s["mode"] in ("gradient", "isocratic") for s in samples)
    assert all(list(s) == ["length", "mode"] for s in samples)
    # both categories appear over 20 draws (deterministic for the fixed seed)
    assert {s["mode"] for s in samples} == {"gradient", "isocratic"}


def test_sample_categorical_only_numeric_dimensions_in_polytope(column):
    # A purely categorical space has an empty numeric polytope; the draw
    # must still produce valid assignments.
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(ChoiceParameter("mode", ["a", "b"]))
    samples = space.sample(5, seed=8, pool_size=100)
    assert all(s["mode"] in ("a", "b") for s in samples)


# ── sample: reproducibility ───────────────────────────────────────────────────


def test_sample_same_seed_reproducible(column):
    space = _space_1d(column)
    s1 = space.sample(5, seed=42, pool_size=BURN_IN)
    s2 = space.sample(5, seed=42, pool_size=BURN_IN)
    assert s1 == s2


# ── sample: exhausted budget raises ───────────────────────────────────────────


def test_sample_exhausted_budget_raises(column):
    # Impossible constraint: b = a * 100 but b.ub = 0.01, so nothing is feasible
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=1.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=0.01)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 100)
    with pytest.raises(ValueError, match="pool_size"):
        space.sample(1, seed=0, pool_size=50)


# ── HopsySampler: direct use ──────────────────────────────────────────────────


def test_hopsy_sampler_direct_use(column):
    space = _space_2d(column)
    sampler = HopsySampler(pool_size=BURN_IN)
    samples = sampler.sample(space, 5, seed=0)
    assert len(samples) == 5
    assert all(0.0 <= s["length"] <= 5.0 for s in samples)
    assert all(0.0 <= s["diameter"] <= 2.0 for s in samples)


def test_hopsy_sampler_same_result_as_space_sample(column):
    space = _space_1d(column)
    sampler = HopsySampler(pool_size=BURN_IN)
    direct = sampler.sample(space, 5, seed=99)
    via_space = space.sample(5, seed=99, pool_size=BURN_IN)
    assert direct == via_space


def test_sample_candidates_drawn_without_replacement(column):
    space = _space_1d(column)
    samples = space.sample(30, seed=2, pool_size=50)
    values = [s["length"] for s in samples]
    assert len(set(values)) == len(values)


# ── linear constraints referencing dependent parameters ──────────────────────


def test_sample_enforces_dependent_linear_constraint_by_rejection(column):
    # a + b <= 4 with b = a means a <= 2; the polytope only sees the
    # independent column (a <= 4), so rejection must enforce the rest
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x)
    space.add_linear_constraint(LinearConstraint([a, b], lhs=[1.0, 1.0], b=4.0))
    samples = space.sample(20, seed=0, pool_size=BURN_IN)
    assert all(2 * s["a"] <= 4.0 + 1e-9 for s in samples)


def test_sample_dependent_linear_constraint_not_overtightened(column):
    # a + b <= 0 with b = -a holds everywhere; slicing the dependent column
    # away would wrongly enforce a <= 0 and reject the entire box
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=-5.0, ub=0.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: -x)
    space.add_linear_constraint(LinearConstraint([a, b], lhs=[1.0, 1.0], b=0.0))
    samples = space.sample(20, seed=1, pool_size=BURN_IN)
    assert len(samples) == 20
    assert any(s["a"] > 2.0 for s in samples)


def test_sample_integer_rounding_cannot_violate_linear_constraint(column):
    # candidates in (3.5, 3.6] satisfy the polytope but round to 4; the
    # resolved-value re-check must reject them, since the polytope only
    # constrains the pre-rounding value
    space = ParameterSpace()
    space.add_evaluation_object(column)
    n = RangedParameter("n", int, lb=1, ub=10)
    space.add_parameter(n, path="length")
    space.add_linear_constraint(LinearConstraint([n], lhs=[1.0], b=3.6))
    samples = space.sample(20, seed=0, pool_size=BURN_IN)
    assert all(s["n"] <= 3.6 for s in samples)


def test_sample_raises_on_dependent_equality_constraint(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x)
    space.add_linear_equality_constraint(
        LinearEqualityConstraint([a, b], lhs=[1.0, -1.0], b=0.0)
    )
    with pytest.raises(ValueError, match="dependent"):
        space.sample(1, seed=0, pool_size=50)


# ── unbounded guard ───────────────────────────────────────────────────────────


def test_sample_raises_on_unbounded_parameter(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=float("inf")), path="length"
    )
    with pytest.raises(ValueError, match="unbounded"):
        space.sample(1, seed=0, pool_size=50)


# ── significant-digits snap ───────────────────────────────────────────────────


def test_sample_significant_digits_snap(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.001, ub=0.999, significant_digits=2),
        path="length",
    )
    samples = space.sample(20, seed=0, pool_size=BURN_IN)
    for s in samples:
        v = s["length"]
        # After snapping to 2 significant digits the value must equal itself re-rounded
        import math
        if v != 0.0:
            magnitude = 10 ** (math.floor(math.log10(abs(v))) - 1)
            assert abs(v - round(v / magnitude) * magnitude) < 1e-12 * abs(v) + 1e-15


# ── LHS and Sobol samplers ────────────────────────────────────────────────────


@pytest.mark.parametrize("SamplerClass", [LatinHypercubeSampler, SobolSampler])
def test_qmc_bounds_containment(column, SamplerClass):
    space = _space_2d(column)
    samples = SamplerClass().sample(space, 20, seed=0)
    assert all(0.0 <= s["length"] <= 5.0 for s in samples)
    assert all(0.0 <= s["diameter"] <= 2.0 for s in samples)


@pytest.mark.parametrize("SamplerClass", [LatinHypercubeSampler, SobolSampler])
def test_qmc_seed_determinism(column, SamplerClass):
    space = _space_1d(column)
    sampler = SamplerClass()
    s1 = sampler.sample(space, 10, seed=7)
    s2 = sampler.sample(space, 10, seed=7)
    assert s1 == s2


@pytest.mark.parametrize("SamplerClass", [LatinHypercubeSampler, SobolSampler])
def test_qmc_raises_on_linear_constraints(column, SamplerClass):
    space = _space_2d(column)
    p1, p2 = space.independent_parameters
    space.add_linear_constraint(LinearConstraint([p1, p2], lhs=[1.0, 1.0], b=4.0))
    with pytest.raises(ValueError, match="linear constraints"):
        SamplerClass().sample(space, 5, seed=0)


@pytest.mark.parametrize("SamplerClass", [LatinHypercubeSampler, SobolSampler])
def test_qmc_integer_and_categorical(column, SamplerClass):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("n", int, lb=1, ub=10), path="length")
    space.add_parameter(ChoiceParameter("mode", ["a", "b"]))
    samples = SamplerClass().sample(space, 10, seed=0)
    assert all(type(s["n"]) is int for s in samples)
    assert all(s["mode"] in ("a", "b") for s in samples)


def test_lhs_returned_set_is_stratified(column):
    # the defining LHS property: n samples, exactly one per axis-aligned
    # stratum; a random subset of a larger design would fail this
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("length", float, lb=0.0, ub=8.0), path="length")
    samples = LatinHypercubeSampler().sample(space, 8, seed=3)
    strata = sorted(int(np.floor(s["length"])) for s in samples)
    assert strata == list(range(8))


def test_sobol_samples_are_sequence_prefix(column):
    # sequential consumption: a smaller request must be a prefix of a larger
    # one for the same seed, which random pool draws would not satisfy
    space = _space_1d(column)
    s4 = SobolSampler().sample(space, 4, seed=5)
    s8 = SobolSampler().sample(space, 8, seed=5)
    assert s8[:4] == s4


def test_sobol_emits_no_balance_warning(column):
    import warnings

    space = _space_1d(column)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        SobolSampler().sample(space, 10, seed=0)


def _categorical_space(column, categories):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=5.0), path="length"
    )
    space.add_parameter(ChoiceParameter("mode", categories))
    return space


def test_lhs_categorical_counts_are_balanced(column):
    # categoricals are extra design dimensions: 10 samples over 2 categories
    # must split 5/5, whereas an independent random draw would not
    space = _categorical_space(column, ["a", "b"])
    samples = LatinHypercubeSampler().sample(space, 10, seed=0)
    modes = [s["mode"] for s in samples]
    assert modes.count("a") == 5
    assert modes.count("b") == 5


def test_sobol_categorical_counts_balanced_over_full_block(column):
    # 8 = 2**3 samples over 2 categories: Sobol base-2 balance gives 4/4
    space = _categorical_space(column, ["a", "b"])
    samples = SobolSampler().sample(space, 8, seed=0)
    modes = [s["mode"] for s in samples]
    assert modes.count("a") == 4
    assert modes.count("b") == 4


def test_lhs_categorical_only_space_is_balanced(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(ChoiceParameter("mode", ["a", "b", "c"]))
    samples = LatinHypercubeSampler().sample(space, 9, seed=1)
    modes = [s["mode"] for s in samples]
    assert sorted(set(modes)) == ["a", "b", "c"]
    assert all(modes.count(m) == 3 for m in "abc")


# ── chebyshev center ──────────────────────────────────────────────────────────


def test_chebyshev_center_raises_on_categorical_space(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=5.0), path="length"
    )
    space.add_parameter(ChoiceParameter("mode", ["a", "b"]))
    with pytest.raises(ValueError, match="categorical"):
        chebyshev_center(space)


def _space_with_dependent_constraint(column, b_constraint):
    # b = 2a; constraint a + b <= b_constraint means the true constraint
    # is 3a <= b_constraint, invisible to the independent-only polytope
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=1.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: 2 * x)
    space.add_linear_constraint(LinearConstraint([a, b], lhs=[1.0, 1.0], b=b_constraint))
    return space


def test_chebyshev_center_warns_and_verifies_slack_dependent_constraint(column):
    # relaxed center a = 0.5 satisfies 3a <= 100: warn, verify, return
    space = _space_with_dependent_constraint(column, b_constraint=100.0)
    with pytest.warns(UserWarning, match="relaxed polytope"):
        center = chebyshev_center(space)
    assert center.keys() == {"a"}
    np.testing.assert_allclose(center["a"], 0.5)


def test_chebyshev_center_raises_when_relaxed_center_infeasible(column):
    # relaxed center a = 0.5 violates 3a <= 1.0: must fail, not return it
    space = _space_with_dependent_constraint(column, b_constraint=1.0)
    with pytest.warns(UserWarning, match="relaxed polytope"):
        with pytest.raises(ValueError, match="dependent"):
            chebyshev_center(space)


def test_chebyshev_center_raises_on_dependent_equality_constraint(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=1.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: 2 * x)
    space.add_linear_equality_constraint(
        LinearEqualityConstraint([a, b], lhs=[1.0, -1.0], b=0.0)
    )
    with pytest.raises(ValueError, match="equality"):
        chebyshev_center(space)


def test_chebyshev_center_raises_on_unbounded_parameter(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=np.inf), path="length"
    )
    with pytest.raises(ValueError, match="unbounded"):
        chebyshev_center(space)


def test_chebyshev_center_rounds_integer_parameter(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("n", int, lb=10, ub=13), path="length")
    center = chebyshev_center(space)
    assert center["n"] == 12
    assert isinstance(center["n"], int)


# ── unseeded sampling ─────────────────────────────────────────────────────────


def test_sample_unseeded_calls_differ(column):
    # pins that unseeded calls do not share a default seed; with the former
    # 0..255 seed range this collided once every 256 calls
    space = _space_1d(column)
    s1 = space.sample(3, pool_size=200)
    s2 = space.sample(3, pool_size=200)
    assert s1 != s2
