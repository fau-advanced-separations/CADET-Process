from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.parameter_space import (
    ChoiceParameter,
    LinearConstraint,
    ParameterSpace,
    RangedParameter,
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
    assert all(s["n"] == round(s["n"]) for s in samples)
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
