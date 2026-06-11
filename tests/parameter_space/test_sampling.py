from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.parameter_space import (
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


# ── sample: shape ─────────────────────────────────────────────────────────────


def test_sample_shape_independent_only(column):
    space = _space_1d(column)
    samples = space.sample(5, seed=0, pool_size=BURN_IN)
    assert samples.shape == (5, 1)


def test_sample_shape_include_derived(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=5.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space.add_parameter(a, path="length")
    space.add_parameter(b, path="diameter")
    space.add_dependency(b, [a], transform=lambda x: x * 2)
    samples = space.sample(3, seed=0, pool_size=BURN_IN, include_dependent=True)
    assert samples.shape == (3, 2)  # a + b


def test_sample_two_params(column):
    space = _space_2d(column)
    samples = space.sample(10, seed=1, pool_size=BURN_IN)
    assert samples.shape == (10, 2)


# ── sample: values within bounds ─────────────────────────────────────────────


def test_sample_values_within_bounds(column):
    space = _space_2d(column)
    samples = space.sample(20, seed=2, pool_size=BURN_IN)
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 5.0)
    assert np.all(samples[:, 1] >= 0.0) and np.all(samples[:, 1] <= 2.0)


def test_sample_1d_within_bounds(column):
    space = _space_1d(column)
    samples = space.sample(10, seed=3, pool_size=BURN_IN)
    assert np.all(samples >= 1.0) and np.all(samples <= 10.0)


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
    # column b = a * 2
    np.testing.assert_allclose(samples[:, 1], samples[:, 0] * 2)


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
    assert np.all(samples[:, 0] <= 2.0 + 1e-9)


# ── sample: reproducibility ───────────────────────────────────────────────────


def test_sample_same_seed_reproducible(column):
    space = _space_1d(column)
    s1 = space.sample(5, seed=42, pool_size=BURN_IN)
    s2 = space.sample(5, seed=42, pool_size=BURN_IN)
    np.testing.assert_array_equal(s1, s2)


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
