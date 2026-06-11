from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.parameter_space import (
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterSpace,
    RangedParameter,
    TransformedSpace,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class Column:
    length: float = 0.1
    diameter: float = 0.01


@pytest.fixture
def column():
    return Column()


def _linear_space(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=10.0, normalization="linear"),
        path="length",
    )
    return space


def _two_param_space(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=10.0, normalization="linear"),
        path="length",
    )
    space.add_parameter(
        RangedParameter("diameter", float, lb=0.0, ub=1.0, normalization="linear"),
        path="diameter",
    )
    return space


# ── bounds ────────────────────────────────────────────────────────────────────


def test_lower_bounds_normalized(column):
    ts = TransformedSpace(_linear_space(column))
    np.testing.assert_array_equal(ts.lower_bounds, [0.0])


def test_upper_bounds_normalized(column):
    ts = TransformedSpace(_linear_space(column))
    np.testing.assert_array_equal(ts.upper_bounds, [1.0])


def test_bounds_unnormalized_parameter(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("length", float, lb=2.0, ub=8.0), path="length")
    ts = TransformedSpace(space)
    np.testing.assert_array_equal(ts.lower_bounds, [2.0])
    np.testing.assert_array_equal(ts.upper_bounds, [8.0])


def test_bounds_two_params(column):
    ts = TransformedSpace(_two_param_space(column))
    np.testing.assert_array_equal(ts.lower_bounds, [0.0, 0.0])
    np.testing.assert_array_equal(ts.upper_bounds, [1.0, 1.0])


# ── set_values ────────────────────────────────────────────────────────────────


def test_set_values_denormalizes(column):
    ts = TransformedSpace(_linear_space(column))
    ts.set_values([0.5])
    assert column.length == pytest.approx(5.0)


def test_set_values_zero_maps_to_lb(column):
    ts = TransformedSpace(_linear_space(column))
    ts.set_values([0.0])
    assert column.length == pytest.approx(0.0)


def test_set_values_one_maps_to_ub(column):
    ts = TransformedSpace(_linear_space(column))
    ts.set_values([1.0])
    assert column.length == pytest.approx(10.0)


# ── check_bounds ──────────────────────────────────────────────────────────────


def test_check_bounds_within(column):
    ts = TransformedSpace(_linear_space(column))
    assert ts.check_bounds([0.5]) is True


def test_check_bounds_below(column):
    ts = TransformedSpace(_linear_space(column))
    assert ts.check_bounds([-0.1]) is False


def test_check_bounds_above(column):
    ts = TransformedSpace(_linear_space(column))
    assert ts.check_bounds([1.1]) is False


# ── no constraints → empty matrices ──────────────────────────────────────────


def test_no_constraints_returns_empty(column):
    ts = TransformedSpace(_linear_space(column))
    assert ts.A.shape == (0, 1)
    assert ts.b.shape == (0,)
    assert ts.A_eq.shape == (0, 1)
    assert ts.b_eq.shape == (0,)


# ── linear constraint transformation ─────────────────────────────────────────


def test_inequality_constraint_single_param(column):
    # x_phys <= 6  with lb=0, ub=10, span=10
    # → x_norm * 10 <= 6  → A=[10], b=[6]
    space = _linear_space(column)
    p = space.independent_parameters[0]
    space.add_linear_constraint(LinearConstraint([p], lhs=[1.0], b=6.0))
    ts = TransformedSpace(space)
    np.testing.assert_allclose(ts.A, [[10.0]])
    np.testing.assert_allclose(ts.b, [6.0])


def test_inequality_constraint_two_params(column):
    # p1 + p2 <= 8 with p1 in [0,10], p2 in [0,1]
    # → 10*x1 + 0*x1_adj + 1*x2 + 0*x2_adj <= 8 - 0 - 0
    # → A = [10, 1], b = [8]
    space = _two_param_space(column)
    p1, p2 = space.independent_parameters
    space.add_linear_constraint(LinearConstraint([p1, p2], lhs=[1.0, 1.0], b=8.0))
    ts = TransformedSpace(space)
    np.testing.assert_allclose(ts.A, [[10.0, 1.0]])
    np.testing.assert_allclose(ts.b, [8.0])


def test_inequality_constraint_nonzero_lb():
    # lb=2, ub=8, span=6; constraint: x_phys <= 5
    # → 6*x_norm + 2 <= 5 → 6*x_norm <= 3 → A=[6], b=[3]
    col = Column()
    space = ParameterSpace()
    space.add_evaluation_object(col)
    space.add_parameter(
        RangedParameter("length", float, lb=2.0, ub=8.0, normalization="linear"),
        path="length",
    )
    p = space.independent_parameters[0]
    space.add_linear_constraint(LinearConstraint([p], lhs=[1.0], b=5.0))
    ts = TransformedSpace(space)
    np.testing.assert_allclose(ts.A, [[6.0]])
    np.testing.assert_allclose(ts.b, [3.0])


def test_equality_constraint_transformation(column):
    # p1 = 5.0  with lb=0, ub=10
    # → 10*x_norm = 5 → A_eq=[10], b_eq=[5]
    space = _linear_space(column)
    p = space.independent_parameters[0]
    space.add_linear_equality_constraint(
        LinearEqualityConstraint([p], lhs=[1.0], b=5.0)
    )
    ts = TransformedSpace(space)
    np.testing.assert_allclose(ts.A_eq, [[10.0]])
    np.testing.assert_allclose(ts.b_eq, [5.0])


def test_nonlinear_normalizer_raises(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(
        RangedParameter("length", float, lb=1.0, ub=100.0, normalization="log"),
        path="length",
    )
    p = space.independent_parameters[0]
    space.add_linear_constraint(LinearConstraint([p], lhs=[1.0], b=50.0))
    ts = TransformedSpace(space)
    with pytest.raises(ValueError, match="non-affine normalizer"):
        _ = ts.A


# ── roundtrip ─────────────────────────────────────────────────────────────────


def test_set_values_roundtrip(column):
    space = _two_param_space(column)
    ts = TransformedSpace(space)
    ts.set_values([0.3, 0.7])
    assert column.length == pytest.approx(3.0)
    assert column.diameter == pytest.approx(0.7)


# ── delegation ────────────────────────────────────────────────────────────────


def test_evaluation_objects_delegates(column):
    space = _linear_space(column)
    ts = TransformedSpace(space)
    assert ts.evaluation_objects == [column]


def test_n_variables_delegates(column):
    ts = TransformedSpace(_two_param_space(column))
    assert ts.n_variables == 2


def test_space_property(column):
    space = _linear_space(column)
    ts = TransformedSpace(space)
    assert ts.space is space
