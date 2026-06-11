from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.parameter_space import (
    CallableMapper,
    ChoiceParameter,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterSpace,
    RangedParameter,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class Column:
    length: float = 0.1
    diameter: float = 0.01


@dataclass
class Feed:
    duration: float = 60.0
    concentration: float = 1.0


@pytest.fixture
def column():
    return Column()


@pytest.fixture
def feed():
    return Feed()


@pytest.fixture
def space_with_column(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    return space


# ── evaluation objects ────────────────────────────────────────────────────────


def test_add_evaluation_object(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    assert column in space.evaluation_objects


def test_add_duplicate_evaluation_object_raises(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    with pytest.raises(ValueError, match="already registered"):
        space.add_evaluation_object(column)


def test_evaluation_objects_preserves_insertion_order(column, feed):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_evaluation_object(feed)
    assert space.evaluation_objects == [column, feed]


# ── add_parameter ─────────────────────────────────────────────────────────────


def test_add_parameter_with_path_writes(space_with_column, column):
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    space_with_column.add_parameter(p, path="length")
    space_with_column.set_values([0.5])
    assert column.length == pytest.approx(0.5)


def test_add_parameter_no_mapper(space_with_column):
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    space_with_column.add_parameter(p)  # no path or mapper — valid, no write
    space_with_column.set_values([0.5])  # should not raise


def test_add_parameter_duplicate_name_raises(space_with_column):
    p1 = RangedParameter("length", float, lb=0.0, ub=1.0)
    p2 = RangedParameter("length", float, lb=0.0, ub=2.0)
    space_with_column.add_parameter(p1, path="length")
    with pytest.raises(ValueError, match="already registered"):
        space_with_column.add_parameter(p2, path="length")


def test_add_parameter_path_and_mapper_raises(space_with_column):
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    mapper = CallableMapper([object()], fn=lambda o, v: None)
    with pytest.raises(ValueError, match="at most one"):
        space_with_column.add_parameter(p, path="length", mapper=mapper)


def test_add_parameter_evaluation_objects_without_path_raises(space_with_column, column):
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="requires 'path'"):
        space_with_column.add_parameter(p, evaluation_objects=[column])


def test_add_parameter_unknown_evaluation_object_raises(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    unregistered = Column(length=0.99)  # distinct value ensures __eq__ differs
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="not registered"):
        space.add_parameter(p, path="length", evaluation_objects=[unregistered])


def test_add_parameter_subset_of_evaluation_objects(column, feed):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_evaluation_object(feed)
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    space.add_parameter(p, path="length", evaluation_objects=[column])
    space.set_values([0.7])
    assert column.length == pytest.approx(0.7)
    assert feed.duration == pytest.approx(60.0)  # untouched


def test_add_parameter_no_evaluation_objects_registered_raises():
    space = ParameterSpace()
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="No evaluation objects"):
        space.add_parameter(p, path="length")


def test_add_parameter_with_callable(space_with_column, column):
    calls = []
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    space_with_column.add_parameter_with_callable(
        p, fn=lambda obj, v: calls.append((obj, v))
    )
    space_with_column.set_values([0.3])
    assert calls == [(column, 0.3)]


# ── independent / dependent parameters ───────────────────────────────────────


def test_independent_parameters_without_dependencies(space_with_column):
    p = RangedParameter("length", float, lb=0.0, ub=1.0)
    space_with_column.add_parameter(p)
    assert space_with_column.independent_parameters == [p]
    assert space_with_column.dependent_parameters == []


def test_independent_and_dependent_split(space_with_column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    c = RangedParameter("c", float, lb=0.0, ub=20.0)
    for p in (a, b, c):
        space_with_column.add_parameter(p)
    space_with_column.add_dependency(c, [a, b], transform=lambda x, y: x + y)
    assert space_with_column.independent_parameters == [a, b]
    assert space_with_column.dependent_parameters == [c]
    assert space_with_column.n_variables == 2


# ── add_dependency ────────────────────────────────────────────────────────────


def test_dependency_resolves_correctly(space_with_column, column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    c = RangedParameter("c", float, lb=0.0, ub=20.0)
    space_with_column.add_parameter(a)
    space_with_column.add_parameter(b)
    space_with_column.add_parameter(c, path="length")
    space_with_column.add_dependency(c, [a, b], transform=lambda x, y: x + y)
    space_with_column.set_values([3.0, 4.0])
    assert column.length == pytest.approx(7.0)


def test_dependency_chain(space_with_column, column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=20.0)
    c = RangedParameter("c", float, lb=0.0, ub=20.0)
    for p in (a, b, c):
        space_with_column.add_parameter(p)
    space_with_column.add_parameter(
        RangedParameter("d", float, lb=0.0, ub=40.0), path="length"
    )
    d = space_with_column.parameters[-1]
    space_with_column.add_dependency(b, [a], transform=lambda x: x * 2)
    space_with_column.add_dependency(d, [b], transform=lambda x: x + 1)
    space_with_column.set_values([3.0, 0.0])
    assert column.length == pytest.approx(7.0)  # b=6, d=7


def test_dependency_cycle_raises(space_with_column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space_with_column.add_parameter(a)
    space_with_column.add_parameter(b)
    space_with_column.add_dependency(b, [a], transform=lambda x: x)
    with pytest.raises(ValueError, match="[Cc]ycle"):
        space_with_column.add_dependency(a, [b], transform=lambda x: x)


def test_duplicate_dependency_raises(space_with_column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space_with_column.add_parameter(a)
    space_with_column.add_parameter(b)
    space_with_column.add_dependency(b, [a], transform=lambda x: x)
    with pytest.raises(ValueError, match="already has a dependency"):
        space_with_column.add_dependency(b, [a], transform=lambda x: x * 2)


def test_dependency_unregistered_parameter_raises(space_with_column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=10.0)
    space_with_column.add_parameter(a)
    with pytest.raises(ValueError, match="not registered"):
        space_with_column.add_dependency(b, [a], transform=lambda x: x)


# ── bounds ────────────────────────────────────────────────────────────────────


def test_lower_upper_bounds(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=1.0, ub=5.0))
    space_with_column.add_parameter(RangedParameter("b", float, lb=2.0, ub=8.0))
    np.testing.assert_array_equal(space_with_column.lower_bounds, [1.0, 2.0])
    np.testing.assert_array_equal(space_with_column.upper_bounds, [5.0, 8.0])


def test_check_bounds_within(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    assert space_with_column.check_bounds([0.5]) is True


def test_check_bounds_below(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    assert space_with_column.check_bounds([-0.1]) is False


def test_check_bounds_above(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    assert space_with_column.check_bounds([1.1]) is False


def test_check_bounds_with_tol(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    assert space_with_column.check_bounds([1.05], tol=0.1) is True


def test_check_bounds_wrong_length_raises(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    with pytest.raises(ValueError, match="Expected 1"):
        space_with_column.check_bounds([0.5, 0.5])


def test_check_bounds_normalized(space_with_column):
    space_with_column.add_parameter(
        RangedParameter("a", float, lb=0.0, ub=10.0, normalization="linear")
    )
    # 0.5 normalized → 5.0 physical, which is within [0, 10]
    assert space_with_column.check_bounds([0.5], normalized=True) is True
    # 1.5 normalized → 15.0 physical, which is outside [0, 10]
    assert space_with_column.check_bounds([1.5], normalized=True) is False


# ── normalization ─────────────────────────────────────────────────────────────


def test_normalize_denormalize_roundtrip(space_with_column):
    space_with_column.add_parameter(
        RangedParameter("a", float, lb=0.0, ub=10.0, normalization="linear")
    )
    x = np.array([3.0])
    np.testing.assert_allclose(space_with_column.denormalize(space_with_column.normalize(x)), x)


def test_normalize_choice_parameter_unchanged(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(ChoiceParameter("mode", ["a", "b"]))
    x = np.array([0.0])
    np.testing.assert_array_equal(space.normalize(x), x)


# ── set_values ────────────────────────────────────────────────────────────────


def test_set_values_writes_to_object(space_with_column, column):
    space_with_column.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=1.0), path="length"
    )
    space_with_column.set_values([0.42])
    assert column.length == pytest.approx(0.42)


def test_set_values_normalized(space_with_column, column):
    space_with_column.add_parameter(
        RangedParameter("length", float, lb=0.0, ub=1.0, normalization="linear"),
        path="length",
    )
    space_with_column.set_values([1.0], normalized=True)  # 1.0 normalized → 1.0 physical
    assert column.length == pytest.approx(1.0)


def test_set_values_validate_bounds_raises_on_violation(space_with_column):
    space_with_column.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    with pytest.raises(ValueError, match="bound"):
        space_with_column.set_values([1.5], validate_bounds=True)


def test_set_values_validate_catches_derived_out_of_bounds(space_with_column, column):
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=5.0)  # derived, tight bound
    space_with_column.add_parameter(a)
    space_with_column.add_parameter(b, path="length")
    space_with_column.add_dependency(b, [a], transform=lambda x: x * 2)
    with pytest.raises(ValueError, match="outside"):
        space_with_column.set_values([4.0])  # b = 8.0 > 5.0


# ── typed parameter subsets ──────────────────────────────────────────────────


def test_continuous_parameters():
    space = ParameterSpace()
    a = RangedParameter("a", float, lb=0, ub=1)
    b = RangedParameter("b", int, lb=0, ub=10)
    c = ChoiceParameter("c", ["x", "y"])
    space.add_parameter(a)
    space.add_parameter(b)
    space.add_parameter(c)
    assert space.continuous_parameters == [a]


def test_integer_parameters():
    space = ParameterSpace()
    a = RangedParameter("a", float, lb=0, ub=1)
    b = RangedParameter("b", int, lb=0, ub=10)
    space.add_parameter(a)
    space.add_parameter(b)
    assert space.integer_parameters == [b]


def test_categorical_parameters():
    space = ParameterSpace()
    a = RangedParameter("a", float, lb=0, ub=1)
    c = ChoiceParameter("c", ["x", "y"])
    space.add_parameter(a)
    space.add_parameter(c)
    assert space.categorical_parameters == [c]


def test_typed_subsets_exclude_dependent():
    space = ParameterSpace()
    a = RangedParameter("a", float, lb=0, ub=10)
    b = RangedParameter("b", float, lb=0, ub=10)
    space.add_parameter(a)
    space.add_parameter(b)
    space.add_dependency(b, [a], transform=lambda x: x)
    assert space.continuous_parameters == [a]


# ── evaluate_bounds ───────────────────────────────────────────────────────────


@pytest.fixture
def bounded_space(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    space.add_parameter(RangedParameter("a", float, lb=0.0, ub=1.0))
    return space


def test_evaluate_bounds_feasible_all_nonpositive(bounded_space):
    cv = bounded_space.evaluate_bounds([0.5])
    assert np.all(cv <= 0)


def test_evaluate_bounds_lower_violation(bounded_space):
    cv = bounded_space.evaluate_bounds([-0.2])
    # lb - x = 0 - (-0.2) = 0.2 > 0
    assert cv[0] == pytest.approx(0.2)
    assert cv[1] <= 0  # x - ub = -0.2 - 1.0 < 0


def test_evaluate_bounds_upper_violation(bounded_space):
    cv = bounded_space.evaluate_bounds([1.2])
    assert cv[0] <= 0  # lb - x = 0 - 1.2 < 0
    # x - ub = 1.2 - 1.0 = 0.2 > 0
    assert cv[1] == pytest.approx(0.2)


def test_evaluate_bounds_includes_dependent_variables(space_with_column, column):
    # With a dependent variable, evaluate_bounds expects a full vector
    # (independent + dependent).  A5: n_all != n_independent — shape must be right.
    a = RangedParameter("a", float, lb=0.0, ub=10.0)
    b = RangedParameter("b", float, lb=0.0, ub=5.0)
    space_with_column.add_parameter(a)
    space_with_column.add_parameter(b)
    space_with_column.add_dependency(b, [a], transform=lambda x: x * 0.5)
    # Full resolved vector: a=4.0, b=2.0 — both within bounds
    cv = space_with_column.evaluate_bounds([4.0, 2.0])
    assert cv.shape == (4,)  # 2 * n_parameters
    assert np.all(cv <= 0)


# ── evaluate_linear_constraints ───────────────────────────────────────────────


@pytest.fixture
def space_with_lincon(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=1.0)
    b = RangedParameter("b", float, lb=0.0, ub=1.0)
    space.add_parameter(a)
    space.add_parameter(b)
    # a + b <= 1
    space.add_linear_constraint(LinearConstraint([a, b], lhs=[1.0, 1.0], b=1.0))
    return space


def test_evaluate_linear_constraints_feasible(space_with_lincon):
    # a=0.4, b=0.4 → 0.4 + 0.4 - 1.0 = -0.2 ≤ 0
    cv = space_with_lincon.evaluate_linear_constraints([0.4, 0.4])
    assert np.all(cv <= 0)


def test_evaluate_linear_constraints_violated(space_with_lincon):
    # a=0.6, b=0.6 → 0.6 + 0.6 - 1.0 = 0.2 > 0
    cv = space_with_lincon.evaluate_linear_constraints([0.6, 0.6])
    assert cv[0] == pytest.approx(0.2)


# ── evaluate_linear_equality_constraints ──────────────────────────────────────


@pytest.fixture
def space_with_lineqcon(column):
    space = ParameterSpace()
    space.add_evaluation_object(column)
    a = RangedParameter("a", float, lb=0.0, ub=1.0)
    b = RangedParameter("b", float, lb=0.0, ub=1.0)
    space.add_parameter(a)
    space.add_parameter(b)
    # a + b = 1
    space.add_linear_equality_constraint(
        LinearEqualityConstraint([a, b], lhs=[1.0, 1.0], b=1.0)
    )
    return space


def test_evaluate_linear_equality_constraints_satisfied(space_with_lineqcon):
    cv = space_with_lineqcon.evaluate_linear_equality_constraints([0.3, 0.7])
    assert cv == pytest.approx([0.0])


def test_evaluate_linear_equality_constraints_violated(space_with_lineqcon):
    cv = space_with_lineqcon.evaluate_linear_equality_constraints([0.3, 0.5])
    # 0.3 + 0.5 - 1.0 = -0.2
    assert cv == pytest.approx([-0.2])
