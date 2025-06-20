import sys
from pathlib import Path

import pytest

# # Add the CADET-Process root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterDependency,
    ParameterSpace,
    RangedParameter,
)

# --- Fixtures for easy test reuse ---


@pytest.fixture
def int_param():
    return RangedParameter(name="int_param", parameter_type=int, lb=0, ub=10)


@pytest.fixture
def float_param():
    return RangedParameter(name="float_param", parameter_type=float, lb=0.0, ub=5.0)


@pytest.fixture
def choice_param():
    return ChoiceParameter(name="mode", valid_values=["fast", "slow", "medium"])

# --- RangedParameter tests ---


@pytest.mark.parametrize("value", [0, 5, 10])
def test_ranged_param_accepts_valid(value, int_param):
    int_param.validate(value)  # Should not raise


@pytest.mark.parametrize("value", [-1, 11])
def test_ranged_param_out_of_bounds(value, int_param):
    with pytest.raises(ValueError):
        int_param.validate(value)


@pytest.mark.parametrize("value", [3.5, "five", None])
def test_ranged_param_type_error(value, int_param):
    with pytest.raises(TypeError):
        int_param.validate(value)


def test_ranged_param_invalid_bounds():
    with pytest.raises(ValueError):
        _ = RangedParameter(name="bad", parameter_type=int, lb=10, ub=5)

# --- ChoiceParameter tests ---


@pytest.mark.parametrize("value", ["fast", "slow", "medium"])
def test_choice_param_accepts(value, choice_param):
    choice_param.validate(value)  # Should not raise


@pytest.mark.parametrize("value", ["ultra", "", 5, None])
def test_choice_param_rejects(value, choice_param):
    with pytest.raises(ValueError):
        choice_param.validate(value)

# --- ParameterSpace tests ---


def test_parameter_space_rejects_duplicate_names():
    space = ParameterSpace()
    p1 = RangedParameter(name="x", parameter_type=int, lb=0, ub=1)
    p2 = RangedParameter(name="x", parameter_type=int, lb=1, ub=2)
    space.add_parameter(p1)
    with pytest.raises(ValueError):
        space.add_parameter(p2)


def test_parameter_space_counts_parameters():
    space = ParameterSpace()
    p1 = RangedParameter(name="foo", parameter_type=int, lb=1, ub=2)
    p2 = ChoiceParameter(name="bar", valid_values=["a", "b"])
    space.add_parameter(p1)
    space.add_parameter(p2)
    assert space.n_parameters == 2

# --- LinearConstraint tests ---


def test_linear_constraint_valid():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=1)
    con = LinearConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=5.0)
    assert con.b == 5.0
    assert len(con.lhs) == 2


def test_linear_constraint_invalid_length():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    with pytest.raises(ValueError):
        LinearConstraint(parameters=[p1], lhs=[1.0, 2.0], b=5.0)


def test_linear_constraint_scalar_lhs():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    con = LinearConstraint(parameters=[p1], lhs=3.0, b=9.0)
    assert con.lhs == [3.0]
    assert con.b == 9.0

# --- LinearEqualityConstraint tests ---


def test_linear_equality_constraint_valid():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=1)
    con = LinearEqualityConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=3.0)
    assert con.b == 3.0
    assert len(con.lhs) == 2


def test_linear_equality_constraint_invalid_length():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    with pytest.raises(ValueError):
        LinearEqualityConstraint(parameters=[p1], lhs=[1.0, 2.0], b=3.0)


def test_linear_equality_constraint_scalar_lhs():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    con = LinearEqualityConstraint(parameters=[p1], lhs=4.0, b=4.0)
    assert con.lhs == [4.0]
    assert con.b == 4.0

# --- Constraint is_satisfied checks ---


def test_linear_constraint_is_satisfied_true():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=10)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=10)
    con = LinearConstraint(parameters=[p1, p2], lhs=[2.0, 3.0], b=20.0)
    assert con.is_satisfied([5, 2])  # 2*5 + 3*2 = 10 + 6 = 16 <= 20


def test_linear_constraint_is_satisfied_false():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=10)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=10)
    con = LinearConstraint(parameters=[p1, p2], lhs=[2.0, 3.0], b=10.0)
    assert not con.is_satisfied([5, 2])  # 2*5 + 3*2 = 16 > 10


def test_linear_equality_constraint_is_satisfied_true():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=10)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=10)
    con = LinearEqualityConstraint(parameters=[p1, p2], lhs=[2.0, 3.0], b=16.0)
    assert con.is_satisfied([5, 2])  # 2*5 + 3*2 = 16


def test_linear_equality_constraint_is_satisfied_false():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=10)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=10)
    con = LinearEqualityConstraint(parameters=[p1, p2], lhs=[2.0, 3.0], b=15.0)
    assert not con.is_satisfied([5, 2])  # 16 != 15


def test_parameter_is_dependent():
    # Set up two base parameters and one dependent parameter
    a = RangedParameter(name="a", parameter_type=int, lb=0, ub=10)
    b = RangedParameter(name="b", parameter_type=int, lb=0, ub=5)
    c = RangedParameter(name="c", parameter_type=int, lb=0, ub=20)

    # Make 'c' depend on [a, b]
    dep = ParameterDependency(
        dependent_parameter=c,
        independent_parameters=[a, b],
        transform=lambda x, y: x + y,
    )

    space = ParameterSpace()
    space.add_parameter(a)
    space.add_parameter(b)
    space.add_parameter(c)
    space.add_parameter_dependency(dep)

    # Test that c is recognized as dependent
    assert c in space.dependent_parameters
    # And that a and b are NOT dependent
    assert a not in space.dependent_parameters
    assert b not in space.dependent_parameters


def test_number_of_independents_for_dependent_parameter():
    a = RangedParameter(name="a", parameter_type=int, lb=0, ub=10)
    b = RangedParameter(name="b", parameter_type=int, lb=0, ub=5)
    c = RangedParameter(name="c", parameter_type=int, lb=0, ub=20)
    d = RangedParameter(name="d", parameter_type=int, lb=0, ub=10)

    dep = ParameterDependency(
        dependent_parameter=c,
        independent_parameters=[a, b, d],
        transform=lambda x, y, z: x + y + z,
    )

    space = ParameterSpace()
    for p in [a, b, c, d]:
        space.add_parameter(p)
    space.add_parameter_dependency(dep)

    # Find the dependency for c and count its independent parameters
    dependency_dict = space._parameter_dependencies
    assert c in dependency_dict
    dep_obj = dependency_dict[c]
    assert len(dep_obj.independent_parameters) == 3
    assert set(dep_obj.independent_parameters) == {a, b, d}
