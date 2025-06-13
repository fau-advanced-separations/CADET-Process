import sys
from pathlib import Path

import pytest

# # Add the CADET-Process root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterSpace,
    RangedParameter,
)


@pytest.fixture
def int_param():
    return RangedParameter(name="int_param", parameter_type=int, lb=0, ub=10)


@pytest.fixture
def float_param():
    return RangedParameter(name="float_param", parameter_type=float, lb=0.0, ub=5.0)


@pytest.fixture
def choice_param():
    return ChoiceParameter(name="mode", valid_values=["fast", "slow", "medium"])


def test_add_parameter_rejects_duplicate_name():
    space = ParameterSpace()
    p1 = RangedParameter(name="x", lb=0, ub=1)
    p2 = RangedParameter(name="x", lb=1, ub=2)
    space.add_parameter(p1)
    with pytest.raises(ValueError):
        space.add_parameter(p2)


@pytest.mark.parametrize("value", [0, 5])
def test_ranged_param_accepts_valid_values(value, int_param):
    int_param.validate(value)


def test_ranged_param_rejects_value_above_upper(int_param):
    with pytest.raises(ValueError):
        int_param.validate(11)


@pytest.mark.parametrize("value", [-1, 11])
def test_ranged_param_out_of_bounds(value, int_param):
    with pytest.raises(ValueError):
        int_param.validate(value)


@pytest.mark.parametrize("value", [3.5, "five", None])
def test_ranged_param_type_error(value, int_param):
    with pytest.raises(TypeError):
        int_param.validate(value)


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        _ = RangedParameter(name="bad", parameter_type=int, lb=10, ub=5)


@pytest.mark.parametrize("value", ["fast", "slow", "medium"])
def test_choice_param_valid_choices(value, choice_param):
    choice_param.validate(value)


@pytest.mark.parametrize("value", ["ultra", "", 5])
def test_choice_param_invalid_choices(value, choice_param):
    with pytest.raises(ValueError):
        choice_param.validate(value)


def test_dependent_parameters_detected():
    p1 = RangedParameter(name="a", parameter_type=int, lb=0, ub=10)
    p2 = RangedParameter(name="b", parameter_type=float, lb=0.0, ub=5.0)

    dep_param = RangedParameter(
        name="c",
        parameter_type=float,
        lb=0.0,
        ub=20.0,
        dependencies=[p1, p2],
    )

    space = ParameterSpace()
    space.add_parameter(p1)
    space.add_parameter(p2)
    space.add_parameter(dep_param)

    dependent = space.dependent_parameters
    assert dependent == [dep_param]


def test_transform_execution():
    val1 = 3
    val2 = 4.5

    def transform(x, y):
        return x + y

    param = RangedParameter(
        name="sum_param",
        parameter_type=float,
        lb=0,
        ub=10,
        dependencies=[],
        transform=transform,
    )

    result = param.transform(val1, val2)
    assert result == pytest.approx(7.5)


def test_dependent_parameter_with_transform():
    p1 = RangedParameter(name="a", parameter_type=int, lb=0, ub=10)
    p2 = RangedParameter(name="b", parameter_type=float, lb=0.0, ub=5.0)

    def transform(x, y):
        return x + y

    dep_param = RangedParameter(
        name="sum_param",
        parameter_type=float,
        lb=0.0,
        ub=20.0,
        dependencies=[p1, p2],
        transform=transform,
    )

    space = ParameterSpace()
    space.add_parameter(p1)
    space.add_parameter(p2)
    space.add_parameter(dep_param)

    assert dep_param in space.dependent_parameters
    assert len(space.dependent_parameters) == 1

    result = dep_param.transform(3, 2.5)
    assert result == pytest.approx(5.5)


def test_linear_constraint_valid():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    p2 = RangedParameter(name="p2", lb=0, ub=1, parameter_type=float)
    con = LinearConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=5.0)
    assert con.b == 5.0
    assert len(con.lhs) == 2


def test_linear_constraint_invalid_length():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    with pytest.raises(ValueError):
        LinearConstraint(parameters=[p1], lhs=[1.0, 2.0], b=5.0)


def test_linear_constraint_scalar_lhs():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    constraint = LinearConstraint(parameters=[p1], lhs=3.0, b=9.0)
    assert constraint.lhs == [3.0]
    assert constraint.b == 9.0


def test_parameter_space_add_linear_constraint():
    pspace = ParameterSpace()
    p = RangedParameter(name="x", lb=0, ub=10, parameter_type=float)
    pspace.add_parameter(p)
    pspace.add_linear_constraint(parameters=p, lhs=2.0, b=10.0)
    assert len(pspace.linear_constraints) == 1
    assert pspace.linear_constraints[0].lhs == [2.0]


def test_parameter_space_add_multiple_parameters():
    pspace = ParameterSpace()
    pspace._parameters.append(RangedParameter(name="foo", lb=1, ub=2))
    pspace._parameters.append(RangedParameter(name="bar", lb=-10, ub=0))
    assert len(pspace.parameters) == 2


def test_linear_equality_constraint_valid():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    p2 = RangedParameter(name="p2", lb=0, ub=1, parameter_type=float)
    con = LinearEqualityConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=3.0)
    assert con.b == 3.0
    assert len(con.lhs) == 2


def test_linear_equality_constraint_invalid_length():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    with pytest.raises(ValueError):
        LinearEqualityConstraint(parameters=[p1], lhs=[1.0, 2.0], b=3.0)


def test_linear_equality_constraint_scalar_lhs():
    p1 = RangedParameter(name="p1", lb=0, ub=1, parameter_type=float)
    con = LinearEqualityConstraint(parameters=[p1], lhs=4.0, b=4.0)
    assert con.lhs == [4.0]
    assert con.b == 4.0


def test_parameter_space_add_linear_equality_constraint():
    pspace = ParameterSpace()
    p = RangedParameter(name="x", lb=0, ub=10, parameter_type=float)
    pspace.add_parameter(p)
    pspace.add_linear_equality_constraint(parameters=p, lhs=3.0, b=6.0)
    assert len(pspace.linear_equality_constraints) == 1
    assert pspace.linear_equality_constraints[0].lhs == [3.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
