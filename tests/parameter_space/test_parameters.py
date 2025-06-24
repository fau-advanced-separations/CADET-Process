import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterDependency,
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


@pytest.mark.parametrize("value", ["fast", "slow", "medium"])
def test_choice_param_accepts(value, choice_param):
    choice_param.validate(value)  # Should not raise


@pytest.mark.parametrize("value", ["ultra", "", 5, None])
def test_choice_param_rejects(value, choice_param):
    with pytest.raises(ValueError):
        choice_param.validate(value)


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


def test_linear_equality_constraint_valid():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    p2 = RangedParameter(name="p2", parameter_type=float, lb=0, ub=1)
    con = LinearEqualityConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=3.0)
    assert con.b == 3.0
    assert len(con.lhs) == 2


def test_parameter_space_add_multiple_parameters():
    pspace = ParameterSpace()
    pspace.add_parameter(RangedParameter(name="foo", lb=1, ub=2))
    pspace.add_parameter(RangedParameter(name="bar", lb=-10, ub=0))
    assert pspace.n_parameters == 2


def test_linear_equality_constraint_invalid_length():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    with pytest.raises(ValueError):
        LinearEqualityConstraint(parameters=[p1], lhs=[1.0, 2.0], b=3.0)


def test_linear_equality_constraint_scalar_lhs():
    p1 = RangedParameter(name="p1", parameter_type=float, lb=0, ub=1)
    con = LinearEqualityConstraint(parameters=[p1], lhs=4.0, b=4.0)
    assert con.lhs == [4.0]
    assert con.b == 4.0


def test_parameter_space_add_linear_equality_constraint():
    pspace = ParameterSpace()
    p = RangedParameter(name="x", lb=0, ub=10, parameter_type=float)
    pspace.add_parameter(p)

    constraint = LinearEqualityConstraint(parameters=p, lhs=3.0, b=6.0)
    pspace.add_linear_equality_constraint(constraint)

    assert len(pspace.linear_equality_constraints) == 1
    assert pspace.linear_equality_constraints[0].lhs == [3.0]


def test_parameter_is_dependent():
    a = RangedParameter(name="a", parameter_type=int, lb=0, ub=10)
    b = RangedParameter(name="b", parameter_type=int, lb=0, ub=5)
    c = RangedParameter(name="c", parameter_type=int, lb=0, ub=20)

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

    assert c in space.dependent_parameters
    assert a not in space.dependent_parameters
    assert b not in space.dependent_parameters


@pytest.mark.parametrize(
    "norm_type, lb, ub, value, expected",
    [
        ("linear", 0.0, 10.0, 5.0, 0.5),
        ("log", 1.0, 100.0, 10.0, 0.5),
        ("auto", 1.0, 100.0, 10.0, pytest.approx(0.5, abs=0.1)),
        (None, 0.0, 10.0, 5.0, 5.0),
    ],
)
def test_ranged_parameter_normalization(norm_type, lb, ub, value, expected):
    param = RangedParameter(
        name="test_param",
        parameter_type=float,
        lb=lb,
        ub=ub,
        normalization=norm_type,
    )
    normalized = param.normalize(value)
    denormalized = param.denormalize(normalized)

    if isinstance(expected, float):
        assert normalized == pytest.approx(expected)
    else:
        assert normalized == expected

    assert denormalized == pytest.approx(value, rel=1e-5)


def test_normalization_requires_finite_bounds():
    """Ensure normalization raises if bounds are infinite."""
    with pytest.raises(ValueError, match="Normalization requires finite bounds"):
        RangedParameter(
            name="test_inf",
            parameter_type=float,
            lb=0.0,
            ub=float("inf"),
            normalization="linear",
        )


def test_invalid_normalization_type():
    """Check error is raised for unknown normalization type."""
    with pytest.raises(ValueError, match="Unknown normalization type"):
        RangedParameter(
            name="invalid_norm",
            parameter_type=float,
            lb=1.0,
            ub=10.0,
            normalization="invalid_type",
        )


def test_normalize_without_normalizer_raises():
    """Manually unset normalizer and ensure it raises at runtime."""
    param = RangedParameter(
        name="manual_fail",
        parameter_type=float,
        lb=0.0,
        ub=1.0,
        normalization="linear",
    )
    param.normalizer = None  # forcefully break internal state
    with pytest.raises(AttributeError):  # might raise AttributeError or RuntimeError
        param.normalize(0.5)


def test_denormalize_without_normalizer_raises():
    """Manually unset normalizer and ensure denormalization fails."""
    param = RangedParameter(
        name="manual_fail",
        parameter_type=float,
        lb=0.0,
        ub=1.0,
        normalization="linear",
    )
    param.normalizer = None
    with pytest.raises(AttributeError):
        param.denormalize(0.5)
