import pytest
from CADETProcess.parameter_space import (
    ChoiceParameter,
    LinearConstraint,
    LinearEqualityConstraint,
    ParameterBase,
    ParameterDependency,
    RangedParameter,
)

# ── ParameterBase ─────────────────────────────────────────────────────────────


def test_parameter_base_accepts_any_value():
    p = ParameterBase(name="x")
    p.validate(42)
    p.validate("anything")
    p.validate(None)


# ── RangedParameter ───────────────────────────────────────────────────────────


@pytest.fixture
def int_param():
    return RangedParameter(name="n", parameter_type=int, lb=0, ub=10)


@pytest.fixture
def float_param():
    return RangedParameter(name="x", parameter_type=float, lb=0.0, ub=5.0)


@pytest.mark.parametrize("value", [0, 5, 10])
def test_ranged_int_accepts_valid(value, int_param):
    int_param.validate(value)


@pytest.mark.parametrize("value", [0.0, 5.0, 10.0])
def test_ranged_int_accepts_whole_number_float(value, int_param):
    assert int_param.validate(value) == int(value)
    assert type(int_param.validate(value)) is int


@pytest.mark.parametrize("value", [-1, 11])
def test_ranged_int_rejects_out_of_bounds(value, int_param):
    with pytest.raises(ValueError, match="outside"):
        int_param.validate(value)


@pytest.mark.parametrize("value", [3.5, "five", None])
def test_ranged_int_rejects_wrong_type(value, int_param):
    with pytest.raises(TypeError):
        int_param.validate(value)


def test_ranged_inverted_bounds_raises():
    with pytest.raises(ValueError, match="lower bound"):
        RangedParameter(name="bad", lb=10, ub=5)


def test_ranged_equal_bounds_raises():
    with pytest.raises(ValueError):
        RangedParameter(name="bad", lb=3.0, ub=3.0)


@pytest.mark.parametrize(
    "norm, lb, ub, value, expected",
    [
        ("linear", 0.0, 10.0, 5.0, 0.5),
        ("log", 1.0, 100.0, 10.0, 0.5),
        (None, 0.0, 10.0, 5.0, 5.0),
    ],
)
def test_normalize_roundtrip(norm, lb, ub, value, expected):
    p = RangedParameter(name="p", parameter_type=float, lb=lb, ub=ub, normalization=norm)
    assert p.normalize(value) == pytest.approx(expected, rel=1e-5)
    assert p.denormalize(p.normalize(value)) == pytest.approx(value, rel=1e-5)


def test_normalization_requires_finite_bounds():
    with pytest.raises(ValueError, match="finite bounds"):
        RangedParameter(name="p", lb=0.0, ub=float("inf"), normalization="linear")


def test_unknown_normalization_raises():
    with pytest.raises(ValueError, match="unknown normalization"):
        RangedParameter(name="p", lb=0.0, ub=1.0, normalization="unknown")


# ── ChoiceParameter ───────────────────────────────────────────────────────────


@pytest.fixture
def mode_param():
    return ChoiceParameter(name="mode", valid_values=["fast", "slow", "medium"])


@pytest.mark.parametrize("value", ["fast", "slow", "medium"])
def test_choice_accepts_valid(value, mode_param):
    mode_param.validate(value)


@pytest.mark.parametrize("value", ["ultra", "", 5, None])
def test_choice_rejects_invalid(value, mode_param):
    with pytest.raises(ValueError, match="not a valid choice"):
        mode_param.validate(value)


# ── LinearConstraint ──────────────────────────────────────────────────────────


def test_linear_constraint_scalar_lhs():
    p = RangedParameter(name="x", lb=0, ub=1)
    con = LinearConstraint(parameters=p, lhs=3.0, b=9.0)
    assert con.lhs == [3.0]
    assert con.b == 9.0


def test_linear_constraint_list_lhs():
    p1 = RangedParameter(name="x", lb=0, ub=1)
    p2 = RangedParameter(name="y", lb=0, ub=1)
    con = LinearConstraint(parameters=[p1, p2], lhs=[1.0, 2.0], b=5.0)
    assert con.lhs == [1.0, 2.0]


def test_linear_constraint_mismatched_lhs_raises():
    p = RangedParameter(name="x", lb=0, ub=1)
    with pytest.raises(ValueError, match="coefficients"):
        LinearConstraint(parameters=[p], lhs=[1.0, 2.0], b=0.0)


# ── LinearEqualityConstraint ──────────────────────────────────────────────────


def test_linear_equality_constraint_scalar_lhs():
    p = RangedParameter(name="x", lb=0, ub=1)
    con = LinearEqualityConstraint(parameters=p, lhs=4.0, b=4.0)
    assert con.lhs == [4.0]
    assert con.b == 4.0


def test_linear_equality_constraint_mismatched_raises():
    p = RangedParameter(name="x", lb=0, ub=1)
    with pytest.raises(ValueError, match="coefficients"):
        LinearEqualityConstraint(parameters=[p], lhs=[1.0, 2.0], b=0.0)


# ── ParameterDependency ───────────────────────────────────────────────────────


def test_dependency_holds_fields():
    a = RangedParameter(name="a", lb=0, ub=10)
    b = RangedParameter(name="b", lb=0, ub=10)
    c = RangedParameter(name="c", lb=0, ub=20)
    fn = lambda x, y: x + y  # noqa: E731
    dep = ParameterDependency(
        dependent_parameter=c,
        independent_parameters=[a, b],
        transform=fn,
    )
    assert dep.dependent_parameter is c
    assert dep.independent_parameters == (a, b)
    assert dep.transform(3, 4) == 7
