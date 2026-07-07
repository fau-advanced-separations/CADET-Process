import numpy as np
import pytest
from CADETProcess.metric_space import Constraint, Metric, MetricSpace
from CADETProcess.parameter_space.normalize import LinearNormalizer

# ── Registration ──────────────────────────────────────────────────────────────


def test_add_metric_registers_declaration():
    space = MetricSpace()
    m = space.add_metric(Metric("yield", n_metrics=2))
    assert space.metrics == [m]
    assert space.metric_names == ["yield"]
    assert space.n_metrics == 2
    assert space.labels == ["yield_0", "yield_1"]


def test_add_metric_accepts_name_shorthand():
    space = MetricSpace()
    m = space.add_metric("purity")
    assert m.n_metrics == 1
    assert space.metric_names == ["purity"]


def test_duplicate_metric_name_raises():
    space = MetricSpace()
    space.add_metric("yield")
    with pytest.raises(ValueError, match="already registered"):
        space.add_metric(Metric("yield", n_metrics=2))


def test_annotating_unknown_name_raises():
    space = MetricSpace()
    with pytest.raises(ValueError, match="Unknown metric"):
        space.add_objective("missing")


# ── Objectives ────────────────────────────────────────────────────────────────


def test_add_objective_registers_metric_on_the_fly():
    space = MetricSpace()
    obj = space.add_objective(Metric("yield", n_metrics=2), minimize=False)
    assert space.metric_names == ["yield"]
    assert space.objectives == [obj]
    assert space.n_objectives == 2
    assert space.objective_labels == ["yield_0", "yield_1"]


def test_add_objective_annotates_registered_metric_by_name():
    space = MetricSpace()
    space.add_metric("purity")
    obj = space.add_objective("purity", minimize=False)
    assert obj.metric is space.metrics_dict["purity"]


def test_duplicate_direction_annotation_raises():
    space = MetricSpace()
    space.add_objective(Metric("yield"))
    with pytest.raises(ValueError, match="direction annotation"):
        space.add_objective("yield")


def test_minimize_expands_per_scalar_entry():
    space = MetricSpace()
    space.add_objective(Metric("yield", n_metrics=2), minimize=False)
    space.add_objective(Metric("cost"))
    np.testing.assert_array_equal(space.minimize, [False, False, True])


def test_metric_without_direction_is_plain_output():
    space = MetricSpace()
    space.add_metric("diagnostic")
    space.add_objective(Metric("yield"))
    assert space.n_metrics == 2
    assert space.objective_names == ["yield"]


# ── Constraints ───────────────────────────────────────────────────────────────


def test_add_constraint_broadcasts_scalar_bound():
    space = MetricSpace()
    con = space.add_constraint(Metric("purity", n_metrics=2), bound=0.95, comparison_operator="ge")
    np.testing.assert_array_equal(con.bounds, [0.95, 0.95])
    np.testing.assert_array_equal(space.constraints_bounds, [0.95, 0.95])
    assert space.n_constraints == 2


def test_constraint_bound_length_mismatch_raises():
    with pytest.raises(ValueError, match="bounds"):
        Constraint(Metric("purity", n_metrics=2), bound=[0.9, 0.95, 0.99])


def test_invalid_comparison_operator_raises():
    with pytest.raises(ValueError, match="comparison_operator"):
        Constraint(Metric("purity"), comparison_operator="lt")


def test_metric_can_be_objective_and_constraint():
    space = MetricSpace()
    m = Metric("purity")
    space.add_objective(m, minimize=False)
    space.add_constraint("purity", bound=0.9, comparison_operator="ge")
    assert space.metric_names == ["purity"]
    assert space.n_objectives == 1
    assert space.n_constraints == 1


@pytest.mark.parametrize(
    "operator, value, expected",
    [
        ("le", 0.5, -0.5),  # value <= 1: satisfied by margin 0.5
        ("le", 1.5, 0.5),  # violated by 0.5
        ("ge", 0.5, 0.5),  # value >= 1: violated by 0.5
        ("ge", 1.5, -0.5),  # satisfied by margin 0.5
    ],
)
def test_violation_is_positive_iff_constraint_violated(operator, value, expected):
    con = Constraint(Metric("purity"), bound=1.0, comparison_operator=operator)
    np.testing.assert_allclose(con.violation(value), [expected])


# ── Result validation ─────────────────────────────────────────────────────────


@pytest.fixture
def yield_and_purity_space():
    space = MetricSpace()
    space.add_objective(Metric("yield", n_metrics=2), minimize=False)
    space.add_constraint(Metric("purity"), bound=0.95, comparison_operator="ge")
    return space


def test_validate_returns_declared_metrics_in_canonical_shape(yield_and_purity_space):
    results = yield_and_purity_space.validate(
        {"yield": [0.8, 0.9], "purity": [0.99], "intermediate": "ignored"}
    )
    assert list(results) == ["yield", "purity"]
    assert results["purity"].shape == ()


def test_validate_missing_metric_raises(yield_and_purity_space):
    with pytest.raises(ValueError, match="Missing metrics.*purity"):
        yield_and_purity_space.validate({"yield": [0.8, 0.9]})


def test_validate_wrong_shape_raises(yield_and_purity_space):
    with pytest.raises(ValueError, match="shape"):
        yield_and_purity_space.validate({"yield": [0.8], "purity": 0.99})


# ── Normalization ─────────────────────────────────────────────────────────────


def test_normalize_roundtrip_with_registered_normalizer():
    space = MetricSpace()
    space.add_metric(Metric("cost"), normalizer=LinearNormalizer(lb_input=0.0, ub_input=10.0))
    normalized = space.normalize({"cost": 5.0})
    np.testing.assert_allclose(normalized["cost"], 0.5)
    denormalized = space.denormalize(normalized)
    np.testing.assert_allclose(denormalized["cost"], 5.0)


def test_unnormalized_metric_passes_through():
    space = MetricSpace()
    space.add_metric("cost")
    assert space.normalize({"cost": 5.0})["cost"] == 5.0


def test_set_normalizer_after_registration():
    space = MetricSpace()
    space.add_metric("cost")
    space.set_normalizer("cost", LinearNormalizer(lb_input=0.0, ub_input=2.0))
    np.testing.assert_allclose(space.normalize({"cost": 1.0})["cost"], 0.5)
