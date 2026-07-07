import warnings

import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.optimization import Individual, OptimizationProblem, Population

from tests.optimization.conftest import (
    EvaluationObject,
    LinearConstraintsSooTestProblem2,
    LinearEqualityConstraintsSooTestProblem,
    make_optimization_problem,
)

# ── Variables ─────────────────────────────────────────────────────────────────


@pytest.fixture
def op_basic():
    op = OptimizationProblem("simple", use_diskcache=False)
    op.add_variable("var_0", lb=0, ub=1)
    op.add_variable("var_1", lb=0, ub=10)
    return op


def test_variable_names(op_basic):
    assert op_basic.variable_names == ["var_0", "var_1"]
    with pytest.raises(CADETProcessError):
        op_basic.add_variable("var_0")


def test_bounds(op_basic):
    np.testing.assert_allclose(op_basic.lower_bounds, [0, 0])
    np.testing.assert_allclose(op_basic.upper_bounds, [1, 10])
    with pytest.raises(ValueError):
        op_basic.add_variable("spam", lb=0, ub=0)


# ── Linear constraints ────────────────────────────────────────────────────────


@pytest.fixture
def op_with_linear_constraint():
    return make_optimization_problem(n_lincon=1, use_diskcache=False)


def test_add_linear_constraints(op_with_linear_constraint):
    op = op_with_linear_constraint
    op.add_linear_constraint("var_0")
    op.add_linear_constraint(["var_0", "var_1"])
    op.add_linear_constraint(["var_0", "var_1"], [2, 2])
    op.add_linear_constraint(["var_0", "var_1"], [3, 3], 1)
    op.add_linear_constraint(["var_0", "var_1"], 4)

    A_expected = np.array([
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    np.testing.assert_allclose(op.A, A_expected)
    np.testing.assert_allclose(op.b, [0, 0, 0, 0, 1, 0])

    with pytest.raises(CADETProcessError):
        op.add_linear_constraint("inexistent")
    with pytest.raises(CADETProcessError):
        op.add_linear_constraint("var_0", [])


def test_add_linear_equality_constraints(op_with_linear_constraint):
    op = op_with_linear_constraint
    op.add_linear_equality_constraint("var_0")
    op.add_linear_equality_constraint(["var_0", "var_1"])
    op.add_linear_equality_constraint(["var_0", "var_1"], [2, 2])
    op.add_linear_equality_constraint(["var_0", "var_1"], [3, 3], 1)
    op.add_linear_equality_constraint(["var_0", "var_1"], 4)

    Aeq_expected = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    np.testing.assert_allclose(op.Aeq, Aeq_expected)
    np.testing.assert_allclose(op.beq, [0, 0, 0, 1, 0])

    with pytest.raises(CADETProcessError):
        op.add_linear_equality_constraint("inexistent")
    with pytest.raises(CADETProcessError):
        op.add_linear_equality_constraint("var_0", [])


def test_remove_linear_constraint():
    op = make_optimization_problem(n_lincon=1)
    assert op.n_linear_constraints == 1
    op.remove_linear_constraint(0)
    assert op.n_linear_constraints == 0


def test_remove_linear_equality_constraint():
    op = make_optimization_problem(n_lineqcon=1)
    assert op.n_linear_equality_constraints == 1
    op.remove_linear_equality_constraint(0)
    assert op.n_linear_equality_constraints == 0


# ── Chebyshev center and initial values ──────────────────────────────────────


def test_chebyshev_center_feasible(op_with_linear_constraint):
    op = op_with_linear_constraint
    x0 = op.get_chebyshev_center(include_dependent_variables=True)

    assert x0.shape == (2,)
    assert np.all(x0 >= op.lower_bounds)
    assert np.all(x0 <= op.upper_bounds)
    assert op.check_linear_constraints(x0)


def test_create_initial_values_invariants(op_with_linear_constraint):
    op = op_with_linear_constraint

    x = op.create_initial_values(10, seed=1)
    assert x.shape == (10, 2)
    assert np.all(x >= op.lower_bounds)
    assert np.all(x <= op.upper_bounds)
    for xi in x:
        assert op.check_linear_constraints(xi)

    # Deterministic: same seed gives same result.
    x_again = op.create_initial_values(10, seed=1)
    np.testing.assert_allclose(x, x_again)

    # Different seed gives different result.
    x_other = op.create_initial_values(10, seed=2)
    assert not np.allclose(x, x_other)


def test_create_initial_values_with_custom_sampler():
    from CADETProcess.parameter_space import LatinHypercubeSampler

    op = OptimizationProblem("box", use_diskcache=False)
    op.add_variable("x_0", lb=0, ub=8)
    op.add_variable("x_1", lb=0, ub=1)

    x = op.create_initial_values(8, seed=1, sampler=LatinHypercubeSampler())
    assert x.shape == (8, 2)
    assert np.all(x >= op.lower_bounds)
    assert np.all(x <= op.upper_bounds)
    # LHS stratification: one point per unit stratum along x_0
    assert sorted(int(np.floor(v)) for v in x[:, 0]) == list(range(8))


def test_create_initial_values_custom_sampler_rejects_linear_constraints(
    op_with_linear_constraint,
):
    from CADETProcess.parameter_space import SobolSampler

    with pytest.raises(ValueError, match="linear constraints"):
        op_with_linear_constraint.create_initial_values(4, seed=0, sampler=SobolSampler())


# ── Dependent variables ──────────────────────────────────────────────────────


@pytest.fixture
def op_with_dependent_variable():
    op = OptimizationProblem("simple", use_diskcache=False)
    op.add_variable("foo", lb=0, ub=1)
    op.add_variable("bar", lb=0, ub=1)
    op.add_variable("spam", lb=0, ub=1)
    op.add_variable("eggs", lb=0, ub=1)
    op.add_linear_constraint(["foo", "spam"], [-1, 1])
    op.add_linear_constraint(["foo", "eggs"], [-1, 1])
    op.add_linear_constraint(["eggs", "spam"], [-1, 1])
    op.add_variable_dependency("spam", "bar", lambda var: var)
    return op


def test_dependent_variable_names(op_with_dependent_variable):
    op = op_with_dependent_variable
    assert op.independent_variable_names == ["foo", "bar", "eggs"]
    assert op.dependent_variable_names == ["spam"]
    assert op.variable_names == ["foo", "bar", "spam", "eggs"]

    with pytest.raises(CADETProcessError):
        op.add_variable_dependency("inexistent", ["bar", "spam"], lambda x: x)
    with pytest.raises(CADETProcessError):
        op.add_variable_dependency("foo", ["inexistent", "spam"], lambda x: x)
    with pytest.raises(CADETProcessError):
        op.add_variable_dependency("spam", ["bar", "spam"], lambda x: x)
    with pytest.raises(CADETProcessError):
        op.add_variable_dependency("spam", ["bar", "spam"], transform=None)


def test_chebyshev_center_dependent_constraints_never_silently_violated(
    op_with_dependent_variable,
):
    # Constraints referencing the dependent 'spam' are dropped from the
    # polytope with a warning; the contract is that the returned center is
    # verified against them, so the outcome is either a feasible point or a
    # ValueError, never a silently infeasible point.  Which of the two occurs
    # depends on where the LP places the degenerate 'bar' coordinate.
    op = op_with_dependent_variable
    with pytest.warns(UserWarning, match="relaxed polytope"):
        try:
            x = op.get_chebyshev_center(include_dependent_variables=True)
        except ValueError:
            return
    assert op.check_individual(
        x, get_dependent_values=False, check_nonlinear_constraints=False
    )


def test_create_initial_values_with_dependent_variable(op_with_dependent_variable):
    op = op_with_dependent_variable

    # Without dependent values: shape is (n, n_independent).
    x_ind = op.create_initial_values(10, seed=1, include_dependent_variables=False)
    assert x_ind.shape == (10, 3)
    for xi in x_ind:
        assert op.check_linear_constraints(xi, get_dependent_values=True)

    # With dependent values: shape is (n, n_total).
    x_full = op.create_initial_values(10, seed=1, include_dependent_variables=True)
    assert x_full.shape == (10, 4)
    for xi in x_full:
        assert op.check_linear_constraints(xi, get_dependent_values=False)
    # spam (col 2) == bar (col 1).
    np.testing.assert_allclose(x_full[:, 2], x_full[:, 1])

    # Round-trip: independent values extracted from full match the independent-only call.
    x_ind_from_full = np.array([op.get_independent_values(xi) for xi in x_full])
    np.testing.assert_allclose(x_ind_from_full, x_ind)

    # Determinism.
    x_full_again = op.create_initial_values(10, seed=1, include_dependent_variables=True)
    np.testing.assert_allclose(x_full, x_full_again)

    # Different seed.
    x_other = op.create_initial_values(10, seed=2, include_dependent_variables=True)
    assert not np.allclose(x_full, x_other)


# ── evaluate_* with dependent variables (core regression) ────────────────────


@pytest.fixture
def op_dep_with_objective(eval_obj):
    """OP with one independent variable, one dependent (copies independent), and an objective."""
    op = OptimizationProblem("dep_obj", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=10)
    op.add_variable("sized_list_param", lb=0, ub=10, indices=0)
    op.add_variable_dependency("sized_list_param", "scalar_param", lambda var: var)
    op.add_objective(lambda obj: obj.scalar_param, n_objectives=1)
    return op


def test_evaluate_objectives_with_dependent_variable(op_dep_with_objective):
    """Independent-only input (default) produces correct result, not bad metrics."""
    # scalar_param=3.0 → sized_list_param[0]=3.0; objective reads scalar_param
    np.testing.assert_allclose(op_dep_with_objective.evaluate_objectives([3.0]), [3.0])


def test_create_initial_values_into_evaluate_objectives(op_dep_with_objective):
    """create_initial_values() default returns independent-only; feeds into evaluate_objectives."""
    op = op_dep_with_objective
    x0 = op.create_initial_values(1, seed=1)[0]
    assert x0.shape == (1,)  # independent-only
    f = op.evaluate_objectives(x0)
    assert np.all(np.isfinite(f))


def test_evaluate_objectives_full_vector_matches_independent(op_dep_with_objective):
    """get_dependent_values=False accepts a full vector and gives the same result."""
    op = op_dep_with_objective
    x_ind = np.array([3.0])
    x_full = op.get_dependent_values(x_ind)
    f_ind = op.evaluate_objectives(x_ind)
    f_full = op.evaluate_objectives(x_full, get_dependent_values=False)
    np.testing.assert_allclose(f_ind, f_full)


def test_evaluate_meta_scores_with_dependent_variable(op_dep_with_objective):
    """Meta scores work correctly when the problem has dependent variables."""
    op = op_dep_with_objective
    op.add_meta_score(lambda obj: obj.scalar_param, n_meta_scores=1)
    m = op.evaluate_meta_scores([3.0])
    np.testing.assert_allclose(m, [3.0])


def test_evaluate_meta_scores_full_vector_matches_independent(op_dep_with_objective):
    """get_dependent_values=False on meta scores gives same result as independent input."""
    op = op_dep_with_objective
    op.add_meta_score(lambda obj: obj.scalar_param, n_meta_scores=1)
    x_ind = np.array([3.0])
    x_full = op.get_dependent_values(x_ind)
    np.testing.assert_allclose(
        op.evaluate_meta_scores(x_ind),
        op.evaluate_meta_scores(x_full, get_dependent_values=False),
    )


# ── check_* with dependent variables ─────────────────────────────────────────


def test_check_individual_independent_and_full_agree(op_with_dependent_variable):
    """check_individual(x_ind) and check_individual(x_full, get_dependent_values=False) agree."""
    op = op_with_dependent_variable
    x_ind = op.create_initial_values(1, seed=1)[0]
    x_full = op.get_dependent_values(x_ind)
    assert op.check_individual(x_ind) == op.check_individual(x_full, get_dependent_values=False)


def test_check_linear_constraints_dependent_var_independent_input(op_with_dependent_variable):
    """Default (independent-only) input expands to full before checking A @ x_full."""
    op = op_with_dependent_variable
    x_ind = op.create_initial_values(1, seed=1)[0]
    result = op.check_linear_constraints(x_ind)
    assert isinstance(result, bool)


def test_check_linear_constraints_full_vector_matches_independent(op_with_dependent_variable):
    """check_linear_constraints gives consistent result for both input formats."""
    op = op_with_dependent_variable
    x_ind = op.create_initial_values(1, seed=1)[0]
    x_full = op.get_dependent_values(x_ind)
    assert op.check_linear_constraints(x_ind) == op.check_linear_constraints(
        x_full, get_dependent_values=False
    )


# ── _evaluate_individual exception handling ───────────────────────────────────


def test_evaluate_objectives_metric_exception_returns_bad_metrics(caplog):
    """Exception raised inside an objective function logs a warning and returns bad metrics."""
    import logging

    op = OptimizationProblem("err", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    op.add_objective(lambda x: 1 / 0, n_objectives=1)  # ZeroDivisionError
    with caplog.at_level(logging.WARNING):
        f = op.evaluate_objectives([0.5])
    assert not np.all(np.isfinite(f))
    assert any("Metric" in r.message or "failed" in r.message for r in caplog.records)


# ── Jacobian ──────────────────────────────────────────────────────────────────


def _make_jacobian_problem(objectives, n_objectives=1):
    """Build a minimal OP for Jacobian testing."""
    op = OptimizationProblem("jac", use_diskcache=False)
    if n_objectives == 1:
        # Infer variable count from the first test call.
        pass
    for name in ["x"] if n_objectives <= 1 else ["x_1", "x_2"]:
        op.add_variable(name)
    op.add_objective(objectives, n_objectives=n_objectives)
    return op


@pytest.mark.parametrize("x,expected_grad", [
    ([2], [[4]]),
    ([0], [[0]]),
])
def test_jacobian_single_obj_single_var(x, expected_grad):
    op = OptimizationProblem("jac", use_diskcache=False)
    op.add_variable("x")
    op.add_objective(lambda x: x[0] ** 2)
    np.testing.assert_allclose(op.objective_jacobian(x), expected_grad, atol=1e-2)


@pytest.mark.parametrize("x,expected_grad", [
    ([2, 2], [[-4, 4]]),
    ([0, 0], [[0, 0]]),
])
def test_jacobian_single_obj_two_vars(x, expected_grad):
    op = OptimizationProblem("jac", use_diskcache=False)
    op.add_variable("x_1")
    op.add_variable("x_2")
    op.add_objective(lambda x: x[1] ** 2 - x[0] ** 2)
    np.testing.assert_allclose(op.objective_jacobian(x), expected_grad, atol=1e-2)


@pytest.mark.parametrize("x,expected_grad", [
    ([2], [[4], [4]]),
    ([0], [[0], [0]]),
])
def test_jacobian_two_obj_single_var(x, expected_grad):
    op = OptimizationProblem("jac", use_diskcache=False)
    op.add_variable("x")
    op.add_objective(lambda x: [x[0] ** 2, x[0] ** 2], n_objectives=2)
    np.testing.assert_allclose(op.objective_jacobian(x), expected_grad, atol=1e-2)


@pytest.mark.parametrize("x,expected_grad", [
    ([2, 2], [[4, 0], [0, 4]]),
    ([0, 0], [[0, 0], [0, 0]]),
])
def test_jacobian_two_obj_two_var(x, expected_grad):
    op = OptimizationProblem("jac", use_diskcache=False)
    op.add_variable("x_1")
    op.add_variable("x_2")
    op.add_objective(lambda x: [x[0] ** 2, x[1] ** 2], n_objectives=2)
    np.testing.assert_allclose(op.objective_jacobian(x), expected_grad, atol=1e-2)


@pytest.fixture
def op_with_linear_transform():
    op = OptimizationProblem("linear_transform")
    op.add_variable("x", lb=-2, ub=2, normalization="linear")
    op.add_objective(lambda x: x[0] ** 2)
    op.add_nonlinear_constraint(lambda x: [x[0] ** 2 - 1], n_nonlinear_constraints=1)
    return op


def test_objective_jacobian_physical_vs_transformed(op_with_linear_transform):
    # Physical: ∂(x²)/∂x at x=1 is 2.
    np.testing.assert_allclose(
        op_with_linear_transform.objective_jacobian([1.0]), [[2]], atol=1e-2,
    )
    # Transformed: x_t = 0.75 → x_p = 1.0; chain rule scales by (ub - lb) = 4.
    np.testing.assert_allclose(
        op_with_linear_transform.objective_jacobian([0.75], untransform=True),
        [[8]], atol=0.1,
    )


def test_nonlinear_constraint_jacobian_physical_vs_transformed(op_with_linear_transform):
    # g(x) = x² - 1; same gradient as the objective.
    np.testing.assert_allclose(
        op_with_linear_transform.nonlinear_constraint_jacobian([1.0]),
        [[2]], atol=1e-2,
    )
    np.testing.assert_allclose(
        op_with_linear_transform.nonlinear_constraint_jacobian([0.75], untransform=True),
        [[8]], atol=0.1,
    )


# ── Constraint transforms ─────────────────────────────────────────────────────


def _check_inequality_constraints(X, problem, transformed_space=False):
    A = problem.A_transformed if transformed_space else problem.A
    b = problem.b_transformed if transformed_space else problem.b
    lhs = np.array([A.dot(x) - b for x in X])
    return np.all(lhs <= 0, axis=1)


def _check_equality_constraints(X, problem, transformed_space=False):
    Aeq = problem.Aeq_transformed if transformed_space else problem.Aeq
    beq = problem.beq_transformed if transformed_space else problem.beq
    lhs = np.array([Aeq.dot(x) - beq for x in X])
    return np.all(np.abs(lhs) <= 1e-4, axis=1)


def _check_constraint_transform(problem, check_constraint_func):
    nvars = problem.n_independent_variables
    rng = np.random.default_rng(seed=72729)
    X = rng.uniform(0, 1, size=(100000, nvars))
    CV = check_constraint_func(X=X, problem=problem, transformed_space=True)
    X_valid = problem.untransform(X[CV])
    X_invalid = problem.untransform(X[~CV])
    assert np.all(check_constraint_func(X=X_valid, problem=problem, transformed_space=False))
    assert np.all(~check_constraint_func(X=X_invalid, problem=problem, transformed_space=False))


def test_linear_inequality_constrained_transform():
    _check_constraint_transform(
        LinearConstraintsSooTestProblem2(transform="linear", use_diskcache=False),
        _check_inequality_constraints,
    )


def test_linear_equality_constrained_transform():
    _check_constraint_transform(
        LinearEqualityConstraintsSooTestProblem(transform="linear", use_diskcache=False),
        _check_equality_constraints,
    )


# ── Evaluation object: variable mapping ───────────────────────────────────────


def test_variable_names_with_eval_obj(op_with_dep_var):
    op = op_with_dep_var
    assert op.variable_names == ["scalar_param", "sized_list_param"]

    with pytest.raises(CADETProcessError):
        op.add_variable("bar", lb=0, ub=1)

    op.add_variable("bar", evaluation_objects=None, lb=0, ub=1)
    assert op.variable_names == ["scalar_param", "sized_list_param", "bar"]


def test_set_variables_writes_to_eval_object(op_with_dep_var, eval_obj):
    op_with_dep_var.set_variables([0.5])
    assert eval_obj.scalar_param == pytest.approx(0.5)


def test_get_variable_value_round_trip(op_with_dep_var):
    op_with_dep_var.set_variables([0.7])
    assert op_with_dep_var.get_variable_value("scalar_param") == pytest.approx(0.7)


def test_get_variable_value_unregistered_raises(op_with_dep_var):
    with pytest.raises(KeyError):
        op_with_dep_var.get_variable_value("nonexistent")


def test_indexed_write_through_via_dependency(op_with_dep_var, eval_obj):
    """sized_list_param[0] is a dependent of scalar_param; both must update together."""
    op_with_dep_var.set_variables([3.0])
    assert eval_obj.sized_list_param[0] == pytest.approx(3.0)


def test_duplicate_variables(op_with_dep_var):
    op = op_with_dep_var
    op.check_duplicate_variables()

    op.add_variable("foo", evaluation_objects=None)
    with pytest.raises(CADETProcessError):
        op.add_variable("foo", evaluation_objects=None)

    with pytest.raises(CADETProcessError):
        op.add_variable("another_scalar_param", parameter_path="scalar_param")

    op.add_variable(
        "sized_list_index_1", parameter_path="sized_list_param", lb=0, ub=10, indices=1,
    )
    with pytest.raises(CADETProcessError):
        op.add_variable(
            "sized_list_index_1b", parameter_path="sized_list_param", lb=0, ub=10, indices=1,
        )


# ── Multiple evaluation objects ───────────────────────────────────────────────


@pytest.fixture
def op_with_multiple_eval_objects():
    def single_obj_1(eval_obj):
        return 0

    def single_obj_2(eval_obj):
        return 1

    def multi_obj(eval_obj):
        return [2, 3]

    eval_obj_1 = EvaluationObject(name="foo")
    eval_obj_2 = EvaluationObject(name="bar")
    op = OptimizationProblem("with_evaluator", use_diskcache=False)
    op.add_evaluation_object(eval_obj_1)
    op.add_evaluation_object(eval_obj_2)
    op.add_variable(
        "scalar_eval_obj_1", parameter_path="scalar_param", lb=0, ub=1,
        evaluation_objects=[eval_obj_1],
    )
    op.add_variable(
        "scalar_eval_obj_2", parameter_path="scalar_param", lb=0, ub=1,
        evaluation_objects=[eval_obj_2],
    )
    op.add_variable("scalar_both_eval_obj", parameter_path="scalar_param_2", lb=0, ub=1)
    op.add_objective(single_obj_1)
    op.add_objective(single_obj_2, evaluation_objects=eval_obj_1)
    op.add_objective(multi_obj, n_objectives=2)
    return op


def test_multi_eval_obj_evaluation(op_with_multiple_eval_objects):
    f = op_with_multiple_eval_objects.evaluate_objectives([1, 1, 1])
    np.testing.assert_allclose(f, [0, 0, 1, 2, 3, 2, 3])

    f_batch = op_with_multiple_eval_objects.evaluate_objectives([[1, 1, 1], [1, 1, 1]])
    np.testing.assert_allclose(f_batch, [[0, 0, 1, 2, 3, 2, 3], [0, 0, 1, 2, 3, 2, 3]])


def test_multi_eval_obj_names(op_with_multiple_eval_objects):
    assert op_with_multiple_eval_objects.objective_names == [
        "single_obj_1", "single_obj_2", "multi_obj",
    ]


def test_multi_eval_obj_labels(op_with_multiple_eval_objects):
    # Label order is contractual: per-eval-obj expansion is part of the API.
    expected = [
        "foo_single_obj_1",
        "bar_single_obj_1",
        "single_obj_2",
        "foo_multi_obj_0",
        "bar_multi_obj_0",
        "foo_multi_obj_1",
        "bar_multi_obj_1",
    ]
    assert op_with_multiple_eval_objects.objective_labels == expected


# ── Round-trip: set_variables → eval object → get_variable_value ─────────────


def test_set_variables_propagates_to_eval_object(op_scalar, eval_obj):
    op_scalar.set_variables([1.5])
    assert eval_obj.scalar_param == pytest.approx(1.5)


def test_get_variable_value_matches_eval_object(op_scalar, eval_obj):
    op_scalar.set_variables([0.8])
    assert op_scalar.get_variable_value("scalar_param") == pytest.approx(0.8)
    assert eval_obj.scalar_param == pytest.approx(0.8)


def test_get_variable_value_no_path_returns_none():
    op = OptimizationProblem("no_path")
    op.add_variable("abstract", lb=0, ub=1, evaluation_objects=None)
    op.set_variables([0.5])
    assert op.get_variable_value("abstract") is None


def test_get_variable_value_unknown_name_raises():
    op = OptimizationProblem("empty")
    with pytest.raises(KeyError):
        op.get_variable_value("ghost")


def test_indexed_variable_write_through():
    obj = EvaluationObject()
    op = OptimizationProblem("indexed")
    op.add_evaluation_object(obj)
    op.add_variable("sized_list_param", lb=0, ub=10, indices=0)
    op.set_variables([4.0])
    assert obj.sized_list_param[0] == pytest.approx(4.0)
    assert op.get_variable_value("sized_list_param") == pytest.approx(4.0)


# ── evaluate_meta_scores ──────────────────────────────────────────────────────


def test_evaluate_meta_scores_receives_eval_object(op_scalar, eval_obj):
    """Meta-score function receives the evaluation object, not raw x."""
    received = []

    def meta(obj):
        received.append(obj)
        return obj.scalar_param

    op_scalar.add_meta_score(meta, n_meta_scores=1)
    result = op_scalar.evaluate_meta_scores([0.7])

    assert len(received) == 1
    assert received[0] is eval_obj
    np.testing.assert_allclose(result, [0.7])


def test_evaluate_meta_scores_population_shape(op_scalar):
    """2-D input returns shape (n_individuals, n_meta_scores)."""
    op_scalar.add_meta_score(lambda obj: obj.scalar_param, n_meta_scores=1)
    result = op_scalar.evaluate_meta_scores([[0.5], [1.0]])
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[0.5], [1.0]])


def test_evaluate_meta_scores_no_meta_scores_returns_empty_columns():
    """No meta scores registered returns shape (n_individuals, 0)."""
    op = OptimizationProblem("no_meta")
    op.add_variable("v", lb=0, ub=1, evaluation_objects=None)
    result = op.evaluate_meta_scores([[0.5], [0.7]])
    assert result.shape == (2, 0)


# ── check_individual (feasibility) ────────────────────────────────────────────


def test_check_individual_feasible(op_with_linear_constraint):
    x0 = op_with_linear_constraint.create_initial_values(1, seed=1)[0]
    assert op_with_linear_constraint.check_individual(x0)


def test_check_individual_infeasible_bounds(op_with_linear_constraint):
    assert not op_with_linear_constraint.check_individual([-1, 0.5])


def test_check_individual_infeasible_lincon(op_with_linear_constraint):
    # x0 > x1 violates [1, -1] @ x <= 0
    assert not op_with_linear_constraint.check_individual([0.9, 0.1])


def test_check_individual_with_nonlinear_constraint():
    op = OptimizationProblem("nlc", use_diskcache=False)
    op.add_variable("x", lb=0, ub=2)
    op.add_objective(lambda x: x[0] ** 2)
    op.add_nonlinear_constraint(
        lambda x: [x[0] - 1.5], n_nonlinear_constraints=1, bounds=0,
    )
    # feasible
    assert op.check_individual([1.0])
    # infeasible
    assert not op.check_individual([1.8])


def test_check_individual_nonlinear_constraint_exception():
    op = OptimizationProblem("nlc_exc", use_diskcache=False)
    op.add_variable("x", lb=0, ub=2, evaluation_objects=None)
    op.add_objective(lambda x: x[0])
    op.add_nonlinear_constraint(
        lambda x: (_ for _ in ()).throw(RuntimeError("boom")),
        n_nonlinear_constraints=1,
    )
    assert not op.check_individual([1.0])


# ── Evaluator registration ───────────────────────────────────────────────────


def test_add_evaluator_basic():
    op = OptimizationProblem("ev", use_diskcache=False)

    def my_evaluator(x):
        return x

    op.add_evaluator(my_evaluator)
    assert "my_evaluator" in op.evaluators_dict
    assert op.evaluators_dict_reference[my_evaluator] == "my_evaluator"


def test_add_evaluator_non_callable_raises():
    op = OptimizationProblem("ev", use_diskcache=False)
    with pytest.raises(TypeError):
        op.add_evaluator("not_callable")


def test_add_evaluator_duplicate_name_raises():
    op = OptimizationProblem("ev", use_diskcache=False)
    op.add_evaluator(lambda x: x, name="ev1")
    with pytest.raises(CADETProcessError):
        op.add_evaluator(lambda x: x, name="ev1")


def test_add_evaluator_callable_class_name():
    class MyEvaluator:
        def __call__(self, x):
            return x

    op = OptimizationProblem("ev", use_diskcache=False)
    ev = MyEvaluator()
    op.add_evaluator(ev)
    assert str(ev) in op.evaluators_dict


def test_add_evaluator_with_args_kwargs():
    """Evaluator with fixed args/kwargs passes them through the chain."""
    def ev_with_args(eval_obj, scale, offset=0):
        return eval_obj.scalar_param * scale + offset

    obj = EvaluationObject()
    op = OptimizationProblem("ev", use_diskcache=False)
    op.add_evaluation_object(obj)
    op.add_variable("scalar_param", lb=0, ub=10)
    op.add_evaluator(ev_with_args, args=(2,), kwargs={"offset": 1})
    op.add_objective(lambda res: res, requires=ev_with_args)
    # scalar_param=3 → ev_with_args(obj, 2, offset=1) → 3*2+1 = 7
    result = op.evaluate_objectives([3.0])
    np.testing.assert_allclose(result, [7.0])


# ── Objective registration ───────────────────────────────────────────────────


def test_add_objective_non_callable_raises():
    op = OptimizationProblem("obj", use_diskcache=False)
    with pytest.raises(TypeError):
        op.add_objective("not_callable")


def test_add_objective_duplicate_name_warns():
    op = OptimizationProblem("obj", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    op.add_objective(lambda x: x[0], name="f")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        op.add_objective(lambda x: x[0], name="f")
    assert any("already exists" in str(wi.message) for wi in w)


def test_add_objective_unknown_eval_obj_raises():
    obj = EvaluationObject()
    op = OptimizationProblem("obj", use_diskcache=False)
    with pytest.raises(CADETProcessError, match="Unknown EvaluationObject"):
        op.add_objective(lambda x: 0, evaluation_objects=obj)


def test_add_objective_unknown_evaluator_raises():
    op = OptimizationProblem("obj", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)

    def fake_evaluator(x):
        return x

    with pytest.raises(CADETProcessError, match="Unknown Evaluator"):
        op.add_objective(lambda x: 0, requires=fake_evaluator)


def test_add_objective_with_evaluator_chain():
    obj = EvaluationObject()
    op = OptimizationProblem("chain", use_diskcache=False)
    op.add_evaluation_object(obj)
    op.add_variable("scalar_param", lb=0, ub=2)

    def sim(eval_obj):
        return eval_obj.scalar_param * 2

    def score(sim_result):
        return sim_result + 1

    op.add_evaluator(sim)
    op.add_evaluator(score)
    op.add_objective(lambda res: res, requires=[sim, score])

    result = op.evaluate_objectives([1.0])
    np.testing.assert_allclose(result, [3.0])


def test_add_objective_with_eval_objects_minus_one():
    obj1 = EvaluationObject(name="a")
    obj2 = EvaluationObject(name="b")
    op = OptimizationProblem("all", use_diskcache=False)
    op.add_evaluation_object(obj1)
    op.add_evaluation_object(obj2)
    op.add_variable("scalar_param", lb=0, ub=2)
    op.add_objective(lambda eo: eo.scalar_param, evaluation_objects=-1)
    result = op.evaluate_objectives([1.5])
    np.testing.assert_allclose(result, [1.5, 1.5])


# ── Nonlinear constraint registration ────────────────────────────────────────


def test_add_nonlinear_constraint_non_callable_raises():
    op = OptimizationProblem("nlc", use_diskcache=False)
    with pytest.raises(TypeError):
        op.add_nonlinear_constraint("not_callable", n_nonlinear_constraints=1)


def test_add_nonlinear_constraint_unknown_eval_obj_raises():
    obj = EvaluationObject()
    op = OptimizationProblem("nlc", use_diskcache=False)
    with pytest.raises(CADETProcessError, match="Unknown EvaluationObject"):
        op.add_nonlinear_constraint(
            lambda x: [0], n_nonlinear_constraints=1, evaluation_objects=obj,
        )


def test_add_nonlinear_constraint_unknown_evaluator_raises():
    op = OptimizationProblem("nlc", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)

    def fake(x):
        return x

    with pytest.raises(CADETProcessError, match="Unknown Evaluator"):
        op.add_nonlinear_constraint(
            lambda x: [0], n_nonlinear_constraints=1, requires=fake,
        )


def test_evaluate_nonlinear_constraints_violation():
    op = OptimizationProblem("nlc", use_diskcache=False)
    op.add_variable("x", lb=0, ub=2, evaluation_objects=None)
    op.add_objective(lambda x: x[0])
    op.add_nonlinear_constraint(
        lambda x: [x[0] - 1], n_nonlinear_constraints=1, bounds=0,
    )
    cv = op.evaluate_nonlinear_constraints_violation([1.5])
    np.testing.assert_allclose(cv, [0.5])


# ── Callback registration ────────────────────────────────────────────────────


def test_add_callback_non_callable_raises():
    op = OptimizationProblem("cb", use_diskcache=False)
    with pytest.raises(TypeError):
        op.add_callback("not_callable")


def test_add_callback_duplicate_name_raises():
    op = OptimizationProblem("cb", use_diskcache=False)
    op.add_callback(lambda pop: None, name="cb1")
    with pytest.raises(CADETProcessError):
        op.add_callback(lambda pop: None, name="cb1")


def _make_dummy_population():
    pop = Population()
    pop.add_individual(Individual(x=np.array([])))
    return pop


def test_evaluate_callbacks_frequency_skip():
    calls = []
    op = OptimizationProblem("cb", use_diskcache=False)
    op.add_callback(lambda result: calls.append(1), name="cb", frequency=3)
    pop = _make_dummy_population()
    op.evaluate_callbacks(population=pop, current_iteration=1)
    assert len(calls) == 0
    op.evaluate_callbacks(population=pop, current_iteration=3)
    assert len(calls) == 1


def test_evaluate_callbacks_population_deprecation():
    op = OptimizationProblem("cb", use_diskcache=False)
    op.add_callback(lambda result: None, name="cb")
    with pytest.warns(DeprecationWarning, match="evaluate_callbacks_population"):
        op.evaluate_callbacks_population(population=_make_dummy_population(), current_iteration=0)


def test_evaluate_callbacks_exception_logged(caplog):
    op = OptimizationProblem("cb", use_diskcache=False)
    op.add_callback(lambda result: 1 / 0, name="boom")
    op.evaluate_callbacks(population=_make_dummy_population(), current_iteration=0)
    assert "boom" in caplog.text


# ── Evaluator chain failure handling ─────────────────────────────────────────


def test_evaluate_objectives_returns_bad_on_evaluator_failure():
    obj = EvaluationObject()
    op = OptimizationProblem("fail", use_diskcache=False)
    op.add_evaluation_object(obj)
    op.add_variable("scalar_param", lb=0, ub=2)

    def bad_evaluator(eval_obj):
        raise RuntimeError("boom")

    op.add_evaluator(bad_evaluator)
    op.add_objective(
        lambda res: res, requires=bad_evaluator, bad_metrics=999,
    )
    result = op.evaluate_objectives([1.0])
    np.testing.assert_allclose(result, [999])


def test_evaluate_objectives_inline_no_chain():
    """Objective without evaluator chain receives x directly."""
    op = OptimizationProblem("inline", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    op.add_objective(lambda x: x[0] ** 2)
    result = op.evaluate_objectives([0.5])
    np.testing.assert_allclose(result, [0.25])


# ── Variable edge cases ──────────────────────────────────────────────────────


def test_add_variable_parameter_path_without_eval_obj_raises():
    op = OptimizationProblem("path", use_diskcache=False)
    with pytest.raises(ValueError, match="parameter_path"):
        op.add_variable("x", parameter_path="some.path", evaluation_objects=None)


def test_remove_variable_raises():
    op = OptimizationProblem("rm", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    with pytest.raises(NotImplementedError):
        op.remove_variable("x")


# ── Variable types ────────────────────────────────────────────────────────────


def test_add_integer_variable():
    op = OptimizationProblem("int", use_diskcache=False)
    param = op.add_variable("n", lb=1, ub=10, parameter_type=int, evaluation_objects=None)
    assert param.parameter_type is int
    assert param.lb == 1
    assert param.ub == 10


def test_add_choice_variable():
    op = OptimizationProblem("choice", use_diskcache=False)
    param = op.add_choice_variable("mode", ["fast", "slow"], evaluation_objects=None)
    assert param.valid_values == ["fast", "slow"]
    assert param.name == "mode"


def test_add_choice_variable_duplicate_raises():
    op = OptimizationProblem("dup", use_diskcache=False)
    op.add_choice_variable("mode", ["a", "b"], evaluation_objects=None)
    with pytest.raises(CADETProcessError, match="already exists"):
        op.add_choice_variable("mode", ["c", "d"], evaluation_objects=None)


def test_typed_variable_subsets():
    op = OptimizationProblem("types", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    op.add_variable("n", lb=1, ub=10, parameter_type=int, evaluation_objects=None)
    op.add_choice_variable("mode", ["a", "b"], evaluation_objects=None)

    assert len(op.continuous_variables) == 1
    assert op.continuous_variables[0].name == "x"
    assert op.n_continuous_variables == 1

    assert len(op.integer_variables) == 1
    assert op.integer_variables[0].name == "n"
    assert op.n_integer_variables == 1

    assert len(op.categorical_variables) == 1
    assert op.categorical_variables[0].name == "mode"
    assert op.n_categorical_variables == 1


def test_add_choice_variable_with_eval_object(eval_obj):
    op = OptimizationProblem("choice_eval", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    param = op.add_choice_variable(
        "scalar_param", [1.0, 2.0, 3.0],
    )
    assert param.valid_values == [1.0, 2.0, 3.0]


def test_optimizer_rejects_integer_variables():
    from CADETProcess.optimization.scipyAdapter import NelderMead

    op = OptimizationProblem("int_reject", use_diskcache=False)
    op.add_variable("n", lb=1, ub=10, parameter_type=int, evaluation_objects=None)
    op.add_objective(lambda x: x[0])

    optimizer = NelderMead()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = optimizer.check_optimization_problem(op)
    assert not result
    assert any("integer" in str(warning.message).lower() for warning in w)


def test_optimizer_rejects_categorical_variables():
    from CADETProcess.optimization.scipyAdapter import NelderMead

    op = OptimizationProblem("cat_reject", use_diskcache=False)
    op.add_choice_variable("mode", ["a", "b"], evaluation_objects=None)
    op.add_objective(lambda x: 0.0)

    optimizer = NelderMead()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = optimizer.check_optimization_problem(op)
    assert not result
    assert any("categorical" in str(warning.message).lower() for warning in w)


def test_optimizer_rejects_unbounded_parameters():
    from CADETProcess.optimization.scipyAdapter import NelderMead

    op = OptimizationProblem("unbound_reject", use_diskcache=False)
    op.add_variable("x", lb=0, ub=float("inf"), evaluation_objects=None)
    op.add_objective(lambda x: x[0])

    optimizer = NelderMead()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = optimizer.check_optimization_problem(op)
    assert not result
    assert any("unbounded" in str(warning.message).lower() for warning in w)


def test_optimizer_supporting_unbounded_requires_x0():
    from CADETProcess.optimization.optimizer import OptimizerBase

    class UnboundedOptimizer(OptimizerBase):
        supports_single_objective = True
        supports_bounds = True
        supports_unbounded_parameters = True

        def _run(self, op, x0, *args, **kwargs):
            pass

    op = OptimizationProblem("unbound_x0", use_diskcache=False)
    op.add_variable("x", lb=0, ub=float("inf"), evaluation_objects=None)
    op.add_objective(lambda x: x[0])

    optimizer = UnboundedOptimizer()
    with pytest.raises(ValueError, match="x0 must be provided"):
        optimizer.optimize(op, x0=None)


# ── Multi-criteria decision functions ─────────────────────────────────────────


def test_evaluate_multi_criteria_decision_functions_empty():
    op = OptimizationProblem("mcdf", use_diskcache=False)
    assert op.evaluate_multi_criteria_decision_functions() == []


def test_evaluate_multi_criteria_decision_functions():
    op = OptimizationProblem("mcdf", use_diskcache=False)
    op.add_multi_criteria_decision_function(lambda front: [0, 2])
    result = op.evaluate_multi_criteria_decision_functions(pareto_front="dummy")
    assert result == [0, 2]


# ── create_population ─────────────────────────────────────────────────────────


def test_create_population_basic():
    op = OptimizationProblem("pop", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    op.add_variable("y", lb=0, ub=1, evaluation_objects=None)
    pop = op.create_population([[0.2, 0.3], [0.4, 0.5]])
    assert len(pop) == 2


def test_create_population_with_transform():
    op = OptimizationProblem("pop", use_diskcache=False)
    op.add_variable("x", lb=-2, ub=2, normalization="linear")
    # x_t = 0.75 → x_p = 4*0.75 - 2 = 1.0
    pop = op.create_population([[0.75]], untransform=True)
    np.testing.assert_allclose(pop.x[0], [1.0], atol=1e-10)


def test_create_population_with_dependent_values():
    op = OptimizationProblem("pop", use_diskcache=False)
    op.add_variable("a", lb=0, ub=1)
    op.add_variable("b", lb=0, ub=1)
    op.add_variable_dependency("b", "a", lambda v: v * 2)
    pop = op.create_population([[0.3]], get_dependent_values=True)
    np.testing.assert_allclose(pop.x[0], [0.3, 0.6], atol=1e-10)


# ── String representations ───────────────────────────────────────────────────


def test_str_and_repr():
    op = OptimizationProblem("myname", use_diskcache=False)
    op.add_variable("x", lb=0, ub=1, evaluation_objects=None)
    assert str(op) == "myname"
    assert "myname" in repr(op)
    assert "n_variables=1" in repr(op)


# ── Property accessors ───────────────────────────────────────────────────────


def test_property_accessors(op_scalar):
    assert op_scalar.n_variables == 1
    assert op_scalar.n_independent_variables == 1
    assert op_scalar.n_dependent_variables == 0
    assert op_scalar.independent_variable_names == ["scalar_param"]
    assert op_scalar.dependent_variable_names == []
    assert len(op_scalar.independent_variables) == 1
    assert len(op_scalar.dependent_variables) == 0
    assert len(op_scalar.variables) == 1


# ── Callbacks ────────────────────────────────────────────────────────────────


def test_evaluate_callbacks_called_per_individual(eval_obj):
    """evaluate_callbacks must call the callback once per individual per eval object.

    Before the fix, evaluate_callbacks passed the whole population as the first
    positional argument and never iterated per individual, so the callback was
    called with the wrong signature and always failed silently.
    """
    op = OptimizationProblem("cb_test", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=1)

    def evaluator(evaluation_object):
        return evaluation_object.scalar_param * 2

    op.add_evaluator(evaluator)

    calls = []

    def callback(result, individual, evaluation_object, callbacks_dir):
        calls.append((result, individual, evaluation_object, callbacks_dir))

    op.add_callback(callback, requires=[evaluator])

    ind = Individual(x=np.array([0.5]))
    pop = Population()
    pop.add_individual(ind)

    op.evaluate_callbacks(pop, current_iteration=0)

    assert len(calls) == 1
    result, called_ind, called_eval_obj, called_dir = calls[0]
    assert called_ind is ind
    assert called_eval_obj is eval_obj
    assert called_dir is None
    np.testing.assert_allclose(result, 1.0)


def test_evaluate_callbacks_optional_args_injected_by_signature(eval_obj):
    """Callbacks that omit optional context args must still be called correctly."""
    op = OptimizationProblem("cb_minimal", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=1)

    def evaluator(evaluation_object):
        return evaluation_object.scalar_param

    op.add_evaluator(evaluator)

    calls = []

    def callback(result):
        calls.append(result)

    op.add_callback(callback, requires=[evaluator])

    ind = Individual(x=np.array([0.3]))
    pop = Population()
    pop.add_individual(ind)

    op.evaluate_callbacks(pop, current_iteration=0)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], 0.3)


def test_evaluate_callbacks_no_chain_sees_current_x(eval_obj):
    """No-chain callback must receive eval_obj with parameters set for individual.x.

    Regression: the pipeline refactor dropped set_values for the no-chain path,
    so eval_obj attributes reflected the previous individual's x instead.
    """
    op = OptimizationProblem("cb_no_chain", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=1)

    seen_values = []

    def callback(result):
        seen_values.append(result.scalar_param)

    op.add_callback(callback)

    for x_val in [0.2, 0.7]:
        ind = Individual(x=np.array([x_val]))
        pop = Population()
        pop.add_individual(ind)
        op.evaluate_callbacks(pop, current_iteration=0)

    np.testing.assert_allclose(seen_values, [0.2, 0.7])


def test_evaluate_callbacks_callbacks_dir_passthrough(eval_obj, tmp_path):
    """callbacks_dir passed to evaluate_callbacks reaches the callback function."""
    op = OptimizationProblem("cb_dir", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=1)

    received_dirs = []

    def callback(result, callbacks_dir):
        received_dirs.append(callbacks_dir)

    op.add_callback(callback)

    ind = Individual(x=np.array([0.5]))
    pop = Population()
    pop.add_individual(ind)

    op.evaluate_callbacks(pop, current_iteration=0, callbacks_dir=tmp_path)

    assert len(received_dirs) == 1
    assert received_dirs[0] == tmp_path


def test_evaluate_callbacks_pipeline_cache_reuse(eval_obj):
    """Evaluator runs once even when both an objective and a callback require it."""
    op = OptimizationProblem("cb_cache", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=1)

    call_count = 0

    def evaluator(evaluation_object):
        nonlocal call_count
        call_count += 1
        return evaluation_object.scalar_param * 2

    op.add_evaluator(evaluator)
    op.add_objective(lambda result: result, requires=[evaluator])

    cb_results = []

    def callback(result):
        cb_results.append(result)

    op.add_callback(callback, requires=[evaluator])

    ind = Individual(x=np.array([0.5]))
    pop = Population()
    pop.add_individual(ind)

    op.evaluate_objectives(ind.x)
    op.evaluate_callbacks(pop, current_iteration=0)

    # Evaluator must have run exactly once; callback gets the cached result.
    assert call_count == 1
    assert len(cb_results) == 1
    np.testing.assert_allclose(cb_results[0], 1.0)


# ── create_individual cv fields (T7) ─────────────────────────────────────────


def test_create_individual_cv_bounds_feasible(op_basic):
    op_basic.add_objective(lambda x: x[0])
    ind = op_basic.create_individual([0.5, 5.0], f=[0.5], f_minimized=[0.5])
    assert np.all(ind.cv_bounds <= 0)


def test_create_individual_cv_bounds_upper_violated(op_basic):
    op_basic.add_objective(lambda x: x[0])
    n = op_basic.n_variables
    ind = op_basic.create_individual([1.5, 5.0], f=[1.5], f_minimized=[1.5])
    assert ind.cv_bounds[n + 0] > 0   # upper violation on var_0
    assert ind.cv_bounds[n + 1] <= 0  # var_1 fine


def test_create_individual_cv_lincon_violated(op_with_linear_constraint):
    op_with_linear_constraint.add_objective(lambda x: x[0])
    # var_0=0.8 > var_1=0.3 violates var_0 - var_1 <= 0
    ind = op_with_linear_constraint.create_individual(
        [0.8, 0.3], f=[0.8], f_minimized=[0.8]
    )
    assert ind.cv_lincon[0] > 0


def test_create_individual_cv_lincon_feasible(op_with_linear_constraint):
    op_with_linear_constraint.add_objective(lambda x: x[0])
    ind = op_with_linear_constraint.create_individual(
        [0.3, 0.5], f=[0.3], f_minimized=[0.3]
    )
    assert ind.cv_lincon[0] <= 0
