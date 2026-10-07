"""All-case registrations follow cases added during problem construction."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.optimization import OptimizationProblem

from tests.optimization.conftest import EvaluationObject


@pytest.mark.parametrize("initial_cases", [0, 1, 2])
@pytest.mark.parametrize("mode", ["scalar", "indexed", "preprocessing", "choice"])
def test_all_case_variables_write_to_later_cases(initial_cases, mode):
    op = OptimizationProblem("growing", use_diskcache=False)
    cases = [EvaluationObject(name=name) for name in ("a", "b", "c")]
    for case in cases[:initial_cases]:
        op.add_evaluation_object(case)
    if mode == "choice":
        op.add_choice_variable("scalar_param", [1, 2], targets=-1)
    else:
        kwargs = {
            "scalar": {},
            "indexed": {"parameter_path": "sized_list_param", "indices": 0},
            "preprocessing": {"pre_processing": lambda value: 2 * value},
        }[mode]
        op.add_variable("scalar_param", targets=-1, **kwargs)
    for case in cases[initial_cases:]:
        op.add_evaluation_object(case)
    if mode == "choice":
        op.parameter_space.set_values({"scalar_param": 2})
    else:
        op.set_variables([2])
    for case in cases:
        if mode == "indexed":
            np.testing.assert_allclose(case.sized_list_param, [2, 2])
        else:
            assert case.scalar_param == (4 if mode == "preprocessing" else 2)


def test_explicit_and_free_variable_targets_stay_fixed():
    op = OptimizationProblem("fixed", use_diskcache=False)
    first, later = EvaluationObject("a"), EvaluationObject("b")
    op.add_evaluation_object(first)
    op.add_variable("scalar_param", targets=[first])
    op.add_variable("scalar_param_2", targets=None)
    op.add_evaluation_object(later)
    op.set_variables([2, 3])
    assert first.scalar_param == 2
    assert later.scalar_param == 1
    assert first.scalar_param_2 == later.scalar_param_2 == 1


@pytest.mark.parametrize("initial_cases", [0, 1, 2])
@pytest.mark.parametrize("kind", ["objective", "nonlinear_constraint", "meta_score"])
def test_all_case_metrics_expand_values_labels_and_bounds(initial_cases, kind):
    op = OptimizationProblem("metrics", use_diskcache=False)
    cases = [EvaluationObject(name=name) for name in ("a", "b", "c")]
    for i, case in enumerate(cases):
        case.scalar_param = i + 1
    for case in cases[:initial_cases]:
        op.add_evaluation_object(case)
    op.add_variable("free", targets=None)
    kwargs = {f"n_{kind}s": 2}
    if kind == "nonlinear_constraint":
        kwargs.update(bounds=[1, 5], comparison_operator="ge")
    getattr(op, f"add_{kind}")(
        lambda case: [case.scalar_param, case.scalar_param + 10],
        name="scores",
        labels=["low", "high"],
        evaluation_objects=-1,
        **kwargs,
    )
    metric = op.metric_space.metrics_dict["scores"]
    assert metric.n_metrics == 2 * max(initial_cases, 1)
    for i, case in enumerate(cases[initial_cases:], start=initial_cases + 1):
        op.add_evaluation_object(case)
        assert metric.n_metrics == 2 * i
    # Entries and labels both follow case registration order, then entry order.
    assert metric.labels == [
        f"{name}_{entry}" for name in ("a", "b", "c") for entry in ("low", "high")
    ]
    actual = getattr(op, f"evaluate_{kind}s")([0])
    np.testing.assert_allclose(actual, [1, 11, 2, 12, 3, 13])
    if kind == "nonlinear_constraint":
        np.testing.assert_allclose(op.nonlinear_constraints[0].bounds, [1, 5] * 3)
        np.testing.assert_allclose(
            op.evaluate_nonlinear_constraints_violation([0]), [0, -6, -1, -7, -2, -8]
        )


@pytest.mark.parametrize("all_cases", [False, True])
def test_callback_coverage_after_case_registration(all_cases):
    op = OptimizationProblem("callbacks", use_diskcache=False)
    first, later = EvaluationObject("a"), EvaluationObject("b")
    op.add_evaluation_object(first)
    op.add_variable("scalar_param", targets=-1)
    calls = []
    op.add_callback(
        lambda case: calls.append((str(case), case.scalar_param)),
        name="record",
        evaluation_objects=-1 if all_cases else [first],
    )
    op.add_evaluation_object(later)
    op.evaluate_callbacks(op.create_population([[2]]), current_iteration=0)
    assert calls == ([("a", 2), ("b", 2)] if all_cases else [("a", 2)])


@pytest.mark.parametrize("kind", ["objective", "nonlinear_constraint", "meta_score"])
def test_collectors_keep_one_output_when_cases_are_added(kind):
    op = OptimizationProblem("collector", use_diskcache=False)
    op.add_evaluation_object(EvaluationObject("a"))
    op.add_variable("scalar_param", targets=-1)
    evaluator = lambda case: case.scalar_param
    op.add_evaluator(evaluator, name="read")
    getattr(op, f"add_{kind}")(
        sum,
        name="total",
        requires=evaluator,
        per_object=False,
        evaluation_objects=-1,
    )
    np.testing.assert_allclose(getattr(op, f"evaluate_{kind}s")([2]), [2])
    op.add_evaluation_object(EvaluationObject("b"))
    assert op.metric_space.metrics_dict["total"].n_metrics == 1
    np.testing.assert_allclose(getattr(op, f"evaluate_{kind}s")([2]), [4])


def test_invalid_later_case_does_not_partially_register():
    op = OptimizationProblem("invalid", use_diskcache=False)
    first = EvaluationObject("a")
    op.add_evaluation_object(first)
    op.add_variable("scalar_param", targets=-1)
    with pytest.raises(CADETProcessError, match="not a valid parameter"):
        op.add_evaluation_object(SimpleNamespace(other=1))
    assert op.evaluation_objects == [first]
    op.add_evaluation_object(EvaluationObject("b"))
    op.set_variables([2])
    assert [case.scalar_param for case in op.evaluation_objects] == [2, 2]


def test_all_case_variable_expansion_survives_pickle():
    op = OptimizationProblem("pickle", use_diskcache=False)
    op.add_evaluation_object(EvaluationObject("a"))
    op.add_variable("scalar_param", targets=-1)
    restored = pickle.loads(pickle.dumps(op))
    restored.add_evaluation_object(EvaluationObject("b"))
    restored.set_variables([2])
    assert [case.scalar_param for case in restored.evaluation_objects] == [2, 2]


@pytest.mark.parametrize("kind", ["objective", "nonlinear_constraint", "meta_score"])
def test_explicit_metric_subset_stays_fixed_after_more_cases(kind):
    op = OptimizationProblem("subset", use_diskcache=False)
    first = EvaluationObject("a")
    first.scalar_param = 2
    op.add_evaluation_object(first)
    op.add_variable("free", targets=None)
    getattr(op, f"add_{kind}")(
        lambda case: case.scalar_param,
        name="score",
        evaluation_objects=[first],
    )
    for name in ("b", "c"):
        op.add_evaluation_object(EvaluationObject(name))
    assert op.metric_space.metrics_dict["score"].n_metrics == 1
    np.testing.assert_allclose(getattr(op, f"evaluate_{kind}s")([0]), [2])


def test_none_metric_does_not_gain_a_case_dimension():
    op = OptimizationProblem("no_case", use_diskcache=False)
    op.add_variable("free", targets=None)
    op.add_objective(lambda x: x[0], name="score", evaluation_objects=None)
    op.add_evaluation_object(EvaluationObject("a"))
    op.add_evaluation_object(EvaluationObject("b"))
    assert op.n_objectives == 1
    assert op.metric_space.metrics_dict["score"].dims is None


def test_new_case_cannot_turn_explicit_collector_domain_into_subset():
    op = OptimizationProblem("subset_collector", use_diskcache=False)
    first = EvaluationObject("a")
    op.add_evaluation_object(first)
    evaluator = lambda case: case.scalar_param
    op.add_evaluator(evaluator, name="read")
    op.add_objective(
        sum,
        name="total",
        requires=evaluator,
        per_object=False,
        evaluation_objects=[first],
    )
    with pytest.raises(CADETProcessError, match="subset"):
        op.add_evaluation_object(EvaluationObject("b"))
    assert op.evaluation_objects == [first]


@pytest.mark.parametrize("pickle_roundtrip", [False, True])
def test_new_case_rejects_colliding_all_case_and_explicit_target_writes(pickle_roundtrip):
    op = OptimizationProblem("collision", use_diskcache=False)
    first, later = EvaluationObject("a"), EvaluationObject("b")
    op.add_evaluation_object(first)
    op.add_variable("all", parameter_path="scalar_param", targets=-1)
    op.add_variable("explicit", parameter_path="scalar_param", targets=[later])
    if pickle_roundtrip:
        op, first, later = pickle.loads(pickle.dumps((op, first, later)))
    with pytest.raises(CADETProcessError, match="already registered"):
        op.add_evaluation_object(later)
    assert op.evaluation_objects == [first]
    op.set_variables([2, 3])
    assert first.scalar_param == 2
    assert later.scalar_param == 3


def test_all_case_callback_downstream_of_collector_includes_later_cases():
    op = OptimizationProblem("collector_callback", use_diskcache=False)
    op.add_evaluation_object(EvaluationObject("a"))
    op.add_variable("scalar_param", targets=-1)
    read = lambda case: case.scalar_param
    op.add_evaluator(read, name="read")
    op.add_evaluator(sum, name="total", per_object=False)
    calls = []
    op.add_callback(
        lambda total: calls.append(total),
        name="record",
        requires=[read, sum],
    )
    op.add_evaluation_object(EvaluationObject("b"))
    op.evaluate_callbacks(op.create_population([[2]]), current_iteration=0)
    np.testing.assert_allclose(calls, [4])
