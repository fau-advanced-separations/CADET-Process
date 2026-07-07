from dataclasses import dataclass

import numpy as np
import pytest
from CADETProcess.evaluation_pipeline import EvaluationFailure, EvaluationPipeline
from CADETProcess.metric_space import Metric, MetricSpace
from CADETProcess.parameter_space import ParameterSpace, RangedParameter
from CADETProcess.problem import EvaluationBackend, Problem

# ── Fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class Model:
    """Minimal evaluation object: a single mutable value."""

    value: float = 0.0


class StubBackend:
    """Backend returning a fixed result dict."""

    def __init__(self, results):
        self.results = results

    def evaluate(self, assignment):
        return dict(self.results)


@pytest.fixture
def yield_and_purity_space():
    space = MetricSpace()
    space.add_objective(Metric("yield", n_metrics=2), minimize=False)
    space.add_constraint(Metric("purity"), bound=0.95, comparison_operator="ge")
    return space


# ── EvaluationBackend protocol ────────────────────────────────────────────────


def test_evaluation_pipeline_satisfies_backend_protocol():
    pipeline = EvaluationPipeline(ParameterSpace())
    assert isinstance(pipeline, EvaluationBackend)


def test_stub_backend_satisfies_backend_protocol():
    assert isinstance(StubBackend({}), EvaluationBackend)


def test_object_without_evaluate_rejected():
    with pytest.raises(TypeError, match="EvaluationBackend"):
        Problem(backend="not a backend")


# ── Construction ──────────────────────────────────────────────────────────────


def test_empty_problem_creates_fresh_spaces():
    problem = Problem()
    assert problem.parameter_space.n_parameters == 0
    assert problem.metric_space.n_metrics == 0
    assert problem.backend is None


def test_non_parameter_space_rejected():
    with pytest.raises(TypeError, match="ParameterSpace"):
        Problem(parameter_space="not a space")


def test_non_metric_space_rejected():
    with pytest.raises(TypeError, match="MetricSpace"):
        Problem(metric_space="not a space")


# ── Evaluation ────────────────────────────────────────────────────────────────


def test_evaluate_without_backend_raises(yield_and_purity_space):
    problem = Problem(metric_space=yield_and_purity_space)
    with pytest.raises(RuntimeError, match="backend"):
        problem.evaluate({})


def test_evaluate_returns_declared_metrics_and_drops_intermediates(
    yield_and_purity_space,
):
    backend = StubBackend(
        {"yield": [0.8, 0.9], "purity": 0.99, "simulation_results": object()}
    )
    problem = Problem(metric_space=yield_and_purity_space, backend=backend)
    results = problem.evaluate({})
    assert list(results) == ["yield", "purity"]
    np.testing.assert_array_equal(results["yield"], [0.8, 0.9])


def test_evaluate_missing_declared_metric_raises(yield_and_purity_space):
    problem = Problem(
        metric_space=yield_and_purity_space,
        backend=StubBackend({"yield": [0.8, 0.9]}),
    )
    with pytest.raises(ValueError, match="purity"):
        problem.evaluate({})


def test_evaluate_wrong_shape_raises(yield_and_purity_space):
    problem = Problem(
        metric_space=yield_and_purity_space,
        backend=StubBackend({"yield": [0.8], "purity": 0.99}),
    )
    with pytest.raises(ValueError, match="shape"):
        problem.evaluate({})


def test_evaluation_failure_passes_through_unvalidated(yield_and_purity_space):
    failure = EvaluationFailure(stage="simulate", reason="diverged")
    problem = Problem(
        metric_space=yield_and_purity_space,
        backend=StubBackend({"yield": failure, "purity": 0.99}),
    )
    results = problem.evaluate({})
    assert results["yield"] is failure
    assert results["purity"] == pytest.approx(0.99)


def test_evaluate_with_real_pipeline_backend():
    """End-to-end: assignment reaches the model, pipeline output is validated."""
    model = Model()
    parameter_space = ParameterSpace()
    parameter_space.add_evaluation_object(model)
    parameter_space.add_parameter(
        RangedParameter("v", float, lb=0.0, ub=1.0), path="value"
    )
    pipeline = EvaluationPipeline(parameter_space)
    pipeline.add_evaluator(lambda model: model.value * 2, output_name="doubled")

    metric_space = MetricSpace()
    metric_space.add_objective(Metric("doubled"))

    problem = Problem(parameter_space, metric_space, backend=pipeline)
    results = problem.evaluate({"v": 0.25})
    assert results["doubled"] == pytest.approx(0.5)


# ── Backend swap ──────────────────────────────────────────────────────────────


def test_with_evaluator_returns_new_problem_sharing_spaces(yield_and_purity_space):
    original_backend = StubBackend({"yield": [0.8, 0.9], "purity": 0.99})
    problem = Problem(
        metric_space=yield_and_purity_space, backend=original_backend, name="real"
    )
    surrogate = StubBackend({"yield": [0.7, 0.8], "purity": 0.9})

    swapped = problem.with_evaluator(surrogate)

    assert swapped is not problem
    assert swapped.backend is surrogate
    assert problem.backend is original_backend
    assert swapped.parameter_space is problem.parameter_space
    assert swapped.metric_space is problem.metric_space
    assert swapped.name == "real"


def test_with_evaluator_always_returns_plain_problem(yield_and_purity_space):
    class ProblemSubclass(Problem):
        pass

    subclassed = ProblemSubclass(metric_space=yield_and_purity_space)
    swapped = subclassed.with_evaluator(StubBackend({}))
    assert type(swapped) is Problem
