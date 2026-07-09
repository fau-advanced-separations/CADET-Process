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
    """Backend returning a fixed result dict; ignores targets."""

    def __init__(self, results):
        self.results = results

    def evaluate(self, assignment, targets=None):
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


# ── targets ───────────────────────────────────────────────────────────────────


def test_evaluate_targets_returns_requested_subset(yield_and_purity_space):
    backend = StubBackend({"yield": [0.8, 0.9], "purity": 0.99})
    problem = Problem(metric_space=yield_and_purity_space, backend=backend)
    results = problem.evaluate({}, targets=["purity"])
    assert list(results) == ["purity"]


def test_evaluate_unknown_target_raises(yield_and_purity_space):
    problem = Problem(
        metric_space=yield_and_purity_space, backend=StubBackend({})
    )
    with pytest.raises(ValueError, match="Unknown metric target"):
        problem.evaluate({}, targets=["nope"])


def test_evaluate_passes_metric_names_as_backend_targets(yield_and_purity_space):
    """Side-effect nodes not declared as metrics must never execute."""
    seen_targets = []

    class RecordingBackend(StubBackend):
        def evaluate(self, assignment, targets=None):
            seen_targets.append(targets)
            return dict(self.results)

    backend = RecordingBackend({"yield": [0.8, 0.9], "purity": 0.99})
    problem = Problem(metric_space=yield_and_purity_space, backend=backend)
    problem.evaluate({})
    assert seen_targets == [["yield", "purity"]]


# ── per-object reduction (EvaluationPipeline multi-object convention) ────────


class NamedModel:
    """Evaluation object with a stable name; dataclass repr would embed values."""

    def __init__(self, name, value=0.0):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name


@pytest.fixture
def two_object_setup():
    m1, m2 = NamedModel("m1", 1.0), NamedModel("m2", 2.0)
    parameter_space = ParameterSpace()
    parameter_space.add_evaluation_object(m1)
    parameter_space.add_evaluation_object(m2)
    pipeline = EvaluationPipeline(parameter_space)
    return m1, m2, parameter_space, pipeline


def test_evaluate_reduces_per_object_results_object_major(two_object_setup):
    m1, m2, parameter_space, pipeline = two_object_setup
    pipeline.add_evaluator(lambda m: [m.value, m.value * 10], output_name="v")

    metric_space = MetricSpace()
    metric_space.add_objective(
        Metric(
            "v",
            dims=("evaluation_object", "entry"),
            coords={"evaluation_object": ["m1", "m2"], "entry": ["a", "b"]},
        )
    )
    problem = Problem(parameter_space, metric_space, backend=pipeline)

    results = problem.evaluate({})

    assert results["v"].shape == (2, 2)
    np.testing.assert_allclose(results["v"], [[1.0, 10.0], [2.0, 20.0]])


def test_evaluate_selects_declared_object_subset(two_object_setup):
    m1, m2, parameter_space, pipeline = two_object_setup
    pipeline.add_evaluator(lambda m: m.value, output_name="v")

    metric_space = MetricSpace()
    metric_space.add_objective(
        Metric(
            "v",
            dims=("evaluation_object",),
            coords={"evaluation_object": ["m2"]},
        )
    )
    problem = Problem(parameter_space, metric_space, backend=pipeline)

    results = problem.evaluate({})

    np.testing.assert_allclose(results["v"], [2.0])


def test_evaluate_per_object_failure_passes_through_as_list(two_object_setup):
    m1, m2, parameter_space, pipeline = two_object_setup

    def failing_for_m1(m):
        if m.name == "m1":
            raise ValueError("m1 diverged")
        return m.value

    pipeline.add_evaluator(failing_for_m1, output_name="v")

    metric_space = MetricSpace()
    metric_space.add_objective(
        Metric(
            "v",
            dims=("evaluation_object",),
            coords={"evaluation_object": ["m1", "m2"]},
        )
    )
    problem = Problem(parameter_space, metric_space, backend=pipeline)

    results = problem.evaluate({})

    assert isinstance(results["v"], list)
    assert isinstance(results["v"][0], EvaluationFailure)
    assert results["v"][1] == pytest.approx(2.0)


def test_evaluate_per_object_failures_without_object_dim_raise(two_object_setup):
    """A metric that does not declare the evaluation_object dimension cannot
    absorb per-object failure lists; see the sentinel note in PROJECT.md."""
    m1, m2, parameter_space, pipeline = two_object_setup

    def always_failing(m):
        raise ValueError("diverged")

    pipeline.add_evaluator(always_failing, output_name="v")

    metric_space = MetricSpace()
    metric_space.add_objective(Metric("v", n_metrics=2))
    problem = Problem(parameter_space, metric_space, backend=pipeline)

    with pytest.raises(ValueError, match="evaluation_object"):
        problem.evaluate({})


# ── zero evaluation objects ──────────────────────────────────────────────────


def test_evaluate_objectless_problem():
    """Free-variable problems evaluate through the pipeline: the assignment
    itself is the root."""
    from CADETProcess.parameter_space import RangedParameter

    parameter_space = ParameterSpace()
    parameter_space.add_parameter(RangedParameter("v", float, lb=0.0, ub=1.0))
    pipeline = EvaluationPipeline(parameter_space)
    pipeline.add_evaluator(lambda assignment: assignment["v"] ** 2, output_name="squared")

    metric_space = MetricSpace()
    metric_space.add_objective(Metric("squared"))

    problem = Problem(parameter_space, metric_space, backend=pipeline)
    results = problem.evaluate({"v": 0.5})
    assert results["squared"] == pytest.approx(0.25)


# ── evaluate_batch ────────────────────────────────────────────────────────────


class RecordingParallelBackend:
    """Parallelization backend fake recording the batches it dispatches."""

    def __init__(self):
        self.batches = []

    def evaluate(self, function, population):
        self.batches.append(list(population))
        return [function(individual) for individual in population]


@pytest.fixture
def squared_metric_problem():
    parameter_space = ParameterSpace()
    parameter_space.add_parameter(RangedParameter("v", float, lb=0.0, ub=1.0))
    pipeline = EvaluationPipeline(parameter_space)
    pipeline.add_evaluator(
        lambda assignment: assignment["v"] ** 2, output_name="squared"
    )
    metric_space = MetricSpace()
    metric_space.add_objective(Metric("squared"))
    return Problem(parameter_space, metric_space, backend=pipeline)


def test_evaluate_batch_returns_one_result_per_assignment_in_order(
    squared_metric_problem,
):
    results = squared_metric_problem.evaluate_batch([{"v": 0.2}, {"v": 0.5}])
    assert len(results) == 2
    assert results[0]["squared"] == pytest.approx(0.04)
    assert results[1]["squared"] == pytest.approx(0.25)


def test_evaluate_batch_failing_row_does_not_abort_batch(yield_and_purity_space):
    class ExplodingBackend:
        def evaluate(self, assignment, targets=None):
            if assignment.get("explode"):
                raise RuntimeError("boom")
            return {"yield": [0.8, 0.9], "purity": 0.99}

    problem = Problem(
        metric_space=yield_and_purity_space, backend=ExplodingBackend()
    )
    results = problem.evaluate_batch([{}, {"explode": True}, {}])
    np.testing.assert_array_equal(results[0]["yield"], [0.8, 0.9])
    np.testing.assert_array_equal(results[2]["yield"], [0.8, 0.9])
    assert list(results[1]) == ["yield", "purity"]
    for failure in results[1].values():
        assert isinstance(failure, EvaluationFailure)
        assert "boom" in failure.reason


def test_evaluate_batch_failing_row_reports_requested_targets_only(
    yield_and_purity_space,
):
    class ExplodingBackend:
        def evaluate(self, assignment, targets=None):
            raise RuntimeError("boom")

    problem = Problem(
        metric_space=yield_and_purity_space, backend=ExplodingBackend()
    )
    results = problem.evaluate_batch([{}], targets=["purity"])
    assert list(results[0]) == ["purity"]


def test_evaluate_batch_dispatches_rows_via_parallelization_backend(
    squared_metric_problem,
):
    backend = RecordingParallelBackend()
    assignments = [{"v": 0.2}, {"v": 0.5}]
    results = squared_metric_problem.evaluate_batch(
        assignments, parallelization_backend=backend
    )
    assert backend.batches == [assignments]
    assert results[1]["squared"] == pytest.approx(0.25)


def test_evaluate_batch_unknown_target_raises_before_dispatch(
    yield_and_purity_space,
):
    problem = Problem(
        metric_space=yield_and_purity_space, backend=StubBackend({})
    )
    with pytest.raises(ValueError, match="Unknown metric target"):
        problem.evaluate_batch([{}], targets=["nope"])


def test_evaluate_batch_without_backend_raises(yield_and_purity_space):
    problem = Problem(metric_space=yield_and_purity_space)
    with pytest.raises(RuntimeError, match="backend"):
        problem.evaluate_batch([{}])
