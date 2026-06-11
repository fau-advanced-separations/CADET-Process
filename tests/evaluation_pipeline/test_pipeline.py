from __future__ import annotations

from dataclasses import dataclass

import pytest
from CADETProcess.evaluation_pipeline import EvaluationFailure, EvaluationPipeline
from CADETProcess.parameter_space.space import ParameterSpace

# ── shared fixtures ───────────────────────────────────────────────────────────


@dataclass
class Model:
    """Minimal evaluation object: a single mutable value."""

    value: float = 0.0


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def two_models():
    return Model(value=0.0), Model(value=0.0)


def _make_space(*models: Model) -> ParameterSpace:
    space = ParameterSpace()
    for m in models:
        space.add_evaluation_object(m)
    return space


@pytest.fixture
def single_space(model):
    return _make_space(model)


@pytest.fixture
def two_space(two_models):
    return _make_space(*two_models)


# ── EvaluationFailure ─────────────────────────────────────────────────────────


def test_failure_str_contains_stage_and_reason():
    f = EvaluationFailure(stage="simulate", reason="timeout")
    assert "simulate" in str(f)
    assert "timeout" in str(f)


def test_failure_recoverable_flag_in_str():
    f = EvaluationFailure(stage="s", reason="r", recoverable=True)
    assert "recoverable" in str(f)


def test_failure_not_recoverable_by_default():
    f = EvaluationFailure(stage="s", reason="r")
    assert not f.recoverable


# ── EvaluationPipeline construction ──────────────────────────────────────────


def test_requires_parameter_space(single_space):
    EvaluationPipeline(single_space)  # no error


def test_rejects_non_parameter_space():
    with pytest.raises(TypeError):
        EvaluationPipeline("not a space")  # type: ignore[arg-type]


def test_evaluate_before_add_evaluator_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    with pytest.raises(RuntimeError, match="No evaluators"):
        pipeline.evaluate([])


def test_output_names_empty_initially(single_space):
    pipeline = EvaluationPipeline(single_space)
    assert pipeline.output_names == []


# ── add_evaluator ─────────────────────────────────────────────────────────────


def test_add_evaluator_registers_output_name(single_space):
    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="v")
    assert "v" in pipeline.output_names


def test_add_evaluator_duplicate_output_name_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="v")
    with pytest.raises(ValueError, match="already registered"):
        pipeline.add_evaluator(lambda model: model.value, output_name="v")


def test_add_evaluator_non_callable_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    with pytest.raises(TypeError):
        pipeline.add_evaluator("not a function", output_name="v")  # type: ignore[arg-type]


def test_add_evaluator_invalid_output_name_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    with pytest.raises(ValueError):
        pipeline.add_evaluator(lambda model: model.value, output_name="has space")


# ── evaluate: single evaluation object ───────────────────────────────────────


def test_evaluate_single_object_returns_dict(single_space):
    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value * 2, output_name="doubled")
    results = pipeline.evaluate([])
    assert isinstance(results, dict)
    assert "doubled" in results


def test_evaluate_passes_set_values_to_model(single_space, model):
    """set_values on the space must reach the evaluation object before the node runs."""

    # add a parameter that writes to model.value
    from CADETProcess.parameter_space.parameters import RangedParameter

    p = RangedParameter("v", float, lb=0.0, ub=1.0)
    single_space.add_parameter(p, path="value")

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="v")

    results = pipeline.evaluate([0.42])
    assert results["v"] == pytest.approx(0.42)


def test_evaluate_shared_intermediate_computed_once(single_space):
    """A node that feeds multiple targets must run only once per evaluate call."""
    call_count = 0

    def expensive(model):
        nonlocal call_count
        call_count += 1
        return model.value + 1

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(expensive, output_name="intermediate")
    pipeline.add_evaluator(
        lambda intermediate: intermediate * 2,
        output_name="doubled",
        requires=["intermediate"],
    )
    pipeline.add_evaluator(
        lambda intermediate: intermediate + 10,
        output_name="shifted",
        requires=["intermediate"],
    )

    pipeline.evaluate([], targets=["doubled", "shifted"])
    assert call_count == 1


def test_repeated_x_hits_cache(single_space):
    """Calling evaluate with the same x twice must not recompute root nodes."""
    call_count = 0

    def expensive(model):
        nonlocal call_count
        call_count += 1
        return model.value + 1

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(expensive, output_name="result")

    pipeline.evaluate([])
    pipeline.evaluate([])  # same x — should be a cache hit
    assert call_count == 1


def test_different_x_invalidates_cache(single_space):
    """Calling evaluate with different x values must recompute root nodes."""
    from CADETProcess.parameter_space.parameters import RangedParameter

    p = RangedParameter("v", float, lb=0.0, ub=1.0)
    single_space.add_parameter(p, path="value")

    call_count = 0

    def expensive(model):
        nonlocal call_count
        call_count += 1
        return model.value

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(expensive, output_name="result")

    r1 = pipeline.evaluate([0.1])
    r2 = pipeline.evaluate([0.9])
    assert call_count == 2
    assert r1["result"] != r2["result"]


def test_evaluate_partial_targets_skips_unneeded_nodes(single_space):
    """Requesting a single target must not execute nodes outside its subgraph."""
    side_branch_called = False

    def side_branch(model):
        nonlocal side_branch_called
        side_branch_called = True
        return 99.0

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="a")
    pipeline.add_evaluator(side_branch, output_name="b")

    pipeline.evaluate([], targets=["a"])
    assert not side_branch_called


def test_evaluate_unknown_target_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="a")
    with pytest.raises(ValueError, match="Unknown target"):
        pipeline.evaluate([], targets=["nonexistent"])


# ── evaluate: multiple evaluation objects ────────────────────────────────────


def test_evaluate_multiple_objects_returns_lists(two_space, two_models):
    pipeline = EvaluationPipeline(two_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="v")
    results = pipeline.evaluate([])
    assert isinstance(results["v"], list)
    assert len(results["v"]) == 2


def test_evaluate_multiple_objects_independent_results(two_space, two_models):
    m1, m2 = two_models
    m1.value = 1.0
    m2.value = 2.0

    pipeline = EvaluationPipeline(two_space)
    pipeline.add_evaluator(lambda model: model.value, output_name="v")

    results = pipeline.evaluate([])
    assert results["v"] == [1.0, 2.0]


# ── EvaluationFailure propagation ─────────────────────────────────────────────


def test_failing_node_returns_evaluation_failure(single_space):
    def bad_node(model):
        raise ValueError("boom")

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(bad_node, output_name="result")

    results = pipeline.evaluate([])
    assert isinstance(results["result"], EvaluationFailure)
    assert results["result"].stage == "result"
    assert "boom" in results["result"].reason


def test_failure_propagates_to_downstream_node(single_space):
    def bad_node(model):
        raise RuntimeError("upstream failure")

    downstream_called = False

    def downstream(result):
        nonlocal downstream_called
        downstream_called = True
        return result

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(bad_node, output_name="result")
    pipeline.add_evaluator(downstream, output_name="final", requires=["result"])

    results = pipeline.evaluate([])
    assert isinstance(results["final"], EvaluationFailure)
    assert results["final"].stage == "result"
    assert not downstream_called


def test_failure_preserves_original_stage_through_chain(single_space):
    """Stage name must reflect the node that originally failed, not the propagation nodes."""

    def bad(model):
        raise RuntimeError("root cause")

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(bad, output_name="step1")
    pipeline.add_evaluator(lambda step1: step1, output_name="step2", requires=["step1"])
    pipeline.add_evaluator(lambda step2: step2, output_name="step3", requires=["step2"])

    results = pipeline.evaluate([])
    assert isinstance(results["step3"], EvaluationFailure)
    assert results["step3"].stage == "step1"


# ── requires injection ────────────────────────────────────────────────────────


def test_requires_injects_upstream_output_by_position(single_space):
    """Function argument names need not match upstream output names when requires is given."""

    def producer(model):
        return model.value + 5

    def consumer(x):  # arg named 'x', not 'upstream_value'
        return x * 3

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(producer, output_name="upstream_value")
    pipeline.add_evaluator(consumer, output_name="result", requires=["upstream_value"])

    results = pipeline.evaluate([])
    assert results["result"] == pytest.approx((0.0 + 5) * 3)


def test_requires_multi_input_injects_in_order(single_space):
    """Multiple requires entries are injected positionally."""

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(lambda model: 2.0, output_name="a")
    pipeline.add_evaluator(lambda model: 3.0, output_name="b")
    pipeline.add_evaluator(
        lambda x, y: x - y,
        output_name="diff",
        requires=["a", "b"],
    )

    results = pipeline.evaluate([])
    assert results["diff"] == pytest.approx(2.0 - 3.0)


def test_requires_with_invalid_identifier_raises(single_space):
    pipeline = EvaluationPipeline(single_space)
    with pytest.raises(ValueError):
        pipeline.add_evaluator(
            lambda x: x,
            output_name="out",
            requires=["has-hyphen"],
        )


# ── EvaluationFailure.exc ─────────────────────────────────────────────────────


def test_failure_stores_original_exception(single_space):
    exc = ValueError("boom")

    def bad_node(model):
        raise exc

    pipeline = EvaluationPipeline(single_space)
    pipeline.add_evaluator(bad_node, output_name="result")

    results = pipeline.evaluate([])
    assert results["result"].exc is exc


def test_failure_exc_is_none_when_constructed_directly():
    f = EvaluationFailure(stage="s", reason="r")
    assert f.exc is None


# ── disk cache ────────────────────────────────────────────────────────────────


def test_disk_cache_hit_evaluates_only_once(tmp_path, single_space):
    """Same x on the same pipeline instance must hit the disk cache on the second call."""
    call_count = 0

    def counting_evaluator(model):
        nonlocal call_count
        call_count += 1
        return model.value + 1.0

    pipeline = EvaluationPipeline(single_space, cache_dir=tmp_path / "cache")
    pipeline.add_evaluator(counting_evaluator, output_name="result")

    pipeline.evaluate([])
    pipeline.evaluate([])  # same x → cache hit

    assert call_count == 1


def test_disk_cache_miss_on_different_x(tmp_path):
    """Different x values must produce distinct cache entries."""
    from CADETProcess.parameter_space.parameters import RangedParameter

    call_count = 0
    model = Model()
    space = _make_space(model)
    param = RangedParameter("v", float, lb=0.0, ub=10.0)
    space.add_parameter(param, path="value", evaluation_objects=[model])

    def counting_evaluator(m):
        nonlocal call_count
        call_count += 1
        return m.value

    pipeline = EvaluationPipeline(space, cache_dir=tmp_path / "cache")
    pipeline.add_evaluator(counting_evaluator, output_name="result")

    pipeline.evaluate([1.0])
    pipeline.evaluate([2.0])

    assert call_count == 2


def test_lru_cache_is_used_when_no_cache_dir(single_space):
    """Without cache_dir the pipeline falls back to the LRU backend."""
    call_count = 0

    def counting_evaluator(model):
        nonlocal call_count
        call_count += 1
        return model.value

    pipeline = EvaluationPipeline(single_space)  # no cache_dir
    pipeline.add_evaluator(counting_evaluator, output_name="result")

    pipeline.evaluate([])
    pipeline.evaluate([])

    assert call_count == 1
