"""Peak scores retain their reference layout and use the intended scales."""

import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.comparison import Comparator, PeakHeight, PeakPosition
from CADETProcess.evaluation_pipeline import EvaluationFailure
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.processModel import ComponentSystem, FlowSheet, Outlet, Process
from CADETProcess.reference import ReferenceIO
from CADETProcess.simulationResults import SimulationResults


@pytest.fixture
def unequal_peak_components():
    """Two resolved peaks in A and one in B, with distinct shifts and heights."""
    time = np.linspace(0, 100, 1001)
    components = ComponentSystem(["A", "B"])
    reference = np.zeros((time.size, 2))
    simulation = np.zeros_like(reference)
    for component, position, height, shift, height_change in [
        (0, 20, 2, 2, 1),
        (0, 70, 3, 4, 1),
        (1, 40, 4, 6, 2),
    ]:
        reference[:, component] += height * np.maximum(
            1 - np.abs(time - position) / 2, 0
        )
        simulation[:, component] += (height + height_change) * np.maximum(
            1 - np.abs(time - position - shift) / 2, 0
        )
    return (
        ReferenceIO("reference", time, reference, component_system=components),
        ReferenceIO("simulation", time, simulation, component_system=components),
    )


@pytest.fixture(params=["missing_first", "missing_last", "extra", "same_total"])
def mismatched_peak_components(request, unequal_peak_components):
    reference, _ = unequal_peak_components
    time = reference.time
    values = reference.solution.copy()
    if request.param == "missing_first":
        values[time < 40, 0] = 0
    elif request.param == "missing_last":
        values[time > 50, 0] = 0
    elif request.param == "extra":
        values[:, 0] += 2 * np.maximum(1 - np.abs(time - 90) / 2, 0)
    else:
        # Total counts still agree: A loses a peak while B gains one.
        values[time < 40, 0] = 0
        values[:, 1] += 4 * np.maximum(1 - np.abs(time - 80) / 2, 0)
    simulation = ReferenceIO(
        "mismatched", time, values, component_system=reference.component_system
    )
    return reference, simulation, 3 if request.param == "extra" else 1


@pytest.fixture
def peak_simulation_results(unequal_peak_components):
    _, simulation = unequal_peak_components
    flow_sheet = FlowSheet(simulation.component_system)
    flow_sheet.add_unit(Outlet(simulation.component_system, "outlet"))
    process = Process(flow_sheet, "synthetic")
    process.cycle_time = 100
    return SimulationResults(
        "synthetic",
        {},
        0,
        "",
        0,
        process,
        {"outlet": {"outlet": [simulation]}},
        {},
        {},
        [],
    )


@pytest.mark.parametrize("metric_type", [PeakPosition, PeakHeight])
@pytest.mark.parametrize("normalize_metrics", [False, True])
def test_unequal_peak_counts_raise_with_component_and_counts(
    mismatched_peak_components, metric_type, normalize_metrics
):
    reference, simulation, count = mismatched_peak_components
    metric = metric_type(reference, resample=False, normalize_metrics=normalize_metrics)

    with pytest.raises(CADETProcessError) as error:
        metric.evaluate(simulation)

    assert str(error.value) == (
        f"Peak count mismatch for component 0 ('A'): reference=2, simulation={count}."
    )


@pytest.mark.parametrize("metric_type", [PeakPosition, PeakHeight])
def test_peak_count_check_includes_later_components(
    unequal_peak_components, metric_type
):
    reference, _ = unequal_peak_components
    values = reference.solution.copy()
    values[:, 1] += 4 * np.maximum(1 - np.abs(reference.time - 80) / 2, 0)
    simulation = ReferenceIO(
        "extra B peak",
        reference.time,
        values,
        component_system=reference.component_system,
    )

    with pytest.raises(CADETProcessError) as error:
        metric_type(reference, resample=False).evaluate(simulation)

    assert str(error.value) == (
        "Peak count mismatch for component 1 ('B'): reference=1, simulation=2."
    )


@pytest.mark.parametrize("metric_type", [PeakPosition, PeakHeight])
def test_windowed_peak_counts_ignore_unmatched_peaks_outside_the_window(
    unequal_peak_components, metric_type
):
    reference, _ = unequal_peak_components
    values = reference.solution.copy()
    values[reference.time < 40, 0] = 0
    simulation = ReferenceIO(
        "late peak only",
        reference.time,
        values,
        component_system=reference.component_system,
    )
    metric = metric_type(reference, components=["A"], start=50, end=90, resample=False)

    scores = metric.evaluate(simulation)

    assert scores.shape == (metric.n_metrics,) == (1,)
    np.testing.assert_allclose(scores, [0])


@pytest.mark.parametrize(
    "normalization_factor, scales",
    [(None, [20, 70, 40]), (10, [10, 10, 10]), ([10, 30], [10, 10, 30])],
)
def test_position_factors_produce_one_score_per_reference_peak(
    unequal_peak_components, normalization_factor, scales
):
    reference, simulation = unequal_peak_components
    metric = PeakPosition(
        reference, normalization_factor=normalization_factor, resample=False
    )

    scores = metric.evaluate(simulation)

    assert scores.shape == (metric.n_metrics,) == (3,)
    # Output is component-major, then chronological within each component.
    expected = np.tanh(np.array([2, 4, 6]) / (2 * np.array(scales)))
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize(
    "normalization_factor, scales", [(None, [2, 3, 4]), (10, [10, 10, 10])]
)
def test_height_factors_produce_one_score_per_reference_peak(
    unequal_peak_components, normalization_factor, scales
):
    reference, simulation = unequal_peak_components
    metric = PeakHeight(
        reference, normalization_factor=normalization_factor, resample=False
    )

    scores = metric.evaluate(simulation)

    assert scores.shape == (metric.n_metrics,) == (3,)
    expected = np.tanh(np.array([1, 1, 2]) / (2 * np.array(scales)))
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize(
    "metric_type, expected", [(PeakPosition, [2, 4, 6]), (PeakHeight, [1, 1, 2])]
)
def test_unnormalized_peak_scores_keep_absolute_physical_differences(
    unequal_peak_components, metric_type, expected
):
    reference, simulation = unequal_peak_components
    metric = metric_type(reference, normalize_metrics=False, resample=False)

    scores = metric.evaluate(simulation)

    assert scores.shape == (metric.n_metrics,) == (3,)
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize("metric_type", [PeakPosition, PeakHeight])
@pytest.mark.parametrize(
    "selection, expected_count",
    [
        ({"start": 50, "end": 90, "components": ["A"]}, 1),
        ({"use_total_concentration": True}, 3),
    ],
)
def test_peak_layout_follows_windowing_and_component_summation(
    unequal_peak_components, metric_type, selection, expected_count
):
    reference, _ = unequal_peak_components
    metric = metric_type(reference, resample=False, **selection)

    scores = metric.evaluate(reference)

    assert scores.shape == (metric.n_metrics,) == (expected_count,)
    np.testing.assert_allclose(scores, np.zeros(expected_count))


@pytest.mark.parametrize("factors", [[], [10], [10, 20, 30]])
def test_position_requires_one_factor_per_reference_component(
    unequal_peak_components, factors
):
    reference, _ = unequal_peak_components
    with pytest.raises(ValueError, match="one value per reference component"):
        PeakPosition(reference, normalization_factor=factors)


@pytest.mark.parametrize(
    "selection, factors, displacements",
    [
        ({"components": ["B"]}, [30], [6]),
        ({"components": ["A"], "start": 50, "end": 90}, [10], [4]),
        ({"use_total_concentration": True}, [10], [2, 6, 4]),
    ],
)
def test_position_factors_follow_selected_reference_components(
    unequal_peak_components, selection, factors, displacements
):
    reference, simulation = unequal_peak_components
    metric = PeakPosition(
        reference, normalization_factor=factors, resample=False, **selection
    )

    scores = metric.evaluate(simulation)

    assert scores.shape == (metric.n_metrics,) == (len(displacements),)
    expected = np.tanh(np.array(displacements) / (2 * factors[0]))
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize(
    "metric_type, factors, expected",
    [
        (PeakPosition, [10, 30], np.tanh(np.array([0.1, 0.2, 0.1]))),
        (PeakHeight, 10, np.tanh(np.array([0.05, 0.05, 0.1]))),
    ],
)
def test_comparator_retains_peak_score_values_and_declared_layout(
    unequal_peak_components, peak_simulation_results, metric_type, factors, expected
):
    reference, _ = unequal_peak_components
    metric = metric_type(reference, normalization_factor=factors, resample=False)
    comparator = Comparator()
    comparator.add_difference_metric(metric, "outlet.outlet")

    scores = comparator.evaluate(peak_simulation_results)

    assert len(scores) == comparator.n_metrics == 3
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize("metric_type", [PeakPosition, PeakHeight])
def test_peak_count_failure_propagates_through_comparator_and_optimizer(
    mismatched_peak_components, peak_simulation_results, metric_type
):
    reference, simulation, count = mismatched_peak_components
    peak_simulation_results.solution_cycles["outlet"]["outlet"] = [simulation]
    metric = metric_type(reference, resample=False)
    comparator = Comparator()
    comparator.add_difference_metric(metric, "outlet.outlet")

    with pytest.raises(CADETProcessError, match="Peak count mismatch"):
        comparator.evaluate(peak_simulation_results)

    problem = OptimizationProblem("peak_counts", use_diskcache=False)
    problem.add_evaluation_object(peak_simulation_results)
    problem.add_variable("x", targets=None, lb=0, ub=1)
    # Distinct finite fallback scores make dimension and ordering observable.
    bad_metrics = [101, 102, 103]
    problem.add_objective(
        comparator,
        name="peaks",
        n_objectives=comparator.n_metrics,
        bad_metrics=bad_metrics,
    )

    failure = problem.evaluate({"x": 0.5})["peaks"]
    assert isinstance(failure, EvaluationFailure)
    assert f"reference=2, simulation={count}" in failure.reason
    scores = problem.evaluate_objectives([0.5])
    assert np.asarray(scores).size == comparator.n_metrics == 3
    np.testing.assert_allclose(np.asarray(scores).ravel(), bad_metrics)
