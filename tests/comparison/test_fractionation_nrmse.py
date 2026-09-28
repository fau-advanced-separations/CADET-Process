"""Fraction scores compare flow-weighted concentrations with equal fraction weights."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from CADETProcess.comparison import Comparator, FractionationNRMSE, FractionationSSE
from CADETProcess.fractionation import Fraction
from CADETProcess.processModel import ComponentSystem, FlowSheet, Outlet, Process
from CADETProcess.reference import FractionationReference, ReferenceIO
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import SolutionIO


@pytest.fixture
def varying_flow_solution():
    time = np.linspace(0, 4, 81)
    return SolutionIO(
        "outlet",
        ComponentSystem(["A", "B"]),
        time,
        np.column_stack([time, 2 * time]),
        flow_rate=1 + time,
    )


@pytest.fixture
def unequal_volume_reference():
    return FractionationReference(
        "collected",
        [
            Fraction(mass=[1.5, 6], volume=1.5, start=0, end=1),
            Fraction(mass=[12, 48], volume=6, start=1, end=3),
        ],
        component_system=ComponentSystem(["A", "B"]),
    )


@pytest.mark.parametrize("components, indices", [(None, [0, 1]), (["B"], [1])])
def test_fraction_nrmse_uses_flow_weighting_and_equal_fraction_weights(
    varying_flow_solution,
    unequal_volume_reference,
    components,
    indices,
):
    # Integral of t*(1+t) / integral of (1+t) on [0,1] and [1,3].
    averaged = np.array([[5 / 9, 10 / 9], [19 / 9, 38 / 9]])[:, indices]
    measured = np.array([[1, 4], [2, 8]])[:, indices]
    expected = np.sqrt(np.mean((averaged - measured) ** 2, axis=0)) / measured.max(
        axis=0
    )
    metric = FractionationNRMSE(unequal_volume_reference, components=components)
    original = varying_flow_solution.solution.copy()
    for _ in range(2):
        np.testing.assert_allclose(metric.evaluate(varying_flow_solution), expected)
    assert metric.n_metrics == len(indices)
    np.testing.assert_array_equal(varying_flow_solution.solution, original)
    np.testing.assert_array_equal(unequal_volume_reference.solution, [[1, 4], [2, 8]])


def test_single_component_reference_selects_matching_named_simulation_component(
    varying_flow_solution,
):
    reference = FractionationReference(
        "B only",
        [
            Fraction(mass=[10 / 9], volume=1, start=0, end=1),
            Fraction(mass=[38 / 9], volume=1, start=1, end=3),
        ],
        component_system=ComponentSystem(["B"]),
    )
    metric = FractionationNRMSE(reference, components=["B"])
    np.testing.assert_allclose(metric(varying_flow_solution), [0], atol=1e-12)


@pytest.mark.parametrize("bound", ["start", "end"])
def test_fraction_nrmse_requires_collection_windows_instead_of_trace_slicing(
    unequal_volume_reference,
    bound,
):
    with pytest.raises(ValueError, match="collection windows"):
        FractionationNRMSE(unequal_volume_reference, **{bound: 1.0})


def test_fraction_nrmse_requires_a_fraction_reference(varying_flow_solution):
    with pytest.raises(TypeError, match="reference type"):
        FractionationNRMSE(varying_flow_solution)


def test_zero_reference_maximum_retains_trace_nrmse_division_behavior(
    varying_flow_solution,
):
    reference = FractionationReference(
        "zero",
        [Fraction(mass=[0], volume=1, start=0, end=1)],
        component_system=ComponentSystem(["A"]),
    )
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        score = FractionationNRMSE(reference, components=["A"])(varying_flow_solution)
    assert np.isinf(score[0])


@pytest.mark.parametrize("metric_type", [FractionationSSE, FractionationNRMSE])
def test_fraction_plot_annotation_matches_objective_without_reintegrating(
    varying_flow_solution,
    unequal_volume_reference,
    metric_type,
):
    flow_sheet = FlowSheet(varying_flow_solution.component_system)
    flow_sheet.add_unit(Outlet(varying_flow_solution.component_system, "outlet"))
    process = Process(flow_sheet, "analytical")
    process.cycle_time = 4
    results = SimulationResults(
        "analytical",
        {},
        0,
        "",
        0,
        process,
        {"outlet": {"outlet": [varying_flow_solution]}},
        {},
        {},
        [],
    )
    metric = metric_type(unequal_volume_reference, components=["A"])
    comparator = Comparator()
    comparator.add_difference_metric(metric, "outlet.outlet")
    expected_sse = (4 / 9) ** 2 + (1 / 9) ** 2
    expected = (
        expected_sse
        if metric_type is FractionationSSE
        else np.sqrt(expected_sse / 2) / 2
    )
    np.testing.assert_allclose(comparator.evaluate(results), [expected])
    fig, axes = comparator.plot_comparison(results)
    try:
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [5 / 9, 19 / 9])
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [1, 2])
        annotation = float(axes[0].texts[0].get_text().split(": ")[-1])
        assert annotation == pytest.approx(float(f"{expected:.2g}"))
    finally:
        plt.close(fig)


def test_reference_io_fraction_method_already_returns_window_average():
    time = np.linspace(0, 4, 81)
    reference = ReferenceIO("ramp", time, time[:, None], flow_rate=1 + time)
    assert reference.create_fraction(1, 3).concentration[0] == pytest.approx(19 / 9)
