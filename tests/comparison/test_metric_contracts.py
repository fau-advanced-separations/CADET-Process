"""Every difference metric honors the contract the optimization relies on.

A metric declares `n_metrics` scores and as many `bad_metrics`; it returns that many
finite scores, scores zero against its own reference, and keeps its scores unchanged
inside a `Comparator`.
"""

import types

import numpy as np
import pytest
from CADETProcess.comparison import (
    L1,
    L2,
    NRMSE,
    RMSE,
    SSE,
    AbsoluteArea,
    BreakthroughHeight,
    BreakthroughPosition,
    Comparator,
    FractionationNRMSE,
    FractionationSSE,
    PeakHeight,
    PeakPosition,
    RelativeArea,
    Shape,
)
from CADETProcess.comparison.difference import ShapeFront
from CADETProcess.processModel import ComponentSystem
from CADETProcess.reference import FractionationReference, ReferenceIO

time = np.linspace(0, 100, 1001)


def gauss(position, height, width=2.0):
    return height * np.exp(-((time - position) ** 2) / (2 * width**2))


def step(position, height, width=1.0):
    return height / (1 + np.exp(-(time - position) / width))


# Two components with unequal peak counts (A: two peaks, B: one), so a metric that
# mixes up components, peaks or their order changes the score layout.
traces = {
    "peaks": (
        np.column_stack([gauss(20, 2) + gauss(70, 3), gauss(40, 4)]),
        np.column_stack([gauss(22, 3) + gauss(74, 4), gauss(46, 6)]),
    ),
    "steps": (
        np.column_stack([step(30, 1), step(60, 2)]),
        np.column_stack([step(33, 1.2), step(62, 2.5)]),
    ),
    "single_peak": (gauss(40, 4)[:, None], gauss(46, 6)[:, None]),
}

fraction_windows = [(10, 30), (30, 50), (60, 80)]

metric_cases = [
    (SSE, "peaks"),
    (RMSE, "peaks"),
    (NRMSE, "peaks"),
    (L1, "peaks"),
    (L2, "peaks"),
    (AbsoluteArea, "peaks"),
    (RelativeArea, "peaks"),
    (PeakHeight, "peaks"),
    (PeakPosition, "peaks"),
    (BreakthroughHeight, "steps"),
    (BreakthroughPosition, "steps"),
    (FractionationSSE, "fractions"),
    (FractionationNRMSE, "fractions"),
    # Shape metrics support a single component only.
    (Shape, "single_peak"),
    (ShapeFront, "single_peak"),
]


@pytest.fixture(params=metric_cases, ids=lambda case: case[0].__name__)
def metric_setup(request):
    """Return a metric, its reference trace, and a shifted and rescaled simulation."""
    metric_type, kind = request.param
    reference_values, simulation_values = traces[
        "peaks" if kind == "fractions" else kind
    ]
    component_system = ComponentSystem(reference_values.shape[1])
    reference = ReferenceIO(
        "reference", time, reference_values, component_system=component_system
    )
    simulation = ReferenceIO(
        "simulation", time, simulation_values, component_system=component_system
    )
    if kind == "fractions":
        # Fractions collected from the reference trace, so that trace scores zero.
        fractions = [
            reference.create_fraction(start, end) for start, end in fraction_windows
        ]
        metric = metric_type(
            FractionationReference(
                "fractions", fractions, component_system=component_system
            )
        )
    else:
        metric = metric_type(reference)
    return metric, reference, simulation


def test_scores_match_declared_layout(metric_setup):
    metric, _, simulation = metric_setup

    scores = np.atleast_1d(metric.evaluate(simulation))

    assert scores.shape == (metric.n_metrics,)
    assert len(metric.bad_metrics) == metric.n_metrics
    assert np.all(np.isfinite(scores))
    assert not np.allclose(scores, 0)


def test_scores_zero_against_own_reference(metric_setup):
    metric, reference, _ = metric_setup

    scores = np.atleast_1d(metric.evaluate(reference))
    # Shape metrics locate the offset at a flat Pearson optimum, so it is only resolved
    # to about the square root of machine precision relative to the time span.
    atol = np.full(scores.shape, 1e-8)
    if isinstance(metric, Shape):
        atol[metric.labels.index("Time offset")] = 1e-6 * (time[-1] - time[0])

    assert np.all(np.abs(scores) <= atol), scores


def test_comparator_keeps_metric_scores(metric_setup):
    metric, _, simulation = metric_setup
    comparator = Comparator()
    comparator.add_difference_metric(metric, "outlet.outlet")
    simulation_results = types.SimpleNamespace(
        solution_cycles={"outlet": {"outlet": [simulation]}}
    )

    scores = comparator.evaluate(simulation_results)

    assert len(scores) == comparator.n_metrics == len(comparator.bad_metrics)
    np.testing.assert_allclose(scores, metric.evaluate(simulation))
