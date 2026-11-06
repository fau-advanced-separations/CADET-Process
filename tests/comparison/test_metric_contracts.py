"""
Requirements for every difference metric in `CADETProcess.comparison`.

1. It returns `n_metrics` finite scores and declares as many `bad_metrics`.
2. It scores zero against its own reference.
3. Its scores are unchanged inside a `Comparator`.
4. Its score matches a hand-computed value on more than one component.
5. Every keyword it accepts changes the score as documented, or is rejected with a
   clear error.
6. A flat simulated trace (nothing elutes) gives a finite score or a clear error.

This file checks 1 to 3 and 6 for every metric at its defaults, `Shape` and
`ShapeFront` on the single component they support, and the keywords of the base class;
the other files in this package check 4 and the metric-specific keywords.
Keywords of the base class share one code path for every metric that only overrides
`_evaluate`, so `SSE` stands for those metrics.

Not yet covered
---------------
5. `use_total_concentration` for every metric; `smooth` for the breakthrough metrics.
   The fractionation metrics reject `start`/`end`, `normalize` and `smooth` with errors
   that do not name the keyword.
"""

import types

import numpy as np
import pytest
from CADETProcess import CADETProcessError
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
from CADETProcess.solution import slice_solution

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


def test_flat_simulation_scores_finite_or_raises(metric_setup):
    """A simulation in which nothing elutes must not produce NaN scores."""
    metric, reference, _ = metric_setup
    flat = ReferenceIO(
        "flat",
        time,
        np.zeros_like(reference.solution),
        component_system=reference.component_system,
    )

    try:
        scores = np.atleast_1d(metric.evaluate(flat))
    except CADETProcessError:
        return
    assert scores.shape == (metric.n_metrics,)
    assert np.all(np.isfinite(scores))


def _slice(solution, start, end):
    return slice_solution(solution, coordinates={"time": (start, end)})


# Each keyword is checked against applying the same operation to both traces before
# evaluating without it. Resampling is disabled because it changes the time grid.
keyword_cases = {
    "components": (
        {"components": ["A"]},
        lambda reference, simulation: (
            slice_solution(reference, components=["A"]),
            slice_solution(simulation, components=["A"]),
        ),
    ),
    "start_end": (
        {"start": 10, "end": 90},
        lambda reference, simulation: (
            _slice(reference, 10, 90),
            _slice(simulation, 10, 90),
        ),
    ),
    "normalize": (
        {"normalize": True},
        lambda reference, simulation: (reference.normalize(), simulation.normalize()),
    ),
    "transform": (
        {"transform": lambda values: 2 * values},
        lambda reference, simulation: (
            reference,
            ReferenceIO(
                "doubled",
                simulation.time,
                2 * simulation.solution,
                component_system=simulation.component_system,
            ),
        ),
    ),
}


# Keywords handled by `DifferenceBase` share one code path for every metric that only
# overrides `_evaluate`, so `SSE` stands for those; the breakthrough metrics derive
# their reference in the constructor and are checked separately.
@pytest.mark.parametrize("metric_type", [SSE, BreakthroughHeight, BreakthroughPosition])
@pytest.mark.parametrize("keyword", keyword_cases)
def test_keyword_equals_preprocessing_both_traces(metric_type, keyword):
    kwargs, preprocess = keyword_cases[keyword]
    kind = "peaks" if metric_type is SSE else "steps"
    component_system = ComponentSystem(["A", "B"])
    reference, simulation = (
        ReferenceIO(name, time, values, component_system=component_system)
        for name, values in zip(["reference", "simulation"], traces[kind])
    )
    expected_reference, expected_simulation = preprocess(reference, simulation)

    scores = metric_type(reference, resample=False, **kwargs).evaluate(simulation)

    np.testing.assert_allclose(
        scores,
        metric_type(expected_reference, resample=False).evaluate(expected_simulation),
    )
