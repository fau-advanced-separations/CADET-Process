import unittest

import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.reference import ReferenceIO
from CADETProcess.solution import SolutionIO
from scipy import stats

comp_2 = ComponentSystem(["A", "B"])

time = np.linspace(0, 100, 1001)

mu_0 = 30
sigma_0 = 5

mu_1 = 40
sigma_1 = 5

mu_2 = 50
sigma_2 = 3


solution_2_gaussian = np.zeros((len(time), 2))
solution_2_gaussian[:, 0] = stats.norm.pdf(time, mu_0, sigma_0)
solution_2_gaussian[:, 1] = stats.norm.pdf(time, mu_1, sigma_1)

solution_2_gaussian_switched = np.zeros((len(time), 2))
solution_2_gaussian_switched[:, 1] = stats.norm.pdf(time, mu_0, sigma_0)
solution_2_gaussian_switched[:, 0] = stats.norm.pdf(time, mu_1, sigma_1)

solution_2_gaussian_different_height = np.zeros((len(time), 2))
solution_2_gaussian_different_height[:, 1] = stats.norm.pdf(time, mu_0, sigma_0)
solution_2_gaussian_different_height[:, 0] = stats.norm.pdf(time, mu_2, sigma_2)

q_const = np.ones(time.shape)


from CADETProcess.comparison import SSE


class TestSSE(unittest.TestCase):
    def test_metric(self):
        # Compare with itself
        component_system = ComponentSystem(1)
        reference = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        difference = SSE(reference)
        metrics_expected = [0]
        metrics = difference.evaluate(reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gaussian Peak
        component_system = ComponentSystem(1)
        solution = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [1]],
            component_system=component_system,
        )

        difference = SSE(reference, resample=False)
        metrics_expected = [0.7132717]
        metrics = difference.evaluate(solution)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        difference = SSE(reference, resample=True)
        metrics_expected = [3.56635829]
        metrics = difference.evaluate(solution)
        np.testing.assert_almost_equal(metrics, metrics_expected)


from CADETProcess.comparison import NRMSE


class TestNRMSE(unittest.TestCase):
    def test_metric(self):
        # Compare with itself
        component_system = ComponentSystem(1)
        reference = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        difference = NRMSE(reference)
        metrics_expected = [0]
        metrics = difference.evaluate(reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gaussian Peak
        component_system = ComponentSystem(1)
        solution = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [1]],
            component_system=component_system,
        )

        difference = NRMSE(reference, resample=False)
        metrics_expected = [0.3345572]
        metrics = difference.evaluate(solution)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        difference = NRMSE(reference, resample=True)
        metrics_expected = [0.33469097]
        metrics = difference.evaluate(solution)
        np.testing.assert_almost_equal(metrics, metrics_expected)


from CADETProcess.comparison import L1, L2


@pytest.mark.parametrize("metric_type, expected", [(L1, [5, 2]), (L2, [np.sqrt(5), 2])])
def test_norm_scores_each_component(metric_type, expected):
    """Component A differs by 1 at every point, B by 2 at a single point."""
    norm_time = np.linspace(0, 4, 5)
    reference = ReferenceIO(
        "reference", norm_time, np.zeros((norm_time.size, 2)), component_system=comp_2
    )
    values = np.zeros((norm_time.size, 2))
    values[:, 0] = 1
    values[1, 1] = 2
    simulation = ReferenceIO("simulation", norm_time, values, component_system=comp_2)
    metric = metric_type(reference, resample=False)

    scores = metric.evaluate(simulation)

    assert metric.n_metrics == 2
    np.testing.assert_allclose(scores, expected)


from CADETProcess.comparison import RMSE, AbsoluteArea, RelativeArea


@pytest.mark.parametrize(
    "metric_type, expected",
    [
        (SSE, [5, 4]),
        (RMSE, [1, np.sqrt(4 / 5)]),
        # Divided by the reference maximum, 1 for A and 2 for B.
        (NRMSE, [1, np.sqrt(4 / 5) / 2]),
        # Simpson weights on five unit-spaced points are [1, 4, 2, 4, 1] / 3, so
        # the reference areas are 4 and 8 and B gains 2 * 4/3 from its one point.
        (AbsoluteArea, [4, 8 / 3]),
        (RelativeArea, [1, 1 / 3]),
    ],
)
def test_difference_scores_each_component(metric_type, expected):
    """Component A differs by 1 at every point, B by 2 at a single point."""
    norm_time = np.linspace(0, 4, 5)
    reference_values = np.ones((norm_time.size, 2))
    reference_values[:, 1] = 2
    reference = ReferenceIO(
        "reference", norm_time, reference_values, component_system=comp_2
    )
    values = reference_values.copy()
    values[:, 0] += 1
    values[1, 1] += 2
    simulation = ReferenceIO("simulation", norm_time, values, component_system=comp_2)
    metric = metric_type(reference, resample=False)

    np.testing.assert_allclose(metric.evaluate(simulation), expected)


def test_smoothed_sse_scores_zero_against_itself():
    reference = ReferenceIO(
        "reference", time, solution_2_gaussian, component_system=comp_2
    )
    metric = SSE(reference, smooth=True)

    np.testing.assert_allclose(metric.evaluate(reference), [0, 0], atol=1e-12)


class TestNormalizeFlatTrace(unittest.TestCase):
    def test_flat_solution_normalizes_to_zero(self):
        """A flat solution (e.g. protein never elutes) must not produce NaN.

        Min-max normalization divided by a zero span; the flat trace now
        normalizes to zero, so the metric compares the normalized reference
        against zero.
        """
        component_system = ComponentSystem(1)
        reference = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )
        flat = ReferenceIO(
            "flat",
            time,
            np.zeros((len(time), 1)),
            component_system=component_system,
        )

        metrics = SSE(reference, normalize=True).evaluate(flat)
        metrics_expected = SSE(reference.normalize()).evaluate(flat)

        assert np.all(np.isfinite(metrics))
        np.testing.assert_almost_equal(metrics, metrics_expected)


from CADETProcess.comparison import PeakHeight


class TestPeakHeight(unittest.TestCase):
    def setUp(self):
        # 2 Components, gaussian peaks, constant flow
        component_system = ComponentSystem(1)
        self.reference_single = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        self.reference = ReferenceIO(
            "simple", time, solution_2_gaussian, component_system=comp_2
        )

        self.reference_switched = ReferenceIO(
            "simple", time, solution_2_gaussian_switched, component_system=comp_2
        )

        self.reference_different_height = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian_different_height,
            component_system=comp_2,
        )

    def test_metric(self):
        # Compare with itself
        difference = PeakHeight(self.reference, components=["A"])
        metrics_expected = [0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak
        difference = PeakHeight(
            self.reference_switched, components=["A"], normalize_metrics=False
        )
        metrics_expected = [0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, normalize_metrics
        difference = PeakHeight(
            self.reference_switched, components=["A"], normalize_metrics=True
        )
        metrics_expected = [0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Multi-component, compare with self.
        difference = PeakHeight(self.reference)
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Swich components
        difference = PeakHeight(self.reference, normalize_metrics=False)
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference_switched)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Different Peaks
        difference = PeakHeight(self.reference, normalize_metrics=False)
        metrics_expected = [0.0531923, 0.0]
        metrics = difference.evaluate(self.reference_different_height)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        difference = PeakHeight(self.reference, normalize_metrics=True)
        metrics_expected = [0.3215127, 0.0]
        metrics = difference.evaluate(self.reference_different_height)
        np.testing.assert_almost_equal(metrics, metrics_expected)


from CADETProcess.comparison import PeakPosition


class TestPeakPosition(unittest.TestCase):
    def setUp(self):
        # 2 Components, gaussian peaks, constant flow
        component_system = ComponentSystem(1)
        self.reference_single = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        self.reference = ReferenceIO(
            "simple", time, solution_2_gaussian, component_system=comp_2
        )

        self.reference_switched = ReferenceIO(
            "simple", time, solution_2_gaussian_switched, component_system=comp_2
        )

    def test_metric(self):
        # Compare with itself
        difference = PeakPosition(self.reference, components=["A"])
        metrics_expected = [0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak
        difference = PeakPosition(
            self.reference_switched, components=["A"], normalize_metrics=False
        )
        metrics_expected = [10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, normalize_metrics
        difference = PeakPosition(
            self.reference_switched, components=["B"], normalize_metrics=True
        )
        metrics_expected = [0.1651404]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Multi-component, compare with self.
        difference = PeakPosition(self.reference)
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Swich components
        difference = PeakPosition(self.reference, normalize_metrics=False)
        metrics_expected = [10, 10]
        metrics = difference.evaluate(self.reference_switched)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        difference = PeakPosition(self.reference)
        metrics_expected = [0.1651404, 0.124353]
        metrics = difference.evaluate(self.reference_switched)
        np.testing.assert_almost_equal(metrics, metrics_expected)


from CADETProcess.comparison import Shape


class TestShape(unittest.TestCase):
    def setUp(self):
        # 2 Components, gaussian peaks, constant flow
        component_system = ComponentSystem(1)
        self.reference_single = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        self.reference = ReferenceIO(
            "simple", time, solution_2_gaussian, component_system=comp_2
        )

        self.reference_switched = ReferenceIO(
            "simple", time, solution_2_gaussian_switched, component_system=comp_2
        )

    def test_metric(self):
        # Compare with itself
        difference = Shape(
            self.reference,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak
        difference = Shape(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, normalize_metrics
        difference = Shape(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, include derivative
        difference = Shape(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, include derivative, normalize metrics
        difference = Shape(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Multi-component, currently not implemented
        with self.assertRaises(CADETProcessError):
            difference = Shape(self.reference, use_derivative=False)
            metrics_expected = [0, 0]
        with self.assertRaises(CADETProcessError):
            difference = Shape(self.reference_single)
            metrics = difference.evaluate(self.reference)


from CADETProcess.comparison import ShapeFront


class TestShapeFront(unittest.TestCase):
    def setUp(self):
        # 2 Components, gaussian peaks, constant flow
        component_system = ComponentSystem(1)
        self.reference_single = ReferenceIO(
            "simple",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        )

        self.reference = ReferenceIO(
            "simple", time, solution_2_gaussian, component_system=comp_2
        )

        self.reference_switched = ReferenceIO(
            "simple", time, solution_2_gaussian_switched, component_system=comp_2
        )

    def test_metric(self):
        # Compare with itself
        difference = ShapeFront(
            self.reference,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, normalize_metrics
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, include derivative
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Compare with other Gauss Peak, include derivative, normalize metrics
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0.0, 4.6211716e-01, 0.0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Use maximum slope
        difference = ShapeFront(
            self.reference,
            use_derivative=False,
            use_max_slope=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0.0, 0.0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            use_max_slope=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0.0, 10.0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected, decimal=5)

        # Multi-component, currently not implemented
        with self.assertRaises(CADETProcessError):
            difference = ShapeFront(self.reference, use_derivative=False)
            metrics_expected = [0.0, 0.0]
        with self.assertRaises(CADETProcessError):
            difference = ShapeFront(self.reference_single)
            metrics = difference.evaluate(self.reference)


from CADETProcess.comparison import BreakthroughHeight, BreakthroughPosition
from scipy.special import expit


@pytest.fixture
def breakthrough_steps():
    """A breaks through at 30 (height 1), B at 60 (height 2).

    The simulation shifts A by 5 and raises it to 1.5, and shifts B by 10.
    """
    reference = np.zeros((len(time), 2))
    reference[time >= 30, 0] = 1
    reference[time >= 60, 1] = 2
    simulation = np.zeros((len(time), 2))
    simulation[time >= 35, 0] = 1.5
    simulation[time >= 70, 1] = 2
    return (
        ReferenceIO("reference", time, reference, component_system=comp_2),
        ReferenceIO("simulation", time, simulation, component_system=comp_2),
    )


def sigmoid(difference, factor=1):
    return np.abs(2 * expit(difference / factor) - 1)


@pytest.mark.parametrize("metric_type", [BreakthroughHeight, BreakthroughPosition])
@pytest.mark.parametrize("normalize_metrics", [False, True])
def test_breakthrough_scores_zero_against_itself(
    metric_type, normalize_metrics, breakthrough_steps
):
    reference, _ = breakthrough_steps
    metric = metric_type(reference, normalize_metrics=normalize_metrics, resample=False)

    assert metric.n_metrics == 2
    np.testing.assert_allclose(metric.evaluate(reference), [0, 0])


@pytest.mark.parametrize(
    "metric_type, normalize_metrics, expected",
    [
        (BreakthroughHeight, False, [0.5, 0]),
        (BreakthroughHeight, True, [sigmoid(0.5), 0]),
        (BreakthroughPosition, False, [5, 10]),
        # Default factor: distance from the reference breakthrough to the farther
        # end of the time range, per component (70 for A, 60 for B).
        (BreakthroughPosition, True, [sigmoid(5, 70), sigmoid(10, 60)]),
    ],
)
def test_breakthrough_scores_each_component(
    metric_type, normalize_metrics, expected, breakthrough_steps
):
    reference, simulation = breakthrough_steps
    metric = metric_type(reference, normalize_metrics=normalize_metrics, resample=False)

    np.testing.assert_allclose(metric.evaluate(simulation), expected)


def test_breakthrough_position_applies_scalar_factor_to_every_component(
    breakthrough_steps,
):
    reference, simulation = breakthrough_steps
    metric = BreakthroughPosition(reference, normalization_factor=10, resample=False)

    np.testing.assert_allclose(
        metric.evaluate(simulation), [sigmoid(5, 10), sigmoid(10, 10)]
    )


@pytest.fixture
def translated_peak():
    """The simulation is the reference peak, shifted by 10 and tripled in height."""
    component_system = ComponentSystem(1)
    return (
        ReferenceIO(
            "reference",
            time,
            solution_2_gaussian[:, [0]],
            component_system=component_system,
        ),
        ReferenceIO(
            "simulation",
            time,
            3 * stats.norm.pdf(time, mu_0 + 10, sigma_0)[:, None],
            component_system=component_system,
        ),
    )


@pytest.mark.parametrize("metric_type", [Shape, ShapeFront])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"start": 10, "end": 90},
        {"resample": False},
        {"normalize": True},
        {"smooth": True},
        {"transform": lambda values: 2 * values},
    ],
    ids=["default", "start_end", "no_resample", "normalize", "smooth", "transform"],
)
def test_shape_reports_pure_translation_as_offset(metric_type, kwargs, translated_peak):
    """Shape is unchanged by a translation, so only the offset scores."""
    reference, simulation = translated_peak
    metric = metric_type(reference, normalize_metrics=False, **kwargs)

    # The offset is resolved to about one step of the time grid, 0.1.
    np.testing.assert_allclose(metric.evaluate(simulation), [0, 10, 0], atol=0.1)


@pytest.mark.parametrize("metric_type", [Shape, ShapeFront])
@pytest.mark.parametrize(
    "normalization_factor, expected_offset",
    # The default factor is a tenth of the reference's end time.
    [(None, sigmoid(10, 10)), (20, sigmoid(10, 20))],
)
def test_shape_scales_offset_by_normalization_factor(
    metric_type, normalization_factor, expected_offset, translated_peak
):
    reference, simulation = translated_peak
    metric = metric_type(reference, normalization_factor=normalization_factor)

    np.testing.assert_allclose(
        metric.evaluate(simulation), [0, expected_offset, 0], atol=1e-2
    )


from CADETProcess.comparison import FractionationSSE
from CADETProcess.fractionation import Fraction
from CADETProcess.reference import FractionationReference


class TestFractionation(unittest.TestCase):
    def setUp(self):
        fraction_1 = Fraction(
            start=15,
            end=30,
            mass=[0.49865015, 0.02274985],
            volume=15,
        )
        fraction_2 = Fraction(
            start=30,
            end=45,
            mass=[0.49865015, 0.81859462],
            volume=15,
        )
        self.fractions = [fraction_1, fraction_2]

        component_system = ComponentSystem(["A", "B"])
        self.reference = FractionationReference(
            "fractions", [fraction_1, fraction_2], component_system=component_system
        )

        self.solution = SolutionIO(
            "simple", comp_2, time, solution_2_gaussian, flow_rate=q_const
        )

    def test_metric(self):
        # Compare with itself
        difference = FractionationSSE(
            self.reference,
            components=["A"],
        )
        metrics_expected = [1.30315857e-19]
        metrics = difference.evaluate(self.solution)
        np.testing.assert_almost_equal(metrics, metrics_expected)


if __name__ == "__main__":
    unittest.main()
