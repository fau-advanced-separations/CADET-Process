import unittest

import numpy as np
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
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak
        difference = Shape(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [5.5511151e-16, 10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, normalize_metrics
        difference = Shape(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, include derivative
        difference = Shape(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, include derivative, normalize metrics
        difference = Shape(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

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
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, normalize_metrics
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, include derivative
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Compare with other Gauss Peak, include derivative, normalize metrics
        difference = ShapeFront(
            self.reference_switched,
            use_derivative=True,
            components=["A"],
            normalize_metrics=True,
        )
        metrics_expected = [0, 4.6211716e-01, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Use maximum slope
        difference = ShapeFront(
            self.reference,
            use_derivative=False,
            use_max_slope=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 0]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        difference = ShapeFront(
            self.reference_switched,
            use_derivative=False,
            use_max_slope=True,
            components=["A"],
            normalize_metrics=False,
        )
        metrics_expected = [0, 10]
        metrics = difference.evaluate(self.reference)
        np.testing.assert_almost_equal(metrics, metrics_expected)

        # Multi-component, currently not implemented
        with self.assertRaises(CADETProcessError):
            difference = ShapeFront(self.reference, use_derivative=False)
            metrics_expected = [0, 0]
        with self.assertRaises(CADETProcessError):
            difference = ShapeFront(self.reference_single)
            metrics = difference.evaluate(self.reference)


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
