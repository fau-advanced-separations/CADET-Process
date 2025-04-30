import unittest

import numpy as np
from CADETProcess.performance import Performance, RankedPerformance


class PerformanceTestCase(unittest.TestCase):
    def setUp(self):
        # Create example performance data
        mass = np.array([10.0, 20.0, 30.0])
        concentration = np.array([0.5, 0.4, 0.3])
        purity = np.array([0.9, 0.85, 0.95])
        recovery = np.array([0.8, 0.75, 0.9])
        productivity = np.array([0.7, 0.6, 0.65])
        eluent_consumption = np.array([1.5, 2.0, 1.2])
        mass_balance_difference = np.array([0.2, 0.1, -0.1])

        self.performance = Performance(
            mass,
            concentration,
            purity,
            recovery,
            productivity,
            eluent_consumption,
            mass_balance_difference,
        )

    def test_attributes(self):
        self.assertEqual(self.performance.n_comp, 3)

        np.testing.assert_array_equal(
            self.performance.mass, np.array([10.0, 20.0, 30.0])
        )
        np.testing.assert_array_equal(
            self.performance.concentration, np.array([0.5, 0.4, 0.3])
        )
        np.testing.assert_array_equal(
            self.performance.purity, np.array([0.9, 0.85, 0.95])
        )
        np.testing.assert_array_equal(
            self.performance.recovery, np.array([0.8, 0.75, 0.9])
        )
        np.testing.assert_array_equal(
            self.performance.productivity, np.array([0.7, 0.6, 0.65])
        )
        np.testing.assert_array_equal(
            self.performance.eluent_consumption, np.array([1.5, 2.0, 1.2])
        )
        np.testing.assert_array_equal(
            self.performance.mass_balance_difference, np.array([0.2, 0.1, -0.1])
        )

    def test_to_dict(self):
        expected_dict = {
            "mass": [10.0, 20.0, 30.0],
            "concentration": [0.5, 0.4, 0.3],
            "purity": [0.9, 0.85, 0.95],
            "recovery": [0.8, 0.75, 0.9],
            "productivity": [0.7, 0.6, 0.65],
            "eluent_consumption": [1.5, 2.0, 1.2],
            "mass_balance_difference": [0.2, 0.1, -0.1],
        }
        self.assertDictEqual(self.performance.to_dict(), expected_dict)


class RankedPerformanceTestCase(unittest.TestCase):
    def setUp(self):
        # Create example performance data
        mass = np.array([10.0, 20.0, 30.0])
        concentration = np.array([0.5, 0.4, 0.3])
        purity = np.array([0.9, 0.85, 0.95])
        recovery = np.array([0.8, 0.75, 0.9])
        productivity = np.array([0.7, 0.6, 0.65])
        eluent_consumption = np.array([1.5, 2.0, 1.2])
        mass_balance_difference = np.array([0.2, 0.1, -0.1])

        performance = Performance(
            mass,
            concentration,
            purity,
            recovery,
            productivity,
            eluent_consumption,
            mass_balance_difference,
        )

        self.ranked_performance_equal = RankedPerformance(performance, ranking=1)
        self.ranked_performance_diff = RankedPerformance(performance, ranking=[0, 1, 2])

    def test_attributes(self):
        np.testing.assert_array_equal(self.ranked_performance_equal.mass, 20)
        np.testing.assert_array_equal(
            self.ranked_performance_diff.mass, 26.666666666666668
        )
        np.testing.assert_array_equal(
            self.ranked_performance_equal.concentration, 0.39999999999999997
        )
        np.testing.assert_array_equal(
            self.ranked_performance_diff.concentration, 0.3333333333333333
        )
        np.testing.assert_array_equal(self.ranked_performance_equal.purity, 0.9)
        np.testing.assert_array_equal(
            self.ranked_performance_diff.purity, 0.9166666666666666
        )
        np.testing.assert_array_equal(
            self.ranked_performance_equal.recovery, 0.8166666666666668
        )
        np.testing.assert_array_equal(self.ranked_performance_diff.recovery, 0.85)
        np.testing.assert_array_equal(
            self.ranked_performance_equal.productivity, 0.6499999999999999
        )
        np.testing.assert_array_equal(
            self.ranked_performance_diff.productivity, 0.6333333333333333
        )
        np.testing.assert_array_equal(
            self.ranked_performance_equal.eluent_consumption, 1.5666666666666667
        )
        np.testing.assert_array_equal(
            self.ranked_performance_diff.eluent_consumption, 1.4666666666666668
        )
        np.testing.assert_array_equal(
            self.ranked_performance_equal.mass_balance_difference, 0.06666666666666668
        )
        np.testing.assert_array_equal(
            self.ranked_performance_diff.mass_balance_difference, -0.03333333333333333
        )


if __name__ == "__main__":
    unittest.main()
