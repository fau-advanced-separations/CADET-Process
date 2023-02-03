import unittest

import numpy as np

from CADETProcess.optimization import OptimizationResults
from CADETProcess.optimization import U_NSGA3

from test_population import setup_population
from test_optimization_problem import setup_optimization_problem


def setup_optimization_results(
        n_gen=3, n_ind=3, n_vars=2, n_obj=1, n_nonlin=0, n_meta=0, rng=None,
        initialize_data=True):
    optimization_problem = setup_optimization_problem(n_vars, n_obj, n_nonlin, n_meta)
    optimizer = U_NSGA3()

    optimization_results = OptimizationResults(optimization_problem, optimizer)

    if initialize_data:
        if rng is None:
            rng = np.random.default_rng(12345)

        for gen in range(n_gen):
            pop = setup_population(n_ind, n_vars, n_obj, n_nonlin, n_meta, rng)
            optimization_results.update_population(pop)
            optimization_results.update_pareto()
            if n_meta > 0:
                optimization_results.update_meta()

    return optimization_results


class TestOptimizationResults(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.optimization_results = setup_optimization_results()

    def test_history(self):
        n_gen_expected = 3
        n_gen = self.optimization_results.n_gen
        self.assertEqual(n_gen, n_gen_expected)

        n_evals_expected = 9
        n_evals = self.optimization_results.n_evals
        self.assertEqual(n_evals, n_evals_expected)

        f_min_history_expected = np.array([
            [-0.79736546],
            [-0.94888115],
            [-0.94888115],
        ])
        f_min_history = self.optimization_results.f_min_history
        np.testing.assert_almost_equal(f_min_history, f_min_history_expected)

        g_min_history_expected = None
        g_min_history = self.optimization_results.g_min_history
        self.assertIsNone(g_min_history, g_min_history_expected)

        m_min_history_expected = None
        m_min_history = self.optimization_results.m_min_history
        self.assertIsNone(m_min_history, m_min_history_expected)

    def test_serialization(self):
        data = self.optimization_results.to_dict()

        results_new = setup_optimization_results(initialize_data=False)
        results_new.update_from_dict(data)
        data_new = results_new.to_dict()
        np.testing.assert_equal(data, data_new)


if __name__ == '__main__':
    unittest.main()
