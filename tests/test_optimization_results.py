import shutil
import unittest
from pathlib import Path

import numpy as np

from CADETProcess.optimization import OptimizationResults
from CADETProcess.optimization import U_NSGA3

from tests.test_population import setup_population
from tests.test_optimization_problem import setup_optimization_problem


def setup_optimization_problem_and_results(
        n_gen=3, n_ind=3, n_vars=2, n_obj=1, n_nonlin=0, n_meta=0, rng=None,
        initialize_data=True):
    optimization_problem = setup_optimization_problem(n_vars, n_obj, n_nonlin, n_meta)
    optimizer = U_NSGA3()

    optimization_results = OptimizationResults(optimization_problem, optimizer)
    results_dir = Path("tmp") / "optimization_results"

    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / "figures").mkdir(exist_ok=True, parents=True)
    optimization_results.results_directory = results_dir

    if initialize_data:
        if rng is None:
            rng = np.random.default_rng(12345)

        for gen in range(n_gen):
            pop = setup_population(n_ind, n_vars, n_obj, n_nonlin, n_meta, rng)
            optimization_results.update(pop)
            optimization_results.update_pareto()
            if n_meta > 0:
                optimization_results.update_meta()

    return optimization_problem, optimization_results


class TestOptimizationResults(unittest.TestCase):
    def setUp(self):
        _, self.optimization_results = setup_optimization_problem_and_results()

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

        _, results_new = setup_optimization_problem_and_results(initialize_data=False)
        results_new.update_from_dict(data)
        data_new = results_new.to_dict()
        np.testing.assert_equal(data, data_new)

    def test_io_serialization(self):
        optimization_problem, _ = setup_optimization_problem_and_results()

        # save_results() needs to happen after creation of optimization_problem,
        #  because during optimization_problem creation the results folder gets cleared and that would delete the save
        self.optimization_results.save_results("checkpoint")

        optimizer = U_NSGA3()
        optimization_results_new = optimizer.load_results(
            checkpoint_path=self.optimization_results.results_directory / "checkpoint.h5",
            optimization_problem=optimization_problem
        )
        np.testing.assert_equal(self.optimization_results.to_dict(), optimization_results_new.to_dict())


if __name__ == '__main__':
    unittest.main()
