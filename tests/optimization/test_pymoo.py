import shutil
import unittest

import numpy as np
from CADETProcess import settings
from CADETProcess.optimization import U_NSGA3, OptimizationProblem

from tests.optimization.conftest import (
    make_optimization_problem as setup_optimization_problem,
)


def test_resume_feeds_transformed_independent_coordinates(tmp_path):
    """Checkpoint restore must seed pymoo in its own coordinate system.

    The pymoo problem is defined on the transformed independent bounds; the
    restore path must feed transformed independent coordinates and minimized
    objectives, not the full physical vectors from Population.x.  With a
    normalized variable and a dependent variable, feeding physical full
    vectors mismatches both the coordinate system and the vector width.
    """
    def make_problem():
        op = OptimizationProblem("resume_norm_dep", use_diskcache=False)
        op.add_variable(
            "x0", lb=1e2, ub=1e4, normalization="log", evaluation_objects=None
        )
        op.add_variable("x1", lb=0, ub=1, evaluation_objects=None)
        op.add_variable("x2", lb=0, ub=2e4, evaluation_objects=None)
        op.add_variable_dependency("x2", "x0", lambda x0: 2 * x0)
        op.add_objective(
            lambda x: [abs(x[0] - 5e3) / 1e4 + x[1]], name="f", minimize=True
        )
        return op

    def make_optimizer(n_max_gen):
        optimizer = U_NSGA3()
        optimizer.pop_size = 6
        optimizer.n_max_gen = n_max_gen
        return optimizer

    results_first = make_optimizer(2).optimize(
        make_problem(),
        save_results=True,
        results_directory=tmp_path,
        use_checkpoint=False,
        log_level="ERROR",
    )
    assert results_first.n_gen == 2

    results_resumed = make_optimizer(4).optimize(
        make_problem(),
        save_results=True,
        results_directory=tmp_path,
        use_checkpoint=True,
        log_level="ERROR",
    )

    # Restored generations are replayed from the checkpoint, not re-evaluated.
    assert results_resumed.n_gen == 4
    np.testing.assert_allclose(
        results_resumed.populations[0].x, results_first.populations[0].x
    )

    # Every evaluated point, restored and new, respects bounds and the
    # dependency relation.
    x_all = results_resumed.population_all.x
    np.testing.assert_allclose(x_all[:, 2], 2 * x_all[:, 0])
    assert np.all(results_resumed.population_all.cv_bounds <= 1e-9)


class Test_OptimizationProblemSimple(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("./results_simple", ignore_errors=True)
        settings.working_directory = None

    def test_restart_from_checkpoint(self):
        class Callback:
            def __init__(self, n_calls=0, kill=True):
                self.n_calls = n_calls
                self.kill = kill

            def __call__(self, results):
                if self.kill and self.n_calls == 2:
                    raise Exception("Kill this!")
                self.n_calls += 1

        callback = Callback()
        optimization_problem = setup_optimization_problem()
        optimization_problem.add_callback(callback)

        optimizer = U_NSGA3()
        optimizer.n_max_gen = 5

        try:
            opt_results = optimizer.optimize(
                optimization_problem,
                save_results=True,
                use_checkpoint=False,
            )
        except Exception:
            pass

        callback.kill = False

        optimization_problem = setup_optimization_problem()
        optimization_problem.add_callback(callback)

        opt_results = optimizer.optimize(
            optimization_problem,
            save_results=True,
            use_checkpoint=True,
        )


if __name__ == "__main__":
    unittest.main()
