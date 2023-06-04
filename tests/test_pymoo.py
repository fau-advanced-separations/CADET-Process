import unittest

from CADETProcess.optimization import U_NSGA3

from test_optimization_problem import setup_optimization_problem


class Test_OptimizationProblemSimple(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_restart_from_checkpoint(self):
        class Callback():
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


if __name__ == '__main__':
    unittest.main()
