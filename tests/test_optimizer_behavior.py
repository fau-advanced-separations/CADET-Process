import numpy as np
import pytest

from CADETProcess.optimization import (
    OptimizerBase,
    TrustConstr,
    SLSQP,
    U_NSGA3,
    GPEI,
    NEHVI
)


from tests.optimization_problem_fixtures import (
    TestProblem,
    Rosenbrock,
    LinearConstraintsSooTestProblem,
    LinearConstraintsSooTestProblem2,
    LinearEqualityConstraintsSooTestProblem,
    NonlinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    LinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
    NonlinearConstraintsMooTestProblem
)

# =========================
#   Test-Optimizer Setup
# =========================

ACCURACY_DECIMAL = 1  # 1 means low accuracy only accuracy to one decimal is tested
FTOL = 0.0001
XTOL = 0.0001
GTOL = 0.0001


class TrustConstr(TrustConstr):
    f_tol = FTOL
    x_tol = XTOL
    gtol = GTOL


class SLSQP(SLSQP):
    f_tol = FTOL
    x_tol = XTOL


class U_NSGA3(U_NSGA3):
    ftol = FTOL
    xtol = XTOL
    cvtol = GTOL
    pop_size = 100
    n_max_gen = 100


class GPEI(GPEI):
    n_init_evals = 20
    early_stopping_improvement_bar=1e-4
    early_stopping_improvement_window=10
    n_max_evals=50


class NEHVI(NEHVI):
    n_init_evals = 20
    early_stopping_improvement_bar=1e-4
    early_stopping_improvement_window=10
    n_max_evals=50

# =========================
#   Test problem factory
# =========================

@pytest.fixture(params=[
    # single objective problems
    Rosenbrock,
    LinearConstraintsSooTestProblem,
    LinearConstraintsSooTestProblem2,
    NonlinearConstraintsSooTestProblem,
    LinearEqualityConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,

    # multi objective problems
    LinearConstraintsMooTestProblem,
    NonlinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
])
def optimization_problem(request):
    return request.param()

@pytest.fixture(params=[
    SLSQP,  # tests pass
    TrustConstr,  # tests pass
    U_NSGA3,  # MOO tests --> accuracy too low, LEQ test --> cannot find individuals that fulfill constraints
    # GPEI,
    # NEHVI,
])
def optimizer(request):
    return request.param()

# =========================
#          Tests
# =========================

def test_convergence(optimization_problem: TestProblem, optimizer: OptimizerBase):
    # only test problems that the optimizer can handle. The rest of the tests
    # will be marked as passed
    pytest.skip()

    if optimizer.check_optimization_problem(optimization_problem):
        results = optimizer.optimize(
            optimization_problem=optimization_problem,
            save_results=False,
        )
        optimization_problem.test_if_solved(results, decimal=ACCURACY_DECIMAL)

def test_from_initial_values(optimization_problem: TestProblem, optimizer: OptimizerBase):
    # pytest.skip()

    if optimizer.check_optimization_problem(optimization_problem):
        results = optimizer.optimize(
            optimization_problem=optimization_problem,
            x0=optimization_problem.x0,
            save_results=False,
        )
        optimization_problem.test_if_solved(results, decimal=ACCURACY_DECIMAL)

class AbortingCallback:
    """A callback that raises an exception after a specified number of calls."""

    def __init__(self, n_max_evals=2, abort=True):
        """Initialize callback with maximum number of evaluations and whether to abort."""
        self.n_calls = 0
        self.n_max_evals = n_max_evals
        self.abort = abort

    def reset(self):
        """Reset the number of calls to zero."""
        self.n_calls = 0

    def __call__(self, results):
        """Check the number of calls and raises an exception if the maximum has been reached."""
        if self.abort and self.n_calls >= self.n_max_evals:
            raise RuntimeError("Max number of evaluations reached. Aborting!")
        self.n_calls += 1

def test_resume_from_checkpoint(
        optimization_problem: TestProblem, optimizer: OptimizerBase
    ):
    pytest.skip()

    # TODO: Do we need to run this for all problems?
    if optimizer.check_optimization_problem(optimization_problem):

        callback = AbortingCallback(n_max_evals=2, abort=True)
        optimization_problem.add_callback(callback)

        # TODO: How would this work for evaluation based optimizers (vs generation based)?
        optimizer.n_max_iter = 5
        if optimizer.is_population_based:
            optimizer.pop_size = 3

        try:
            opt_results = optimizer.optimize(
                optimization_problem,
                save_results=True,
                use_checkpoint=False,
            )
        except RuntimeError as e:
            if str(e) == "Max number of evaluations reached. Aborting!":
                pass
            else:
                raise e

        results_aborted = optimizer.results
        assert results_aborted.n_gen == 3
        assert callback.n_calls == 2

        callback.reset()
        callback.abort = False

        results_full = optimizer.optimize(
            optimization_problem,
            save_results=True,
            use_checkpoint=True,
        )

        # Assert all results are present
        assert results_full.n_gen == 5
        # np.testing.assertal
        np.testing.assert_almost_equal(
            results_full.populations[1].x, results_aborted.populations[1].x
        )
        # Assert callback was only called 3 times
        assert callback.n_calls == 3


if __name__ == "__main__":
    pytest.main([__file__])
