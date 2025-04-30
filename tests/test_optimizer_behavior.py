from functools import partial

import numpy as np
import pytest
from CADETProcess.optimization import (
    COBYLA,
    SLSQP,
    U_NSGA3,
    NelderMead,
    OptimizerBase,
    TrustConstr,
)

skip_ax = False
try:
    from CADETProcess.optimization import (
        GPEI,
        NEHVI,
        qNParEGO,
    )
except ImportError:
    skip_ax = True

from tests.optimization_problem_fixtures import (
    LinearConstraintsMooTestProblem,
    LinearConstraintsSooTestProblem,
    LinearConstraintsSooTestProblem2,
    LinearEqualityConstraintsSooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
    NonlinearConstraintsMooTestProblem,
    NonlinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    Rosenbrock,
    TestProblem,
)  # noqa: E402

# %% Optimizer Setup

SOO_TEST_KWARGS = {
    "atol": 0.05,   # allows absolute 0.05 deviation (low values) of solution or
    "rtol": 0.1,    # 0.1 alows 10% deviation of true solution
}

MOO_TEST_KWARGS = {
    "atol": 0.01,
    "rtol": 0.1,
    "mismatch_tol": 0.3333333333,  # 75 % of all solutions must lie on the pareto front
}

F_TOL = 0.001
X_TOL = 0.001

CV_BOUNDS_TOL = 1e-9
CV_LINCON_TOL = 1e-6
CV_LINEQCON_TOL = 1e-9
CV_NONLINCON_TOL = 0.0001

EXCLUDE_COMBINATIONS = [
    (
        U_NSGA3,
        LinearEqualityConstraintsSooTestProblem,
        "See also: https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/152",
    )
]

# this helps to test optimizers for hard problems
NON_DEFAULT_PARAMETERS = [
    (U_NSGA3, NonlinearConstraintsMooTestProblem, {"pop_size": 300, "n_max_gen": 40}),
    (U_NSGA3, Rosenbrock, {"pop_size": 300, "n_max_gen": 20}),
]


def skip_if_combination_excluded(optimizer, problem):
    for o, p, r in EXCLUDE_COMBINATIONS:
        if isinstance(optimizer, o) and isinstance(problem, p):
            pytest.skip(reason=r)


def set_non_default_parameters(optimizer, problem):
    for o, p, params in NON_DEFAULT_PARAMETERS:
        if isinstance(optimizer, o) and isinstance(problem, p):
            # override default parameters of the optimizer
            for pk, pv in params.items():
                setattr(optimizer, pk, pv)


class TrustConstr(TrustConstr):
    x_tol = X_TOL
    cv_nonlincon_tol = CV_NONLINCON_TOL


class COBYLA(COBYLA):
    x_tol = X_TOL
    cv_nonlincon_tol = CV_NONLINCON_TOL


class NelderMead(NelderMead):
    x_tol = X_TOL
    f_tol = F_TOL


class SLSQP(SLSQP):
    x_tol = X_TOL
    cv_lincon_tol = CV_LINCON_TOL


class U_NSGA3(U_NSGA3):
    f_tol = F_TOL
    x_tol = X_TOL
    cv_nonlincon_tol = CV_NONLINCON_TOL
    pop_size = 100
    n_max_gen = 20  # before used 100 generations --> this did not improve the fit


if not skip_ax:

    class GPEI(GPEI):
        cv_lincon_tol = CV_LINCON_TOL
        n_init_evals = 40
        early_stopping_improvement_bar = 1e-4
        early_stopping_improvement_window = 10
        n_max_evals = 50

    class NEHVI(NEHVI):
        cv_lincon_tol = CV_LINCON_TOL
        n_init_evals = 50
        early_stopping_improvement_bar = 1e-4
        early_stopping_improvement_window = 10
        n_max_evals = 60

    class qNParEGO(qNParEGO):
        cv_lincon_tol = CV_LINCON_TOL
        n_init_evals = 50
        early_stopping_improvement_bar = 1e-4
        early_stopping_improvement_window = 10
        n_max_evals = 70


# %% Test problem factory


@pytest.fixture(
    params=[
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
        # transformed problems
        partial(LinearConstraintsSooTestProblem, transform="linear"),
        partial(LinearEqualityConstraintsSooTestProblem, transform="linear"),
        partial(NonlinearLinearConstraintsSooTestProblem, transform="linear"),
    ]
)
def optimization_problem(request):
    return request.param(use_diskcache=False)


params = [
    TrustConstr,
    COBYLA,
    SLSQP,
    NelderMead,
    U_NSGA3,
]

if not skip_ax:
    params.extend(
        [
            GPEI,
            NEHVI,
            qNParEGO,
        ]
    )
    EXCLUDE_COMBINATIONS.append(
        (GPEI, Rosenbrock, "cannot solve problem with enough accuracy fast enough.")
    )

    # this helps to test optimizers for hard problems
    NON_DEFAULT_PARAMETERS.append(
        (NEHVI, LinearConstraintsMooTestProblem, {"n_init_evals": 20, "n_max_evals": 40})
    )


@pytest.fixture(params=params)
def optimizer(request):
    optimizer = request.param()
    optimizer.progress_freqency = None
    return optimizer


# %% Tests
@pytest.mark.slow
def test_convergence(optimization_problem: TestProblem, optimizer: OptimizerBase):
    # only test problems that the optimizer can handle. The rest of the tests
    # will be marked as passed
    pytest.skip()

    if optimizer.check_optimization_problem(optimization_problem):
        results = optimizer.optimize(
            optimization_problem=optimization_problem,
            save_results=False,
        )
        if optimization_problem.n_objectives == 1:
            optimization_problem.test_if_solved(results, SOO_TEST_KWARGS)
        else:
            optimization_problem.test_if_solved(results, MOO_TEST_KWARGS)


@pytest.mark.slow
def test_from_initial_values(
    optimization_problem: TestProblem, optimizer: OptimizerBase
):
    if optimizer.check_optimization_problem(optimization_problem):
        skip_if_combination_excluded(optimizer, optimization_problem)
        set_non_default_parameters(optimizer, optimization_problem)

        results = optimizer.optimize(
            optimization_problem=optimization_problem,
            x0=optimization_problem.x0,
            save_results=False,
        )
        if optimization_problem.n_objectives == 1:
            optimization_problem.test_if_solved(results, SOO_TEST_KWARGS)
        else:
            optimization_problem.test_if_solved(results, MOO_TEST_KWARGS)


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


@pytest.mark.slow
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
