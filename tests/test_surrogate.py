import pytest

import numpy as np

from CADETProcess.optimization import SurrogateModel

from tests.optimization_problem_fixtures import (
    TestProblem,
    Quadratic,
    LinearConstraintsSooTestProblem,
    NonlinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    NonlinearConstraintsMooTestProblem,
    LinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
)


@pytest.fixture(params=[
    LinearConstraintsSooTestProblem,
    NonlinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    NonlinearConstraintsMooTestProblem,
    LinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
])
def test_problem(request):
    return request.param()


def create_surrogate(
        problem: TestProblem,
        n_samples: int = 1000,
        cv_tol: float = 1e-5
        ) -> SurrogateModel:
    X = problem.create_initial_values(n_samples=n_samples, seed=651)

    F = problem.evaluate_objectives(X)

    if problem.n_nonlinear_constraints > 0:
        G = problem.evaluate_nonlinear_constraints(X)
        CV = problem.evaluate_nonlinear_constraints_violation(X)
    else:
        G = None
        CV = None

    if problem.n_meta_scores > 0:
        M = problem.evaluate_meta_scores(
            X,
            n_cores=1,
        )
    else:
        M = None

    population = problem.create_population(X, F, G, M, CV=CV, cv_tol=cv_tol)

    surrogate = SurrogateModel(population)

    return surrogate


def test_objectives(test_problem):
    surrogate = create_surrogate(test_problem)

    # TODO: Evaluate other points than the ones which were used for training.
    X_test = surrogate.population.feasible.x[0:2]

    F_test = surrogate.population.feasible.f[0:2]
    F_est = surrogate.estimate_objectives(X_test)
    np.testing.assert_allclose(F_test, F_est, rtol=1e-3)

    if test_problem.n_meta_scores == 0:
        return

    M_test = surrogate.population.feasible.m[0:2]
    M_est = surrogate.estimate_objectives(X_test)
    np.testing.assert_allclose(M_test, M_est, rtol=1e-3)


def test_nonlinear_constraints(test_problem):
    if test_problem.n_nonlinear_constraints == 0:
        pytest.skip()

    surrogate = create_surrogate(test_problem)

    # TODO: Evaluate other points than the ones which were used for training.
    X_test = surrogate.population.feasible.x[0:2]

    G_test = surrogate.population.feasible.g[0:2]
    G_est = surrogate.estimate_nonlinear_constraints(X_test)
    np.testing.assert_allclose(G_test, G_est, rtol=1e-3)

    CV_test = surrogate.population.feasible.cv[0:2]
    CV_est = surrogate.estimate_nonlinear_constraints_violation(X_test)
    np.testing.assert_allclose(CV_test, CV_est, rtol=1e-3)


# %%

if __name__ == "__main__":
    pytest.main([__file__])
