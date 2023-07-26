import unittest
from copy import deepcopy

import numpy as np

from CADETProcess.optimization import (
    OptimizationResults,
    Population,
    Individual,
    Surrogate,
    settings,
    OptimizationProblem,
)
from tests.optimization_problem_fixtures import (
    TestProblem,
    LinearConstraintsSooTestProblem,
    NonlinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    NonlinearConstraintsMooTestProblem,
    LinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
)


def generate_samples(problem: OptimizationProblem, n_samples):
    X = problem.create_initial_values(n_samples=n_samples, seed=651)
    F = problem.evaluate_objectives_population(X)
    G = problem.evaluate_nonlinear_constraints_population(X)
    CV = problem.evaluate_nonlinear_constraints_violation_population(X)

    if problem.n_meta_scores > 0:
        M = problem.evaluate_meta_scores_population(
            X,
            untransform=True,
            n_cores=1,
        )
    else:
        M = len(X) * [None]

    if problem.n_nonlinear_constraints == 0:
        G = len(X) * [None]
        CV = len(X) * [None]

    return X, F, M, G, CV


def generate_optimization_results(problem, n_samples=200):
    results = OptimizationResults(optimization_problem=problem, optimizer=None)
    cv_tol = 1e-5

    X, F, M, G, CV = generate_samples(problem, n_samples=n_samples)

    population = Population()
    for x, f, g, cv, m in zip(X, F, G, CV, M):
        x_untransformed = problem.get_dependent_values(x, untransform=True)
        ind = Individual(
            x=x,
            f=f,
            g=g,
            m=m,
            cv=cv,
            cv_tol=cv_tol,
            x_transformed=x_untransformed,
            independent_variable_names=problem.independent_variable_names,
            objective_labels=problem.objective_labels,
            contraint_labels=problem.nonlinear_constraint_labels,
            meta_score_labels=problem.meta_score_labels,
            variable_names=problem.variable_names,
        )
        population.add_individual(ind)

    results.update_population(population)

    return results


def surrogate_lc_soo(has_evaluator=False):
    problem = LinearConstraintsSooTestProblem(has_evaluator=has_evaluator)
    results = generate_optimization_results(problem)
    return Surrogate(optimization_results=results)


def surrogate_nlc_soo(has_evaluator=False):
    problem = NonlinearConstraintsSooTestProblem(has_evaluator=has_evaluator)
    results = generate_optimization_results(problem, n_samples=1000)
    return Surrogate(optimization_results=results)


def surrogate_nlc_lc_soo():
    problem = NonlinearLinearConstraintsSooTestProblem()
    results = generate_optimization_results(problem)
    return Surrogate(optimization_results=results)


def surrogate_nlc_moo(has_evaluator=False):
    problem = NonlinearConstraintsMooTestProblem(has_evaluator=has_evaluator)
    results = generate_optimization_results(problem)
    return Surrogate(optimization_results=results)


def surrogate_lc_moo():
    problem = LinearConstraintsMooTestProblem()
    results = generate_optimization_results(problem)
    return Surrogate(optimization_results=results)


def surrogate_nlc_lc_moo(has_evaluator=False):
    problem = LinearNonlinearConstraintsMooTestProblem(has_evaluator=has_evaluator)
    results = generate_optimization_results(problem)
    return Surrogate(optimization_results=results)


fixtures = {
    "lc_soo": surrogate_lc_soo(has_evaluator=False),
    "nlc_soo": surrogate_nlc_soo(has_evaluator=False),
    "nlc_soo_eval": surrogate_lc_soo(has_evaluator=True),
    "lc_soo_eval": surrogate_lc_soo(has_evaluator=True),
    "nlc_lc_soo": surrogate_nlc_lc_soo(),
    "lc_moo": surrogate_lc_moo(),
    "nlc_moo": surrogate_nlc_moo(has_evaluator=False),
    "nlc_moo_eval": surrogate_nlc_moo(has_evaluator=True),
    "nlc_lc_moo": surrogate_nlc_lc_moo(),
    "nlc_lc_moo_eval": surrogate_nlc_lc_moo(has_evaluator=True),
}


class Test_SurrogateBehavior(unittest.TestCase):
    """
    test if dimensionalities of objectives and constraints of surrogate and
    simulator match
    """

    def calculate_nonlinear_constraints_violations(self, surrogate: Surrogate, X):
        """helper function for testing model divergence"""
        nlc_bounds = surrogate.optimization_problem.nonlinear_constraints_bounds
        g_est = surrogate.estimate_non_linear_constraints(X)
        g_est = np.array(g_est, ndmin=2)
        cv_est_calc = g_est - np.array(nlc_bounds)
        return cv_est_calc

    def test_model_divergence(self):
        """
        the issue that usage of g and cv diverge cannot be reproduced.
        Ideally the `nlc_soo` problem is better formulated to match,
        but for now I'm happy that it actually works so well.
        After all I am trying to delibreately break the surrogate model.


        """
        surrogate = fixtures["nlc_soo"]
        x_cand = surrogate.X[surrogate.feasible][0:2]
        g_cand = surrogate.G[surrogate.feasible][0:2]
        cv_cand = surrogate.CV[surrogate.feasible][0:2]

        g_est = surrogate.estimate_non_linear_constraints(x_cand)
        cv_est = surrogate.estimate_nonlinear_constraints_violation(x_cand)

        # test if CV and G match but are not the same
        assert np.allclose(g_cand, g_est, atol=1e-2) and not np.any(g_est == g_cand)
        assert np.allclose(cv_cand, cv_est, atol=1e-2) and not np.any(cv_est == cv_cand)

        ok_est_1 = surrogate.estimate_check_nonlinear_constraints(x_cand)
        ok_est_2 = np.all(
            self.calculate_nonlinear_constraints_violations(
                surrogate=surrogate, X=x_cand
            )
            <= 0,
            axis=1,
        )

        # test if all methods arrive at the same solution depending on
        # this works as long as candidate from the training set is used
        assert all(ok_est_1) and all(ok_est_2)

        # vary x_0
        deviations = [0.001, 0.01, 0.05, 0.1, 1.0]
        stability = []
        for dev in deviations:
            x_cand = surrogate.X[surrogate.feasible][0].copy()
            x_cand[0] += dev

            cv_est_1 = surrogate.estimate_nonlinear_constraints_violation(x_cand)
            cv_est_2 = self.calculate_nonlinear_constraints_violations(
                surrogate=surrogate, X=x_cand
            )
            stability.append(cv_est_1 - cv_est_2)

        stability = np.array(stability)

        assert np.all(stability.std(axis=0) < 0.01)

class Test_Surrogate(unittest.TestCase):
    @classmethod
    def _find_minimum(cls, surrogate):
        problem = surrogate.optimization_problem
        # test if problem runs on surrogate
        for i in range(problem.n_independent_variables):
            fmin, xopt = surrogate.find_minimum(i, use_surrogate=True, n=3)
            problem.test_points_on_conditional_minimum(xopt, fmin, i)

        # test if problem runs on normal model
        for i in range(problem.n_independent_variables):
            fmin, xopt = surrogate.find_minimum(i, use_surrogate=False, n=3)
            problem.test_points_on_conditional_minimum(xopt, fmin, i)

    def test_linear_constraints_soo(self):
        surrogate = fixtures["lc_soo"]
        self._find_minimum(surrogate)

    def test_linear_constraints_soo_evaluator(self):
        surrogate = fixtures["lc_soo_eval"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_soo(self):
        surrogate = fixtures["nlc_soo"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_soo_evaluator(self):
        surrogate = fixtures["nlc_soo_eval"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_linear_constraints_soo(self):
        surrogate = fixtures["nlc_lc_soo"]
        self._find_minimum(surrogate)

    def test_linear_constraints_moo(self):
        surrogate = fixtures["lc_moo"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_moo(self):
        surrogate = fixtures["nlc_moo"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_moo_evaluator(self):
        surrogate = fixtures["nlc_moo_eval"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_linear_constraints_moo(self):
        surrogate = fixtures["nlc_lc_moo"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_linear_constraints_moo_evaluator(self):
        surrogate = fixtures["nlc_lc_moo_eval"]
        self._find_minimum(surrogate)


if __name__ == "__main__":
    unittest.main()
