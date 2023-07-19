import unittest
from copy import deepcopy

from CADETProcess.optimization import (
    OptimizationResults, Population, Individual, Surrogate, settings,
    OptimizationProblem
)
from tests.optimization_problem_fixtures import (
    LinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblem,
    NonlinearConstraintsMooTestProblem,
    LinearConstraintsMooTestProblem,
    LinearNonlinearConstraintsMooTestProblem,
)

def generate_samples(problem: OptimizationProblem):
    X = problem.create_initial_values(n_samples=100)
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
        M = len(X)*[None]

    if problem.n_nonlinear_constraints == 0:
        G = len(X)*[None]
        CV = len(X)*[None]


    return X, F, M, G, CV


def generate_optimization_results(problem):
    results = OptimizationResults(optimization_problem=problem, optimizer=None)
    cv_tol = 1e-5

    X, F, M, G, CV = generate_samples(problem)

    population = Population()
    for x, f, g, cv, m in zip(X, F, G, CV, M):
        x_untransformed \
            = problem.get_dependent_values(
                x, untransform=True
            )
        ind = Individual(
            x, f, g, m, cv, cv_tol, x_untransformed,
            problem.independent_variable_names,
            problem.objective_labels,
            problem.nonlinear_constraint_labels,
            problem.meta_score_labels,
            problem.variable_names,
        )
        population.add_individual(ind)

    results.update_population(population)
    results._results_directory = f"work/tests/{problem.name}"

    return results

def surrogate_lc_soo(has_evaluator=False):
    problem = LinearConstraintsSooTestProblem(has_evaluator=has_evaluator)
    results = generate_optimization_results(problem)
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
    "lc_soo_eval": surrogate_lc_soo(has_evaluator=True),
    "nlc_lc_soo": surrogate_nlc_lc_soo(),
    "lc_moo": surrogate_lc_moo(),
    "nlc_moo": surrogate_nlc_moo(has_evaluator=False),
    "nlc_moo_eval": surrogate_nlc_moo(has_evaluator=True),
    "nlc_lc_moo": surrogate_nlc_lc_moo(),
    "nlc_lc_moo_eval": surrogate_nlc_lc_moo(has_evaluator=True)
}


class Test_SurrogateDimensionality(unittest.TestCase):
    """
    test if dimensionalities of objectives and constraints of surrogate and
    simulator match
    """
    def test_moo(self):
        surrogate = fixtures["nlc_lc_soo"]
        var = surrogate.optimization_problem.variables[0]
        n_objs = surrogate.optimization_problem.n_objectives

        op_sur, cond_var_idx, free_var_idx = surrogate.condition_optimization_problem(
            conditional_vars={var.name: var.lb},
            objective_index=[range(n_objs)[0]],
            use_surrogate=False
        )

        op_sim, cond_var_idx, free_var_idx = surrogate.condition_optimization_problem(
            conditional_vars={var.name: var.lb},
            objective_index=[range(n_objs)[0]],
            use_surrogate=False
        )
        X = surrogate.X
        # TODO: unexpected behavior, both evaluations return the same number
        #       although op_xxx is a deepcopy of surrogate.optimization_problem
        #       when running the same in two different processes, sequentially,
        #       the results differ.
        F_sim = op_sim.evaluate_objectives(X[80, free_var_idx])
        F_sur = op_sur.evaluate_objectives(X[80, free_var_idx])


class Test_Surrogate(unittest.TestCase):
    @staticmethod
    def _find_minimum(surrogate):
        # test if problem runs on surrogate
        for i in range(surrogate.optimization_problem.n_independent_variables):
            surrogate.find_minimum(i, use_surrogate=True, n=3)

        # test if problem runs on normal model
        for i in range(surrogate.optimization_problem.n_independent_variables):
            surrogate.find_minimum(i, use_surrogate=False, n=3)

    def test_linear_constraints_soo(self):
        surrogate = fixtures["lc_soo"]
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

    def test_linear_constraints_soo_evaluator(self):
        surrogate = fixtures["lc_soo_eval"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_linear_constraints_moo(self):
        surrogate = fixtures["nlc_lc_moo"]
        self._find_minimum(surrogate)

    def test_nonlinear_constraints_linear_constraints_moo_evaluator(self):
        surrogate = fixtures["nlc_lc_moo_eval"]
        self._find_minimum(surrogate)

if __name__ == "__main__":
    settings.working_directory = "work"
    # Test_SurrogateDimensionality().test_moo()
    # Test_Surrogate().test_nonlinear_constraints_moo()
    Test_Surrogate().test_nonlinear_constraints_moo_evaluator()
    # Test_Surrogate().test_linear_constraints_soo_evaluator()
    # Test_Surrogate().test_nonlinear_constraints_linear_constraints_soo()
    # Test_Surrogate().test_nonlinear_constraints_linear_constraints_moo()
    unittest.main()
