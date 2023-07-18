import unittest

from CADETProcess.optimization import (
    OptimizationResults, Population, Individual, Surrogate, settings,
    OptimizationProblem
)
from tests.optimization_problem_fixtures import (
    LinearConstraintsSooTestProblem,
    NonlinearLinearConstraintsSooTestProblobjective_indexem,
    NonlinearConstraintsMooTestProblem,

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

class Test_Surrogate(unittest.TestCase):

    def test_linear_constraints_soo(self):
        problem = LinearConstraintsSooTestProblem()
        results = generate_optimization_results(problem)

        surrogate = Surrogate(optimization_results=results)

        # test if problem runs
        for i in range(problem.n_independent_variables):
            surrogate.find_minimum(i, use_surrogate=True, n=2)

    def test_nonlinear_linear_constraints_soo(self):
        problem = NonlinearLinearConstraintsSooTestProblem()
        results = generate_optimization_results(problem)

        surrogate = Surrogate(optimization_results=results)

        # test if problem runs
        for i in range(problem.n_independent_variables):
            surrogate.find_minimum(i, use_surrogate=True, n=2)

    def test_nonlinear_constraints_moo(self):
        problem = NonlinearConstraintsMooTestProblem()
        results = generate_optimization_results(problem)

        surrogate = Surrogate(optimization_results=results)

        # test if problem runs
        for i in range(problem.n_independent_variables):
            surrogate.find_minimum(i, use_surrogate=True, n=2)


if __name__ == "__main__":
    settings.working_directory = "work"
    Test_Surrogate().test_nonlinear_constraints_moo()
    # Test_Surrogate().test_nonlinear_linear_constraints_soo()
    # Test_Surrogate().test_linear_constraints_soo()
    # unittest.main()
