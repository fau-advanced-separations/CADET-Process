# can be moved to separate tests later on, but currently used for implementing
# and testing new features for the ax optimizer
from optimization_problem_fixtures import LinearConstraintsSooTestProblem
from CADETProcess.optimization.axAdapater import AxInterface, GPEI

def test_early_stopping():
    problem = LinearConstraintsSooTestProblem()
    optimizer = GPEI(
        n_max_evals=10,
        early_stopping_improvement_window=2,
        early_stopping_improvement_bar=0.01
    )

    optimizer.optimize(problem)

    assert optimizer.n_evals == 4

if __name__ == "__main__":
    test_early_stopping()
