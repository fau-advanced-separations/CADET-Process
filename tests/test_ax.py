# can be moved to separate tests later on, but currently used for implementing
# and testing new features for the ax optimizer
from optimization_problem_fixtures import LinearConstraintsSooTestProblem
from CADETProcess.optimization.axAdapater import AxInterface, GPEI

def test_early_stopping_nobatch():
    problem = LinearConstraintsSooTestProblem()
    optimizer = GPEI(
        n_init_evals=5,
        n_max_evals=10,
        early_stopping_improvement_window=2,
        early_stopping_improvement_bar=0.01,
        batch_size=1,
    )

    optimizer.optimize(problem)

    # pass
    assert optimizer.ax_experiment.num_trials == 5 + 2


def test_early_stopping_batch():
    problem = LinearConstraintsSooTestProblem()
    optimizer = GPEI(
        n_init_evals=5,
        n_max_evals=10,
        early_stopping_improvement_window=2,
        early_stopping_improvement_bar=0.01,
        batch_size=3,
    )

    optimizer.optimize(problem)

    # pass
    assert optimizer.ax_experiment.num_trials == 5 + 2


if __name__ == "__main__":
    test_early_stopping_batch()
