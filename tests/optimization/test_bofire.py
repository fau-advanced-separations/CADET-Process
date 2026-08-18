"""Tests for the BoFire Bayesian-optimization adapter.

The fast tests pin the ``OptimizationProblem -> Domain`` translation
(``_build_domain``).  The end-to-end tests pin the two port fixes that only
surface inside ``_run``: the ``run_post_processing`` call shape and the
checkpoint-restore evaluation count.
"""

from types import SimpleNamespace

import numpy as np
import pytest

bofire = pytest.importorskip("bofire.strategies.api")

import CADETProcess.optimization.bofireAdapter as bofire_adapter
from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.objectives.api import MinimizeSigmoidObjective
from CADETProcess.optimization import BoFire
from CADETProcess.optimization.bofireAdapter import (
    _build_domain,
    _build_experiments,
    _population_to_experiments,
)

from tests.optimization.conftest import (
    LinearConstraintsMooTestProblem,
    LinearConstraintsSooTestProblem,
    NonlinearConstraintsSooTestProblem,
    Rosenbrock,
)

# %% Domain translation (fast)


def test_build_domain_maps_variables_to_continuous_inputs():
    """One ContinuousInput per independent variable, bounds from transformed space."""
    op = Rosenbrock(use_diskcache=False)
    domain, input_keys, obj_keys, con_keys = _build_domain(op)

    assert input_keys == ["x_var_0", "x_var_1"]
    assert obj_keys == ["obj__objective_function"]
    assert con_keys == []

    ts = op.transformed_space
    for i, key in enumerate(input_keys):
        feature = domain.inputs.get_by_key(key)
        assert isinstance(feature, ContinuousInput)
        assert feature.bounds == (
            pytest.approx(float(ts.lower_bounds[i])),
            pytest.approx(float(ts.upper_bounds[i])),
        )


def test_build_domain_translates_linear_constraints():
    """Linear inequality constraints become BoFire LinearInequalityConstraints."""
    op = LinearConstraintsSooTestProblem(use_diskcache=False)
    domain, input_keys, _, _ = _build_domain(op)

    constraints = domain.constraints.constraints
    assert len(constraints) == op.n_linear_constraints == 1
    assert isinstance(constraints[0], LinearInequalityConstraint)
    assert constraints[0].features == input_keys


def test_build_domain_models_nonlinear_constraints_as_sigmoid_outputs():
    """Each nonlinear constraint becomes a GP output with a sigmoid objective at 0."""
    op = NonlinearConstraintsSooTestProblem(use_diskcache=False)
    domain, _, obj_keys, con_keys = _build_domain(op)

    assert len(con_keys) == op.n_nonlinear_constraints == 3
    # Nonlinear constraints are outputs, not domain constraints.
    assert domain.constraints.constraints == []
    for key in con_keys:
        objective = domain.outputs.get_by_key(key).objective
        assert isinstance(objective, MinimizeSigmoidObjective)
        assert objective.tp == 0.0


def test_build_domain_maps_multiple_objectives():
    """One output per objective for a multi-objective problem."""
    op = LinearConstraintsMooTestProblem(use_diskcache=False)
    _, _, obj_keys, _ = _build_domain(op)

    assert obj_keys == ["obj_f1", "obj_f2"]
    assert len(obj_keys) == op.n_objectives == 2


# %% Experiment translation (fast)


def test_build_experiments_marks_nonfinite_outputs_invalid():
    """Non-finite objectives and constraints are excluded from BoFire fitting."""
    experiments = _build_experiments(
        X_transformed=np.array([[0.1], [0.2], [0.3]]),
        input_keys=["x_var"],
        F=np.array([[1.0, np.inf], [np.nan, 2.0], [3.0, 4.0]]),
        obj_keys=["obj_a", "obj_b"],
        CV=np.array([[0.0], [np.inf], [-1.0]]),
        con_keys=["con_c"],
    )

    np.testing.assert_array_equal(
        experiments["valid_obj_a"], [True, False, True]
    )
    np.testing.assert_array_equal(
        experiments["valid_obj_b"], [False, True, True]
    )
    np.testing.assert_array_equal(
        experiments["valid_con_c"], [True, False, True]
    )
    assert np.isnan(experiments.loc[1, "obj_a"])
    assert np.isnan(experiments.loc[0, "obj_b"])
    assert np.isnan(experiments.loc[1, "con_c"])


def test_population_restore_rebuilds_output_validity_columns():
    """Checkpoint replay preserves the validity semantics of fresh evaluations."""
    pop = SimpleNamespace(
        x_transformed=np.array([[0.1], [0.2]]),
        f_minimized=np.array([[1.0], [np.inf]]),
        cv_nonlincon=np.array([[np.nan], [-1.0]]),
    )

    experiments = _population_to_experiments(
        pop,
        input_keys=["x_var"],
        obj_keys=["obj_a"],
        con_keys=["con_c"],
    )

    np.testing.assert_array_equal(experiments["valid_obj_a"], [True, False])
    np.testing.assert_array_equal(experiments["valid_con_c"], [False, True])
    assert np.isnan(experiments.loc[1, "obj_a"])
    assert np.isnan(experiments.loc[0, "con_c"])


# %% End-to-end (exercise _run)


def test_run_excludes_failed_initial_evaluation_from_surrogate(monkeypatch):
    """A failed initial evaluation is recorded but not passed to the surrogate."""
    op = Rosenbrock(use_diskcache=False)

    def evaluate_batch(_opt, _X_transformed, _parallelization_backend):
        return np.array([[1.0], [np.inf], [2.0]]), None, None

    monkeypatch.setattr(bofire_adapter, "_evaluate_batch", evaluate_batch)

    optimizer = BoFire()
    optimizer.n_init = 3
    optimizer.n_max_evals = 3

    results = optimizer.optimize(op, save_results=False, log_level="ERROR")

    assert results.success
    assert np.isinf(results.populations[0].f).any()


def test_run_records_generations_via_post_processing():
    """A full ask/tell run populates results.

    Pins the ``run_post_processing(X, F, G, generation)`` call shape: the old
    five-positional call misplaced the generation into ``X_opt_transformed``
    and crashed in the pareto-front step.
    """
    op = Rosenbrock(use_diskcache=False)
    optimizer = BoFire()
    optimizer.n_init = 3
    optimizer.batch_size = 1
    optimizer.n_max_evals = 5
    optimizer.seed = 0

    results = optimizer.optimize(op, save_results=False, log_level="ERROR")

    assert results.success
    assert results.n_evals >= optimizer.n_max_evals
    assert results.n_gen >= 2
    assert np.all(np.isfinite(results.f))


@pytest.mark.slow
def test_multi_objective_run_builds_pareto_front():
    """The auto strategy runs MoboStrategy for a multi-objective problem."""
    op = LinearConstraintsMooTestProblem(use_diskcache=False)
    optimizer = BoFire()
    optimizer.n_init = 4
    optimizer.batch_size = 2
    optimizer.n_max_evals = 8
    optimizer.seed = 0

    results = optimizer.optimize(op, save_results=False, log_level="ERROR")

    assert results.success
    assert results.f.shape[1] == op.n_objectives == 2
    assert len(results.pareto_front) >= 1


@pytest.mark.slow
def test_resume_from_checkpoint_continues_evaluations(tmp_path):
    """Restore replays saved populations and continues.

    Pins the restore evaluation count fix: the old code read the removed
    ``pop.individuals`` attribute and raised ``AttributeError`` on resume.
    """
    def run(n_max_evals, use_checkpoint):
        optimizer = BoFire()
        optimizer.n_init = 3
        optimizer.batch_size = 1
        optimizer.n_max_evals = n_max_evals
        optimizer.seed = 0
        return optimizer.optimize(
            Rosenbrock(use_diskcache=False),
            save_results=True,
            results_directory=tmp_path,
            use_checkpoint=use_checkpoint,
            log_level="ERROR",
        )

    first = run(n_max_evals=4, use_checkpoint=False)
    resumed = run(n_max_evals=6, use_checkpoint=True)

    assert resumed.n_evals > first.n_evals
    # Restored generations are replayed from the checkpoint, not re-evaluated.
    np.testing.assert_allclose(
        resumed.populations[0].x, first.populations[0].x
    )


if __name__ == "__main__":
    pytest.main([__file__])
