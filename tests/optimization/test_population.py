import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.metric_space import Metric, MetricSpace
from CADETProcess.optimization import IndividualView, ParetoFront, Population


@pytest.fixture
def objective_space():
    """One scalar minimization objective."""
    space = MetricSpace()
    space.add_objective(Metric("f"))
    return space


@pytest.fixture
def constrained_space():
    """One objective plus one le-constraint with bound 1."""
    space = MetricSpace()
    space.add_objective(Metric("f"))
    space.add_constraint(Metric("g"), bound=1.0, comparison_operator="le")
    return space


@pytest.fixture
def mixed_space():
    """Maximization vector objective, ge-constraint, and a plain output."""
    space = MetricSpace()
    space.add_objective(Metric("yield", n_metrics=2), minimize=False)
    space.add_constraint(Metric("purity"), bound=0.95, comparison_operator="ge")
    space.add_metric(Metric("cost"))
    return space


@pytest.fixture
def population(objective_space):
    return Population(
        X={"var_0": [1.0, 2.0, 1.001], "var_1": [2.0, 3.0, 2.0]},
        metrics={"f": [-1.0, -2.0, -1.001]},
        metric_space=objective_space,
    )


@pytest.fixture
def constrained_population(constrained_space):
    return Population(
        X={"var_0": [1.0, 2.0], "var_1": [2.0, 3.0]},
        metrics={"f": [-1.0, -2.0], "g": [3.0, 0.0]},
        metric_space=constrained_space,
    )


@pytest.fixture
def mixed_population(mixed_space):
    return Population.from_records(
        [
            {
                "X": {"flow": 1.2, "resin": "A"},
                "metrics": {"yield": [0.8, 0.9], "purity": 0.97, "cost": 3.0},
            },
            {
                "X": {"flow": 1.5, "resin": "B"},
                "metrics": {"yield": [0.7, 0.95], "purity": 0.90, "cost": 2.0},
            },
        ],
        metric_space=mixed_space,
    )


# ── Construction and validation ──────────────────────────────────────────────


def test_metric_space_is_required():
    with pytest.raises(TypeError):
        Population(X={"x": [1.0]}, metric_space=None)


def test_undeclared_metric_raises(objective_space):
    with pytest.raises(ValueError, match="not declared"):
        Population(
            X={"x": [1.0]},
            metrics={"unknown": [1.0]},
            metric_space=objective_space,
        )


def test_metric_shape_mismatch_raises(mixed_space):
    with pytest.raises(ValueError, match="expected shape"):
        Population(
            X={"x": [1.0]},
            metrics={"yield": [[0.8, 0.9, 0.7]]},
            metric_space=mixed_space,
        )


def test_column_length_mismatch_raises(objective_space):
    with pytest.raises(ValueError, match="length"):
        Population(
            X={"a": [1.0, 2.0], "b": [1.0]},
            metric_space=objective_space,
        )


def test_metadata_restricted_to_universal_keys(objective_space):
    with pytest.raises(ValueError, match="generation"):
        Population(
            X={"x": [1.0]},
            metadata={"generation": [1]},
            metric_space=objective_space,
        )


def test_metadata_timestamp_roundtrip(objective_space):
    pop = Population(
        X={"x": [1.0]},
        metrics={"f": [0.5]},
        metadata={"timestamp": [123.0], "evaluation_time": [1.5]},
        metric_space=objective_space,
    )
    assert pop[0].metadata == {"timestamp": 123.0, "evaluation_time": 1.5}


def test_from_sample_creates_one_row(objective_space):
    pop = Population.from_sample(
        {"x": 1.0}, metrics={"f": 0.5}, metric_space=objective_space
    )
    assert len(pop) == 1
    np.testing.assert_allclose(pop.f, [[0.5]])


def test_empty_population_has_no_rows(objective_space):
    pop = Population.empty(metric_space=objective_space)
    assert len(pop) == 0


def test_columns_are_read_only(population):
    with pytest.raises(ValueError):
        population.X["var_0"][0] = 99.0
    with pytest.raises(ValueError):
        population.metrics["f"][0] = 99.0


# ── Indexing ──────────────────────────────────────────────────────────────────


def test_scalar_index_returns_view(population):
    view = population[0]
    assert isinstance(view, IndividualView)
    assert view.X == {"var_0": 1.0, "var_1": 2.0}
    np.testing.assert_allclose(view.metrics["f"], -1.0)


def test_slice_and_mask_return_population(population):
    assert isinstance(population[0:2], Population)
    assert len(population[0:2]) == 2
    mask = np.array([True, False, True])
    assert len(population[mask]) == 2
    assert len(population[[1]]) == 1


def test_view_as_population_is_one_row(population):
    one = population[1].as_population()
    assert isinstance(one, Population)
    assert len(one) == 1
    np.testing.assert_allclose(one.f, [[-2.0]])


def test_view_as_record_roundtrips(objective_space, population):
    record = population[0].as_record()
    rebuilt = Population.from_records([record], metric_space=objective_space)
    np.testing.assert_allclose(rebuilt.f, [[-1.0]])


def test_iteration_yields_views(population):
    views = list(population)
    assert len(views) == 3
    assert all(isinstance(v, IndividualView) for v in views)


# ── Identity (id) ─────────────────────────────────────────────────────────────


def test_id_matches_between_view_and_population(population):
    """IndividualView.id and Population.ids must agree for the same row."""
    for i, view in enumerate(population):
        assert view.id == population.ids[i]


def test_id_is_deterministic(objective_space):
    """Identical parameter values, even across separate populations, share an id."""
    pop_a = Population(
        X={"var_0": [1.0], "var_1": [2.0]},
        metrics={"f": [-1.0]},
        metric_space=objective_space,
    )
    pop_b = Population(
        X={"var_0": [1.0], "var_1": [2.0]},
        metrics={"f": [-9.0]},
        metric_space=objective_space,
    )
    assert pop_a[0].id == pop_b[0].id


def test_id_differs_for_different_rows(population):
    assert population[0].id != population[1].id


def test_id_short_is_prefix_of_id(population):
    view = population[0]
    assert view.id_short == view.id[:7]
    assert len(view.id_short) == 7


def test_id_handles_categorical_columns(mixed_population):
    """Object-dtype (categorical) parameter columns must not hash pointer bytes."""
    assert mixed_population[0].id != mixed_population[1].id


def test_id_is_deterministic_for_categorical_columns(mixed_space):
    """Same categorical value in two independently-built populations must
    yield the same id.

    A naive ``ndarray.tobytes()`` on an object-dtype row hashes Python object
    pointers rather than values, which would pass within a single population
    (same underlying objects) but fail here, since the two rows are distinct
    Python string objects with the same value.
    """
    pop_a = Population.from_records(
        [{"X": {"flow": 1.2, "resin": "A"},
          "metrics": {"yield": [0.8, 0.9], "purity": 0.97, "cost": 3.0}}],
        metric_space=mixed_space,
    )
    pop_b = Population.from_records(
        [{"X": {"flow": 1.2, "resin": "A"},
          "metrics": {"yield": [0.1, 0.2], "purity": 0.50, "cost": 9.0}}],
        metric_space=mixed_space,
    )
    assert pop_a[0].id == pop_b[0].id


# ── Value matching ────────────────────────────────────────────────────────────


def test_index_of_exact_match(population):
    assert population.index_of([2.0, 3.0]) == 1
    assert population.index_of({"var_0": 2.0, "var_1": 3.0}) == 1


def test_index_of_no_match_raises(population):
    with pytest.raises(KeyError):
        population.index_of([9.0, 9.0])


def test_contains_uses_exact_values(population):
    assert [1.0, 2.0] in population
    assert [1.0000001, 2.0] not in population


def test_drop_duplicates_keeps_first(objective_space):
    pop = Population(
        X={"x": [1.0, 2.0, 1.0]},
        metrics={"f": [0.1, 0.2, 0.3]},
        metric_space=objective_space,
    )
    deduped = pop.drop_duplicates()
    assert len(deduped) == 2
    np.testing.assert_allclose(deduped.metrics["f"], [0.1, 0.2])


# ── Concat ────────────────────────────────────────────────────────────────────


def test_concat_stacks_rows(population):
    combined = Population.concat([population, population[[0]]])
    assert len(combined) == 4
    np.testing.assert_allclose(combined.X["var_0"][-1], 1.0)


def test_concat_mismatched_columns_raises(objective_space):
    p1 = Population(X={"a": [1.0]}, metric_space=objective_space)
    p2 = Population(X={"b": [1.0]}, metric_space=objective_space)
    with pytest.raises(CADETProcessError):
        Population.concat([p1, p2])


def test_concat_ignores_empty(objective_space, population):
    combined = Population.concat(
        [Population.empty(metric_space=objective_space), population]
    )
    assert len(combined) == 3


# ── Projections ───────────────────────────────────────────────────────────────


def test_x_projection_column_order(population):
    np.testing.assert_allclose(
        population.x,
        [[1.0, 2.0], [2.0, 3.0], [1.001, 2.0]],
    )


def test_x_projection_holds_categoricals(mixed_population):
    x = mixed_population.x
    assert x.dtype == object
    assert x[0, 1] == "A"


def test_f_direction_projections(mixed_population):
    # yield is a maximization objective: f is raw, f_minimized is negated
    np.testing.assert_allclose(mixed_population.f, [[0.8, 0.9], [0.7, 0.95]])
    np.testing.assert_allclose(
        mixed_population.f_minimized, [[-0.8, -0.9], [-0.7, -0.95]]
    )
    np.testing.assert_allclose(mixed_population.f_best, [0.8, 0.95])


def test_f_statistics(constrained_population):
    np.testing.assert_allclose(constrained_population.f_min, [-2.0])
    np.testing.assert_allclose(constrained_population.f_max, [-1.0])
    np.testing.assert_allclose(constrained_population.f_avg, [-1.5])


def test_g_and_cv_nonlincon(constrained_population):
    np.testing.assert_allclose(constrained_population.g, [[3.0], [0.0]])
    # le-constraint with bound 1: cv = g - 1
    np.testing.assert_allclose(
        constrained_population.cv_nonlincon, [[2.0], [-1.0]]
    )


def test_cv_nonlincon_ge_constraint(mixed_population):
    # ge-constraint with bound 0.95: cv = 0.95 - purity
    np.testing.assert_allclose(
        mixed_population.cv_nonlincon, [[-0.02], [0.05]], atol=1e-12
    )


def test_plain_metrics_are_unannotated_only(mixed_population):
    assert mixed_population.plain_metric_labels == ["cost"]
    np.testing.assert_allclose(mixed_population.plain_metrics, [[3.0], [2.0]])


def test_empty_projections_have_zero_width(objective_space):
    pop = Population(
        X={"x": [1.0]}, metrics={"f": [0.5]}, metric_space=objective_space
    )
    assert pop.g.shape == (1, 0)
    assert pop.cv_nonlincon.shape == (1, 0)
    assert pop.plain_metrics.shape == (1, 0)


# ── Feasibility ───────────────────────────────────────────────────────────────


def test_is_feasible_from_nonlinear_constraints(mixed_population):
    np.testing.assert_array_equal(
        mixed_population.is_feasible(), [True, False]
    )


def test_is_feasible_respects_tolerance(mixed_population):
    np.testing.assert_array_equal(
        mixed_population.is_feasible(cv_nonlincon_tol=0.1), [True, True]
    )


def test_feasible_infeasible_split(mixed_population):
    assert len(mixed_population.feasible) == 1
    assert len(mixed_population.infeasible) == 1
    assert mixed_population.feasible.X["resin"][0] == "A"


# ── Dominance and similarity ──────────────────────────────────────────────────


def test_dominates_reads_direction_from_metric_space(mixed_space):
    pop = Population.from_records(
        [
            {"X": {"flow": 1.0, "resin": "A"},
             "metrics": {"yield": [0.9, 0.9], "purity": 0.99, "cost": 1.0}},
            {"X": {"flow": 2.0, "resin": "A"},
             "metrics": {"yield": [0.8, 0.8], "purity": 0.99, "cost": 1.0}},
        ],
        metric_space=mixed_space,
    )
    # Higher yield wins because the objective is a maximization.
    assert pop.dominates(0, 1)
    assert not pop.dominates(1, 0)


def test_feasible_dominates_infeasible(mixed_population):
    assert mixed_population.dominates(0, 1)
    assert not mixed_population.dominates(1, 0)


def test_no_domination_on_tradeoff(objective_space):
    space = MetricSpace()
    space.add_objective(Metric("f", n_metrics=2))
    pop = Population(
        X={"x": [1.0, 2.0]},
        metrics={"f": [[-1.0, -2.0], [-2.0, -1.0]]},
        metric_space=space,
    )
    assert not pop.dominates(0, 1)
    assert not pop.dominates(1, 0)


def test_is_similar_relative_tolerance(population):
    assert population.is_similar(0, 2, tol=1e-1)
    assert not population.is_similar(0, 2, tol=1e-8)
    assert not population.is_similar(0, 1, tol=1e-1)


def test_drop_similar_removes_near_duplicates(population):
    reduced = population.drop_similar(1e-1)
    assert len(reduced) == 2


# ── ParetoFront ───────────────────────────────────────────────────────────────


def test_pareto_front_keeps_nondominated(objective_space):
    front = ParetoFront(similarity_tol=0, metric_space=objective_space)
    pop = Population(
        X={"x": [1.0, 2.0]},
        metrics={"f": [-1.0, -2.0]},
        metric_space=objective_space,
    )
    new, significant = front.update_population(pop)
    assert significant
    assert len(front) == 1
    np.testing.assert_allclose(front.f, [[-2.0]])


def test_pareto_front_dominated_candidate_rejected(objective_space):
    front = ParetoFront(similarity_tol=0, metric_space=objective_space)
    front.update_population(
        Population(
            X={"x": [1.0]}, metrics={"f": [-2.0]}, metric_space=objective_space
        )
    )
    new, significant = front.update_population(
        Population(
            X={"x": [2.0]}, metrics={"f": [-1.0]}, metric_space=objective_space
        )
    )
    assert len(new) == 0
    assert len(front) == 1
    np.testing.assert_allclose(front.f, [[-2.0]])


def test_pareto_front_multi_objective_tradeoff_kept(mixed_space):
    front = ParetoFront(similarity_tol=0, metric_space=mixed_space)
    pop = Population.from_records(
        [
            {"X": {"flow": 1.0, "resin": "A"},
             "metrics": {"yield": [0.9, 0.7], "purity": 0.99, "cost": 1.0}},
            {"X": {"flow": 2.0, "resin": "B"},
             "metrics": {"yield": [0.7, 0.9], "purity": 0.99, "cost": 1.0}},
        ],
        metric_space=mixed_space,
    )
    front.update_population(pop)
    assert len(front) == 2


def test_pareto_front_exact_duplicate_not_added(objective_space):
    """Re-evaluating a front member must not grow the front.

    Scipy-family adapters report the final point through run_post_processing
    a second time; with similarity_tol=0 the near-duplicate check is disabled,
    so exact value matching must reject the duplicate on its own.
    """
    front = ParetoFront(similarity_tol=0, metric_space=objective_space)
    pop = Population(
        X={"x": [1.0]}, metrics={"f": [-2.0]}, metric_space=objective_space
    )
    front.update_population(pop)
    new, _ = front.update_population(pop)
    assert len(front) == 1
    assert len(new) == 0


def test_pareto_front_infeasible_fallback(constrained_space):
    """With only infeasible candidates, the least infeasible ones are kept."""
    front = ParetoFront(similarity_tol=0, metric_space=constrained_space)
    pop = Population(
        X={"x": [1.0, 2.0]},
        metrics={"f": [-1.0, -2.0], "g": [3.0, 2.0]},
        metric_space=constrained_space,
    )
    front.update_population(pop)
    assert len(front) == 1
    np.testing.assert_allclose(front.g, [[2.0]])


def test_pareto_front_merge_deduplicates(objective_space, population):
    front = ParetoFront(similarity_tol=0, metric_space=objective_space)
    front.merge(population)
    front.merge(population)
    assert len(front) == 3


# ── Serialization ─────────────────────────────────────────────────────────────


def test_to_dict_from_dict_roundtrip(mixed_population):
    data = mixed_population.to_dict()
    rebuilt = Population.from_dict(data)
    assert len(rebuilt) == 2
    np.testing.assert_allclose(rebuilt.f, mixed_population.f)
    np.testing.assert_allclose(
        rebuilt.cv_nonlincon, mixed_population.cv_nonlincon
    )
    assert rebuilt.metric_space.objective_names == ["yield"]
    assert list(rebuilt.X["resin"]) == ["A", "B"]


def test_from_dict_with_existing_metric_space(mixed_space, mixed_population):
    rebuilt = Population.from_dict(
        mixed_population.to_dict(), metric_space=mixed_space
    )
    assert rebuilt.metric_space is mixed_space


def test_from_dict_rejects_legacy_format(objective_space):
    with pytest.raises(CADETProcessError, match="predates"):
        Population.from_dict({"individuals": {}, "id": "0"})


def test_pareto_front_to_dict_roundtrip(objective_space):
    front = ParetoFront(similarity_tol=0.5, metric_space=objective_space)
    front.update_population(
        Population(
            X={"x": [1.0]}, metrics={"f": [-2.0]}, metric_space=objective_space
        )
    )
    rebuilt = ParetoFront.from_dict(front.to_dict())
    assert rebuilt.similarity_tol == 0.5
    np.testing.assert_allclose(rebuilt.f, [[-2.0]])


# ── Module-level pairwise plot ───────────────────────────────────────────────


def test_plot_pairwise_all_nonfinite_does_not_raise():
    """plot_pairwise must not crash when all values in a column are non-finite.

    Before the fix, _plot_pairwise_histogram called x.min() on an empty array
    after np.isfinite filtering, raising ValueError.
    """
    import matplotlib
    matplotlib.use("Agg")
    from CADETProcess.optimization.population import plot_pairwise

    data = np.full((4, 2), np.inf)
    plot_pairwise(data, variable_names=["f0", "f1"])
