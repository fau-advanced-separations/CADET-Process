import numpy as np
import pytest
from CADETProcess.metric_space import Metric

# ── Declaration ───────────────────────────────────────────────────────────────


def test_scalar_metric_defaults_to_single_entry():
    m = Metric("yield")
    assert m.n_metrics == 1
    assert m.shape == ()
    assert m.labels == ["yield"]


def test_vector_metric_shape_follows_n_metrics():
    m = Metric("yield", n_metrics=3)
    assert m.shape == (3,)
    assert m.labels == ["yield_0", "yield_1", "yield_2"]


def test_dims_derive_shape_and_coord_labels():
    m = Metric("yield", dims=("component",), coords={"component": ["A", "B"]})
    assert m.shape == (2,)
    assert m.n_metrics == 2
    assert m.labels == ["yield_A", "yield_B"]


def test_multidim_metric_flattens_entry_count():
    m = Metric(
        "conc",
        dims=("component", "stage"),
        coords={"component": ["A", "B"], "stage": [1, 2, 3]},
    )
    assert m.shape == (2, 3)
    assert m.n_metrics == 6
    assert m.labels == [f"conc_{i}" for i in range(6)]


def test_explicit_labels_override_generated():
    m = Metric("yield", n_metrics=2, labels=["light", "heavy"])
    assert m.labels == ["light", "heavy"]


@pytest.mark.parametrize("name", ["", None, 42])
def test_invalid_name_raises(name):
    with pytest.raises(ValueError):
        Metric(name)


def test_zero_entries_raises():
    with pytest.raises(ValueError, match="n_metrics"):
        Metric("yield", n_metrics=0)


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="labels"):
        Metric("yield", n_metrics=2, labels=["only_one"])


def test_dims_without_coords_raises():
    with pytest.raises(ValueError, match="coords"):
        Metric("yield", dims=("component",))


def test_coords_without_dims_raises():
    with pytest.raises(ValueError, match="dims"):
        Metric("yield", coords={"component": ["A"]})


def test_missing_coordinates_for_dimension_raises():
    with pytest.raises(ValueError, match="component"):
        Metric("yield", dims=("component",), coords={"stage": [1]})


def test_n_metrics_contradicting_dims_raises():
    with pytest.raises(ValueError, match="contradicts"):
        Metric("yield", n_metrics=3, dims=("component",), coords={"component": ["A", "B"]})


# ── Shape validation ──────────────────────────────────────────────────────────


def test_scalar_accepts_python_scalar():
    m = Metric("yield")
    assert m.validate(0.5) == 0.5


def test_scalar_canonicalizes_length_one_vector():
    m = Metric("yield")
    assert m.validate([0.5]).shape == ()


def test_vector_accepts_declared_shape():
    m = Metric("yield", n_metrics=2)
    np.testing.assert_array_equal(m.validate([0.1, 0.2]), [0.1, 0.2])


def test_vector_rejects_wrong_length():
    m = Metric("yield", n_metrics=2)
    with pytest.raises(ValueError, match="shape"):
        m.validate([0.1, 0.2, 0.3])


def test_scalar_rejects_vector():
    m = Metric("yield")
    with pytest.raises(ValueError, match="shape"):
        m.validate([0.1, 0.2])


def test_multidim_rejects_flat_vector():
    m = Metric(
        "conc",
        dims=("component", "stage"),
        coords={"component": ["A", "B"], "stage": [1, 2]},
    )
    with pytest.raises(ValueError, match="shape"):
        m.validate([1.0, 2.0, 3.0, 4.0])
    assert m.validate([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
