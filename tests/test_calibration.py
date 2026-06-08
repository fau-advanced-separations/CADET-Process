import numpy as np
import pytest
from CADETProcess.calibration import (
    apply_beer_lambert,
    apply_polynomial_calibration,
    correct_baseline,
    correct_baseline_and_normalize,
    crop,
    deconvolve_extinction,
    fit_baseline,
    fit_polynomial,
    normalize_area,
)
from CADETProcess.reference import ReferenceIO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_reference():
    """Constant signal of 2.0 over [0, 10] s with unit flow rate."""
    time = np.linspace(0, 10, 101)
    solution = np.full((101, 1), 2.0)
    return ReferenceIO("flat", time, solution, flow_rate=1.0)


@pytest.fixture
def drifted_reference():
    """Signal with a linear baseline drift: signal = 1 + 0.1*t (no peak)."""
    time = np.linspace(0, 10, 101)
    solution = (1.0 + 0.1 * time).reshape(-1, 1)
    return ReferenceIO("drifted", time, solution, flow_rate=1.0)


@pytest.fixture
def peak_reference():
    """Gaussian peak on a flat baseline of 0."""
    time = np.linspace(0, 20, 201)
    solution = np.exp(-0.5 * ((time - 10) / 2) ** 2).reshape(-1, 1)
    return ReferenceIO("peak", time, solution, flow_rate=1.0)


# ---------------------------------------------------------------------------
# fit_baseline
# ---------------------------------------------------------------------------

def test_fit_baseline_flat_signal():
    time = np.linspace(0, 10, 11)
    signal = np.ones(11) * 3.0
    baseline = fit_baseline(time, signal)
    np.testing.assert_allclose(baseline, 3.0, atol=1e-10)


def test_fit_baseline_linear_drift():
    time = np.linspace(0, 10, 101)
    signal = 1.0 + 0.5 * time
    baseline = fit_baseline(time, signal)
    # All points are "baseline" for a pure drift; recovered slope ≈ 0.5
    np.testing.assert_allclose(baseline, signal, atol=0.1)


def test_fit_baseline_empty_window_raises():
    time = np.linspace(0, 10, 11)
    signal = np.ones(11)
    with pytest.raises(ValueError, match="No points"):
        fit_baseline(time, signal, start=20.0, end=30.0)


def test_fit_baseline_too_few_points_raises():
    time = np.array([0.0, 1.0, 2.0])
    # threshold=0 → no points selected
    signal = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Fewer than two"):
        fit_baseline(time, signal, threshold=0.0)


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------

def test_crop_returns_new_reference(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    assert result is not flat_reference


def test_crop_name(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    assert "cropped" in result.name


def test_crop_time_is_re_zeroed(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    assert result.time[0] == pytest.approx(0.0)


def test_crop_time_span(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    np.testing.assert_allclose(result.time[-1], 6.0, atol=0.1)


def test_crop_excludes_outside_points(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    assert result.time[0] >= 0.0
    assert result.time[-1] <= 6.1


def test_crop_preserves_signal_values(flat_reference):
    result = crop(flat_reference, start=2.0, end=8.0)
    np.testing.assert_allclose(result.solution, 2.0)


# ---------------------------------------------------------------------------
# correct_baseline
# ---------------------------------------------------------------------------

def test_correct_baseline_returns_new_reference(drifted_reference):
    result = correct_baseline(drifted_reference)
    assert result is not drifted_reference


def test_correct_baseline_name(drifted_reference):
    result = correct_baseline(drifted_reference)
    assert "baseline_corrected" in result.name


def test_correct_baseline_removes_drift(drifted_reference):
    result = correct_baseline(drifted_reference)
    # After subtracting the linear drift the signal should be near-zero
    np.testing.assert_allclose(result.solution, 0.0, atol=0.05)


def test_correct_baseline_multi_component():
    time = np.linspace(0, 10, 101)
    # Two components, each with a different linear drift
    col1 = (1.0 + 0.1 * time).reshape(-1, 1)
    col2 = (2.0 + 0.3 * time).reshape(-1, 1)
    solution = np.hstack([col1, col2])
    ref = ReferenceIO("multi", time, solution, flow_rate=1.0)
    result = correct_baseline(ref)
    np.testing.assert_allclose(result.solution, 0.0, atol=0.05)


def test_correct_baseline_does_not_zero_outside_window(peak_reference):
    result = correct_baseline(peak_reference, start=5.0, end=15.0)
    time = result.time
    outside = (time < 5.0) | (time > 15.0)
    # baseline is subtracted everywhere; outside points are not zeroed
    assert not np.all(result.solution[outside] == 0.0)


# ---------------------------------------------------------------------------
# normalize_area
# ---------------------------------------------------------------------------

def test_normalize_area_returns_new_reference(flat_reference):
    result = normalize_area(flat_reference, target_area=1.0)
    assert result is not flat_reference


def test_normalize_area_name(flat_reference):
    result = normalize_area(flat_reference, target_area=1.0)
    assert "normalized" in result.name


def test_normalize_area_correct_integral(flat_reference):
    target = 5.0
    result = normalize_area(flat_reference, target_area=target)
    actual_area = result.fraction_mass()
    np.testing.assert_allclose(actual_area, target, rtol=1e-6)


def test_normalize_area_with_window(flat_reference):
    # Integrate only over [2, 8]; signal is constant so area = 6 * 2 = 12
    target = 1.0
    result = normalize_area(flat_reference, target_area=target, start=2.0, end=8.0)
    area_in_window = result.fraction_mass(2.0, 8.0)
    np.testing.assert_allclose(area_in_window, target, rtol=1e-5)


# ---------------------------------------------------------------------------
# correct_baseline_and_normalize
# ---------------------------------------------------------------------------

def test_correct_baseline_and_normalize(peak_reference):
    target = 2.0
    result = correct_baseline_and_normalize(peak_reference, target_area=target)
    np.testing.assert_allclose(result.fraction_mass(), target, rtol=1e-4)


# ---------------------------------------------------------------------------
# fit_polynomial
# ---------------------------------------------------------------------------

def test_fit_polynomial_linear():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    coeffs, r2 = fit_polynomial(x, y, degree=1)
    assert r2 == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(np.polyval(coeffs, x), y, atol=1e-8)


def test_fit_polynomial_quadratic():
    x = np.linspace(0, 5, 20)
    y = 3.0 * x**2 - x + 0.5
    coeffs, r2 = fit_polynomial(x, y, degree=2)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_fit_polynomial_returns_list():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    coeffs, _ = fit_polynomial(x, y, degree=2)
    assert isinstance(coeffs, list)


# ---------------------------------------------------------------------------
# apply_polynomial_calibration
# ---------------------------------------------------------------------------

def test_apply_polynomial_calibration_returns_new_reference(flat_reference):
    result = apply_polynomial_calibration(flat_reference, [1.0, 0.0])
    assert result is not flat_reference


def test_apply_polynomial_calibration_name(flat_reference):
    result = apply_polynomial_calibration(flat_reference, [1.0, 0.0])
    assert "calibrated" in result.name


def test_apply_polynomial_calibration_linear(flat_reference):
    # y = 3*x + 1: signal is 2.0 everywhere → calibrated = 3*2 + 1 = 7
    result = apply_polynomial_calibration(flat_reference, [3.0, 1.0])
    np.testing.assert_allclose(result.solution, 7.0)


def test_apply_polynomial_calibration_roundtrip(peak_reference):
    """Fit a calibration curve to known data then apply it and recover."""
    x = np.linspace(0, 1, 10)
    y = 5.0 * x + 0.5
    coeffs, _ = fit_polynomial(x, y, degree=1)
    ref = ReferenceIO("test", peak_reference.time, peak_reference.solution * 0.5, 1.0)
    result = apply_polynomial_calibration(ref, coeffs)
    expected = np.polyval(coeffs, ref.solution)
    np.testing.assert_allclose(result.solution, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# apply_beer_lambert
# ---------------------------------------------------------------------------

def test_apply_beer_lambert_returns_new_reference(flat_reference):
    result = apply_beer_lambert(flat_reference, extinction_coefficient=1000.0, path_length=1.0)
    assert result is not flat_reference


def test_apply_beer_lambert_name(flat_reference):
    result = apply_beer_lambert(flat_reference, extinction_coefficient=1000.0, path_length=1.0)
    assert "concentration" in result.name


def test_apply_beer_lambert_scalar():
    time = np.linspace(0, 5, 51)
    # Absorbance = 0.5 AU, ε = 1000 L/(mol·cm), l = 1 cm → c = 5e-4 mol/L
    solution = np.full((51, 1), 0.5)
    ref = ReferenceIO("uv", time, solution, flow_rate=1.0)
    result = apply_beer_lambert(ref, extinction_coefficient=1000.0, path_length=1.0)
    np.testing.assert_allclose(result.solution, 5e-4)


def test_apply_beer_lambert_path_length_scaling():
    time = np.linspace(0, 5, 51)
    solution = np.full((51, 1), 1.0)
    ref = ReferenceIO("uv", time, solution, flow_rate=1.0)
    r1 = apply_beer_lambert(ref, extinction_coefficient=500.0, path_length=2.0)
    r2 = apply_beer_lambert(ref, extinction_coefficient=1000.0, path_length=1.0)
    np.testing.assert_allclose(r1.solution, r2.solution)


# ---------------------------------------------------------------------------
# deconvolve_extinction
# ---------------------------------------------------------------------------

@pytest.fixture
def two_component_references():
    """Two synthetic UV channels from a known two-component mixture."""
    time = np.linspace(0, 10, 101)
    # True concentrations
    c1 = np.exp(-0.5 * ((time - 3) / 1) ** 2)
    c2 = np.exp(-0.5 * ((time - 7) / 1) ** 2)
    C = np.stack([c1, c2], axis=1)  # (101, 2)
    # Extinction matrix: ε at 280 nm and 260 nm for two components
    E = np.array([[1000.0, 200.0],
                  [300.0, 800.0]])
    path_length = 1.0
    A = C @ E.T * path_length  # (101, 2)
    ref280 = ReferenceIO("A280", time, A[:, 0:1], flow_rate=1.0)
    ref260 = ReferenceIO("A260", time, A[:, 1:2], flow_rate=1.0)
    return [ref280, ref260], E, path_length, C


def test_deconvolve_extinction_recovers_concentrations(two_component_references):
    refs, E, pl, C_true = two_component_references
    results = deconvolve_extinction(refs, E, pl)
    assert len(results) == 2
    C_recovered = np.hstack([r.solution for r in results])
    np.testing.assert_allclose(C_recovered, C_true, atol=1e-8)


def test_deconvolve_extinction_component_names(two_component_references):
    refs, E, pl, _ = two_component_references
    results = deconvolve_extinction(refs, E, pl, component_names=["protein", "salt"])
    assert results[0].name == "protein"
    assert results[1].name == "salt"


def test_deconvolve_extinction_default_names(two_component_references):
    refs, E, pl, _ = two_component_references
    results = deconvolve_extinction(refs, E, pl)
    assert results[0].name == "component_0"
    assert results[1].name == "component_1"


def test_deconvolve_extinction_wrong_n_references_raises(two_component_references):
    refs, E, pl, _ = two_component_references
    with pytest.raises(ValueError, match="extinction_matrix has 2 rows"):
        deconvolve_extinction(refs[:1], E, pl)


def test_deconvolve_extinction_mismatched_time_raises(two_component_references):
    refs, E, pl, _ = two_component_references
    bad = ReferenceIO("bad", np.linspace(0, 5, 101), refs[1].solution, 1.0)
    with pytest.raises(ValueError, match="same time axis"):
        deconvolve_extinction([refs[0], bad], E, pl)


def test_deconvolve_extinction_overdetermined(two_component_references):
    """Three wavelengths for two components — least-squares should still recover C."""
    refs, E2, pl, C_true = two_component_references
    time = refs[0].time
    E3 = np.vstack([E2, [600.0, 500.0]])
    A_extra = (C_true @ E3.T * pl)[:, 2:3]
    ref_extra = ReferenceIO("A320", time, A_extra, flow_rate=1.0)
    results = deconvolve_extinction(refs + [ref_extra], E3, pl)
    C_recovered = np.hstack([r.solution for r in results])
    np.testing.assert_allclose(C_recovered, C_true, atol=1e-8)
