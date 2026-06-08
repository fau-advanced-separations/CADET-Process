import numpy as np
import pytest
from CADETProcess import CADETProcessError


@pytest.mark.parametrize("simulation_results,purity_required", [
    ("single_component_wide_peak", [0.95]),
    ("single_component_two_peaks", [0.95]),
    ("two_components_separated", [0.95, 0.95]),
    ("two_components_touching", [0.95, 0.95]),
    ("two_components_overlapping", [0.95, 0.95]),
    ("two_components_gaussian_separated", [0.95, 0.95]),
    ("two_components_gaussian_close", [0.80, 0.80]),
    ("three_components_gaussian_separated", [0.95, 0.95, 0.95]),
], ids=[
    "single_component_wide_peak",
    "single_component_two_peaks",
    "two_components_separated",
    "two_components_touching",
    "two_components_overlapping",
    "two_components_gaussian_separated",
    "two_components_gaussian_close",
    "three_components_gaussian_separated",
], indirect=["simulation_results"])
def test_purity_requirement_met(simulation_results, purity_required, optimizer):
    chromatograms, process_meta = simulation_results
    frac = optimizer.optimize_fractionation(
        chromatograms, purity_required, process_meta=process_meta
    )
    purity = frac.performance.purity
    np.testing.assert_array_less(
        np.array(purity_required) - 1e-3,
        purity,
        err_msg=f"Purity {purity} did not meet required {purity_required}",
    )


def test_infeasible_raises(optimizer, two_components_fully_overlapping):
    chromatograms, process_meta = two_components_fully_overlapping
    with pytest.raises(CADETProcessError):
        optimizer.optimize_fractionation(chromatograms, [0.95, 0.95], process_meta=process_meta)


def test_recovery_bounded(optimizer, two_components_separated_mass_matched):
    chromatograms, process_meta = two_components_separated_mass_matched
    frac = optimizer.optimize_fractionation(chromatograms, [0.95, 0.95], process_meta=process_meta)
    recovery = frac.performance.recovery
    assert np.all(recovery >= 0), f"Negative recovery: {recovery}"
    assert np.all(recovery <= 1 + 1e-4), f"Recovery exceeds 1: {recovery}"


def test_recovery_reduced_by_overlap(
    optimizer,
    two_components_separated_mass_matched,
    two_components_overlapping_mass_matched,
):
    chroms_sep, pm_sep = two_components_separated_mass_matched
    chroms_ov, pm_ov = two_components_overlapping_mass_matched
    frac_sep = optimizer.optimize_fractionation(chroms_sep, [0.95, 0.95], process_meta=pm_sep)
    frac_ov = optimizer.optimize_fractionation(chroms_ov, [0.95, 0.95], process_meta=pm_ov)
    recovery_sep = frac_sep.performance.recovery
    recovery_ov = frac_ov.performance.recovery
    assert np.all(recovery_sep >= recovery_ov - 1e-3), (
        f"Separated recovery {recovery_sep} not >= overlapping {recovery_ov}"
    )


@pytest.mark.parametrize("simulation_results,purity_required", [
    ("two_components_separated", [0.95, 0.95]),
    ("two_components_gaussian_separated", [0.95, 0.95]),
], ids=["separated", "gaussian_separated"], indirect=["simulation_results"])
def test_productivity_and_eluent_nonnegative(simulation_results, purity_required, optimizer):
    chromatograms, process_meta = simulation_results
    frac = optimizer.optimize_fractionation(
        chromatograms, purity_required, process_meta=process_meta
    )
    perf = frac.performance
    assert np.all(perf.productivity >= 0), f"Negative productivity: {perf.productivity}"
    assert np.all(perf.eluent_consumption >= 0), (
        f"Negative eluent_consumption: {perf.eluent_consumption}"
    )


def test_multi_outlet_fractionation(optimizer, two_outlets_two_components):
    chromatograms, process_meta = two_outlets_two_components
    frac = optimizer.optimize_fractionation(chromatograms, [0.95, 0.95], process_meta=process_meta)
    purity = frac.performance.purity
    np.testing.assert_array_less(
        np.array([0.95, 0.95]) - 1e-3,
        purity,
        err_msg=f"Multi-outlet purity {purity} did not meet 0.95",
    )
