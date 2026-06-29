import numpy as np
import pytest

from CADETProcess.processModel import (
    ComponentSystem,
    FlowSheet,
    GeneralRateModel,
    Inlet,
    LumpedRateModelWithPores,
    Outlet,
    Process,
    StericMassAction,
)
from CADETProcess.simulator import Cadet
from CADETProcess.tools.yamamoto import GradientExperiment, fit_parameters

from tests.simulator.test_cadet_adapter import found_cadet


# ---------------------------------------------------------------------------
# Rüdt et al. 2015, J. Chromatogr. A 1413:68-76, Table 4.
# Single protein (mAb1), 8×200 mm LRMP column, N≈2440 plates.
# Capacity convention differs from CADET-Process (per total column volume vs
# per solid-phase volume): Λ_CADET = Λ_paper/(1−ε_t),
#                           K_CADET = K_paper·(1−ε_t)^(ν−1)
# ---------------------------------------------------------------------------

_PAPER_NU = 9.17
_PAPER_EPS_T = 0.542          # total porosity accessible to mAb1 (Table 4)
_PAPER_LAMBDA_COL = 100.0     # mM, per total column volume (paper convention)
_PAPER_K_COL = 10**1.54       # K from log K=1.54 (paper convention)
_PAPER_C_SALT_INITIAL = 5.0   # mM  (equilibration buffer, no added NaCl)
_PAPER_C_SALT_FINAL = 300.0   # mM  (elution buffer)
_PAPER_D_AX = 1.20e-4         # m   dispersivity (Table 4; D_ax = d_ax · u_sf)
_PAPER_CV_LENGTHS = [10, 25, 45]

_PAPER_CAPACITY = _PAPER_LAMBDA_COL / (1 - _PAPER_EPS_T)
_PAPER_K_CADET = _PAPER_K_COL * (1 - _PAPER_EPS_T) ** (_PAPER_NU - 1)


def _create_paper_column(component_system):
    binding_model = StericMassAction(component_system, name="SMA")
    binding_model.is_kinetic = False
    binding_model.characteristic_charge = [0.0, _PAPER_NU]
    binding_model.capacity = _PAPER_CAPACITY
    binding_model.steric_factor = [0.0, 10]
    # reference_solid/liquid_phase_conc left at default 1.0, so adsorption_rate
    # equals the physical equilibrium constant directly.
    binding_model.adsorption_rate = [0.0, _PAPER_K_CADET]
    binding_model.desorption_rate = [0.0, 1.0]

    bed_porosity = 0.34
    particle_porosity = (_PAPER_EPS_T - bed_porosity) / (1 - bed_porosity)

    column = LumpedRateModelWithPores(component_system, name="column")
    column.length = 0.2
    column.diameter = 0.008
    column.particle_radius = 1.7e-5
    column.bed_porosity = bed_porosity
    column.particle_porosity = particle_porosity
    column.film_diffusion = [1e-4, 1e-4]
    column.binding_model = binding_model
    column.c = [_PAPER_C_SALT_INITIAL, 0]
    column.cp = [_PAPER_C_SALT_INITIAL, 0]
    column.q = [binding_model.capacity, 0]
    return column


def _create_paper_process(component_system, column, cv_length):
    column_volume = column.length * (column.diameter / 2) ** 2 * np.pi
    rt_s = 6.0 * 60
    Q_m3_s = column_volume / rt_s
    column.axial_dispersion = _PAPER_D_AX * (Q_m3_s / ((column.diameter / 2) ** 2 * np.pi))

    inlet = Inlet(component_system, name="inlet")
    inlet.flow_rate = Q_m3_s
    outlet = Outlet(component_system, name="outlet")

    flow_sheet = FlowSheet(component_system)
    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)
    flow_sheet.add_connection(inlet, column)
    flow_sheet.add_connection(column, outlet)

    process = Process(flow_sheet, f"{cv_length}")

    load_duration = 9
    t_gradient_start = 3 * 60.0
    gradient_volume = cv_length * column_volume
    gradient_duration = gradient_volume / inlet.flow_rate[0]
    duration_post_gradient_wash = gradient_duration / 10 + 180
    process.cycle_time = t_gradient_start + gradient_duration + duration_post_gradient_wash
    t_gradient_end = t_gradient_start + gradient_duration

    c_load = np.array([_PAPER_C_SALT_INITIAL, 0.01])
    c_wash = np.array([_PAPER_C_SALT_INITIAL, 0.0])
    c_elute = np.array([_PAPER_C_SALT_FINAL, 0.0])
    gradient_slope = (c_elute - c_wash) / gradient_duration
    c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))
    c_post_gradient_wash = np.array([_PAPER_C_SALT_FINAL, 0.0])

    process.add_event("load", "flow_sheet.inlet.c", c_load)
    process.add_event("wash", "flow_sheet.inlet.c", c_wash, load_duration)
    process.add_event("grad_start", "flow_sheet.inlet.c", c_gradient_poly, t_gradient_start)
    process.add_event("grad_end", "flow_sheet.inlet.c", c_post_gradient_wash, t_gradient_end)

    return process


@pytest.mark.skipif(found_cadet is False, reason="Skip if CADET is not installed.")
def test_fit_parameters_paper_case():
    """Single-protein fit using Rüdt et al. 2015 (JCA 1413:68-76) parameters.

    High plate count (N≈2440) means Yamamoto's local-equilibrium assumption
    holds well and both ν and k_eq are recovered within 5 %.
    """
    np.random.seed(0)

    component_system = ComponentSystem(["Salt", "Protein"])
    column = _create_paper_column(component_system)
    column_volume = column.length * (column.diameter / 2) ** 2 * np.pi

    simulator = Cadet()
    experiments = []
    for cv in _PAPER_CV_LENGTHS:
        process = _create_paper_process(component_system, column, cv)
        results = simulator.simulate(process)
        outlet = results.solution["outlet"]["outlet"]
        solution = outlet.solution.copy()

        solution_without_salt = solution[:, 1:]
        noise = (
            (np.random.random(solution_without_salt.shape) - 0.5)
            * 5
            / 100
            * solution_without_salt.max(axis=0)
        )
        solution[:, 1:] = solution_without_salt + noise

        experiments.append(
            GradientExperiment(outlet.time, solution[:, 0], solution[:, 1], cv * column_volume)
        )

    yamamoto_results = fit_parameters(experiments, column)

    print(
        f"characteristic_charge: fitted={yamamoto_results.characteristic_charge}, "
        f"expected={[_PAPER_NU]}"
    )
    print(f"k_eq: fitted={yamamoto_results.k_eq}, expected={[_PAPER_K_CADET]}")

    np.testing.assert_allclose(
        yamamoto_results.characteristic_charge, [_PAPER_NU], rtol=0.05
    )
    np.testing.assert_allclose(yamamoto_results.k_eq, [_PAPER_K_CADET], rtol=0.05)


# ---------------------------------------------------------------------------
# Püttmann et al. 2013, Comput. Chem. Eng. 56:46-57, Tables 2+3 (benchmark 2).
# Three proteins on SP Sepharose FF, GeneralRateModel, 14 mm column, N≈70.
# Gradient starts at the equilibration concentration (50 mM) — a step-up to
# 100 mM before the ramp (as in the paper's inlet profile) would desorb
# Ribonuclease prematurely and break the Yamamoto fit for that protein.
# ---------------------------------------------------------------------------

_P2_LAMBDA  = 1200.0   # mM, ionic capacity per solid-phase volume (Table 3)
_P2_NU      = [0.0, 4.70, 5.29, 3.70]
_P2_SIGMA   = [0.0, 11.83, 10.60, 10.00]
_P2_KEQV    = [0.0, 3.55e-2, 1.59e-3, 7.70e-3]
_P2_L       = 1.4e-2   # m
_P2_D       = 1.0e-2   # m  (not specified in paper; 1 cm chosen)
_P2_RP      = 4.5e-5   # m
_P2_EPS_C   = 0.37
_P2_EPS_P   = 0.75
_P2_U       = 5.75e-4  # m/s  interstitial velocity
_P2_DAX     = 5.75e-8  # m2/s
_P2_KF      = 6.90e-6  # m/s  film mass transfer
_P2_DP_PROT = 6.07e-11 # m2/s pore diffusion (proteins)
_P2_DP_SALT = 7.00e-10 # m2/s pore diffusion (Na+)
_P2_C_EQUIL = 50.0     # mM
_P2_C_END   = 350.0    # mM   (gradient start = _P2_C_EQUIL, end here)
_P2_CV      = [10, 25, 45]
_P2_Q_M3S   = _P2_U * _P2_EPS_C * np.pi * (_P2_D / 2) ** 2
_P2_COL_VOL = np.pi * (_P2_D / 2) ** 2 * _P2_L


def _create_p2_column(component_system):
    bm = StericMassAction(component_system, name="SMA")
    bm.is_kinetic = False
    bm.capacity = _P2_LAMBDA
    bm.characteristic_charge = _P2_NU
    bm.steric_factor = _P2_SIGMA
    bm.adsorption_rate = _P2_KEQV
    bm.desorption_rate = [1.0] * component_system.n_comp

    col = GeneralRateModel(component_system, name="column")
    col.length = _P2_L
    col.diameter = _P2_D
    col.particle_radius = _P2_RP
    col.bed_porosity = _P2_EPS_C
    col.particle_porosity = _P2_EPS_P
    col.axial_dispersion = _P2_DAX
    col.film_diffusion = [_P2_KF] * component_system.n_comp
    col.pore_diffusion = [_P2_DP_SALT, _P2_DP_PROT, _P2_DP_PROT, _P2_DP_PROT]
    col.binding_model = bm
    col.c  = [_P2_C_EQUIL, 0, 0, 0]
    col.cp = [_P2_C_EQUIL, 0, 0, 0]
    col.q  = [_P2_LAMBDA, 0, 0, 0]
    return col


def _create_p2_process(component_system, column, cv_length):
    inlet  = Inlet(component_system, name="inlet")
    inlet.flow_rate = _P2_Q_M3S
    outlet = Outlet(component_system, name="outlet")

    fs = FlowSheet(component_system)
    fs.add_unit(inlet)
    fs.add_unit(column)
    fs.add_unit(outlet)
    fs.add_connection(inlet, column)
    fs.add_connection(column, outlet)

    process = Process(fs, f"p2_cv{cv_length}")

    load_duration = 10.0
    t_wash_end    = 90.0
    grad_vol      = cv_length * _P2_COL_VOL
    grad_dur      = grad_vol / _P2_Q_M3S
    post_dur      = grad_dur / 5 + 300
    process.cycle_time = t_wash_end + grad_dur + post_dur
    t_grad_end    = t_wash_end + grad_dur

    slope  = (_P2_C_END - _P2_C_EQUIL) / grad_dur
    c_load = np.array([_P2_C_EQUIL, 1.0, 1.0, 1.0])
    c_wash = np.array([_P2_C_EQUIL, 0.0, 0.0, 0.0])
    c_ramp = np.array([[_P2_C_EQUIL, slope], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    c_post = np.array([_P2_C_END, 0.0, 0.0, 0.0])

    process.add_event("load",       "flow_sheet.inlet.c", c_load)
    process.add_event("wash",       "flow_sheet.inlet.c", c_wash, load_duration)
    process.add_event("grad_start", "flow_sheet.inlet.c", c_ramp, t_wash_end)
    process.add_event("grad_end",   "flow_sheet.inlet.c", c_post, t_grad_end)
    return process


@pytest.mark.skipif(found_cadet is False, reason="Skip if CADET is not installed.")
def test_fit_parameters_puttmann_case():
    """Three-protein GRM fit using Püttmann et al. 2013 (CChE 56:46-57) parameters.

    Realistic 14 mm GRM column (N≈70): ν within ~1 %, k_eq within ~25 %
    (inherent plate-count limit, not reducible by adding more experiments).
    """
    component_system = ComponentSystem(
        ["Salt", "Lysozyme", "Cytochrome", "Ribonuclease"]
    )
    simulator = Cadet()
    experiments = []
    for cv in _P2_CV:
        column  = _create_p2_column(component_system)
        process = _create_p2_process(component_system, column, cv)
        results = simulator.simulate(process)
        outlet  = results.solution["outlet"]["outlet"]
        sol     = outlet.solution
        experiments.append(
            GradientExperiment(
                outlet.time, sol[:, 0], sol[:, 1:],
                cv * _P2_COL_VOL,
                c_salt_start=_P2_C_EQUIL,
                c_salt_end=_P2_C_END,
            )
        )

    column_for_fit = _create_p2_column(component_system)
    yamamoto_results = fit_parameters(experiments, column_for_fit)

    nu_true  = np.array(_P2_NU[1:])
    keq_true = np.array(_P2_KEQV[1:])

    print(
        f"characteristic_charge: fitted={yamamoto_results.characteristic_charge}, "
        f"expected={nu_true.tolist()}"
    )
    print(f"k_eq: fitted={yamamoto_results.k_eq}, expected={keq_true.tolist()}")

    np.testing.assert_allclose(
        yamamoto_results.characteristic_charge, nu_true, rtol=0.05
    )
    np.testing.assert_allclose(yamamoto_results.k_eq, keq_true, rtol=0.30)


# ---------------------------------------------------------------------------
# Same Püttmann et al. 2013 binding parameters, but on a 200 mm LRMP column
# (N≈1000). LRMP is used because pore diffusion in the GRM caps the effective
# plate count at N≈70 regardless of D_ax, whereas LRMP reaches N≈1000 at
# L=200 mm with the same D_ax.
# ---------------------------------------------------------------------------

_P2_NEAR_IDEAL_L = 0.2   # m — 10× longer than the GRM benchmark column


def _create_p2_near_ideal_column(component_system):
    bm = StericMassAction(component_system, name="SMA")
    bm.is_kinetic = False
    bm.capacity = _P2_LAMBDA
    bm.characteristic_charge = _P2_NU
    bm.steric_factor = _P2_SIGMA
    bm.adsorption_rate = _P2_KEQV
    bm.desorption_rate = [1.0] * component_system.n_comp

    col = LumpedRateModelWithPores(component_system, name="column")
    col.length = _P2_NEAR_IDEAL_L
    col.diameter = _P2_D
    col.particle_radius = _P2_RP
    col.bed_porosity = _P2_EPS_C
    col.particle_porosity = _P2_EPS_P
    col.axial_dispersion = _P2_DAX   # same as GRM benchmark (not reduced further)
    col.film_diffusion = [_P2_KF] * component_system.n_comp
    col.binding_model = bm
    col.c  = [_P2_C_EQUIL, 0, 0, 0]
    col.cp = [_P2_C_EQUIL, 0, 0, 0]
    col.q  = [_P2_LAMBDA,  0, 0, 0]
    return col


def _create_p2_near_ideal_process(component_system, column, cv_length):
    column_volume = column.length * (column.diameter / 2) ** 2 * np.pi
    inlet  = Inlet(component_system, name="inlet")
    inlet.flow_rate = _P2_Q_M3S
    outlet = Outlet(component_system, name="outlet")

    fs = FlowSheet(component_system)
    fs.add_unit(inlet)
    fs.add_unit(column)
    fs.add_unit(outlet)
    fs.add_connection(inlet, column)
    fs.add_connection(column, outlet)

    process = Process(fs, f"ni_cv{cv_length}")

    load_duration = 10.0
    t_wash_end    = 90.0
    grad_vol      = cv_length * column_volume
    grad_dur      = grad_vol / _P2_Q_M3S
    post_dur      = grad_dur / 5 + 300
    process.cycle_time = t_wash_end + grad_dur + post_dur
    t_grad_end    = t_wash_end + grad_dur

    slope  = (_P2_C_END - _P2_C_EQUIL) / grad_dur
    c_load = np.array([_P2_C_EQUIL, 1.0, 1.0, 1.0])
    c_wash = np.array([_P2_C_EQUIL, 0.0, 0.0, 0.0])
    c_ramp = np.array([[_P2_C_EQUIL, slope], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    c_post = np.array([_P2_C_END, 0.0, 0.0, 0.0])

    process.add_event("load",       "flow_sheet.inlet.c", c_load)
    process.add_event("wash",       "flow_sheet.inlet.c", c_wash, load_duration)
    process.add_event("grad_start", "flow_sheet.inlet.c", c_ramp, t_wash_end)
    process.add_event("grad_end",   "flow_sheet.inlet.c", c_post, t_grad_end)
    return process


@pytest.mark.skipif(found_cadet is False, reason="Skip if CADET is not installed.")
def test_fit_parameters_puttmann_near_ideal():
    """Three-protein LRMP fit using Püttmann et al. 2013 parameters, 200 mm column.

    Higher plate count (N≈1000) gives tighter accuracy: ν within ~0.5 %,
    k_eq within ~8 % for all three proteins.
    """
    component_system = ComponentSystem(
        ["Salt", "Lysozyme", "Cytochrome", "Ribonuclease"]
    )
    simulator = Cadet()
    experiments = []
    for cv in _P2_CV:
        column  = _create_p2_near_ideal_column(component_system)
        process = _create_p2_near_ideal_process(component_system, column, cv)
        results = simulator.simulate(process)
        outlet  = results.solution["outlet"]["outlet"]
        sol     = outlet.solution
        col_vol = column.length * (column.diameter / 2) ** 2 * np.pi
        experiments.append(
            GradientExperiment(
                outlet.time, sol[:, 0], sol[:, 1:],
                cv * col_vol,
                c_salt_start=_P2_C_EQUIL,
                c_salt_end=_P2_C_END,
            )
        )

    column_for_fit = _create_p2_near_ideal_column(component_system)
    yamamoto_results = fit_parameters(experiments, column_for_fit)

    nu_true  = np.array(_P2_NU[1:])
    keq_true = np.array(_P2_KEQV[1:])

    print(
        f"characteristic_charge: fitted={yamamoto_results.characteristic_charge}, "
        f"expected={nu_true.tolist()}"
    )
    print(f"k_eq: fitted={yamamoto_results.k_eq}, expected={keq_true.tolist()}")

    np.testing.assert_allclose(
        yamamoto_results.characteristic_charge, nu_true, rtol=0.02
    )
    np.testing.assert_allclose(yamamoto_results.k_eq, keq_true, rtol=0.08)
