import copy

import numpy as np
import pytest
from CADETProcess.characterization import (
    CharacterizeAdsorptionParameters,
    CharacterizeBase,
    CharacterizeBed,
    CharacterizeCapacity,
    CharacterizeParticles,
    CharacterizePreInjection,
    CharacterizeTubing,
    setup_comparators,
)
from CADETProcess.comparison import Comparator
from CADETProcess.instruments import LWE, LCFlowSheet, PulseInjection
from CADETProcess.processModel import (
    ComponentSystem,
    GeneralRateModel,
    Linear,
    LumpedRateModelWithPores,
    StericMassAction,
)
from CADETProcess.reference import ReferenceIO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FLOW_RATE = 8.3e-9   # m³/s
LOOP_VOLUME = 50e-9  # m³


@pytest.fixture
def salt_cs():
    return ComponentSystem(["Salt"])


@pytest.fixture
def two_component_cs():
    return ComponentSystem(["Salt", "Protein"])


@pytest.fixture
def column_fs(salt_cs):
    return LCFlowSheet(
        salt_cs,
        sample_loop_volume=LOOP_VOLUME,
        ColumnModel=LumpedRateModelWithPores,
        BindingModel=Linear,
    )


@pytest.fixture
def sma_fs(two_component_cs):
    return LCFlowSheet(
        two_component_cs,
        sample_loop_volume=LOOP_VOLUME,
        ColumnModel=LumpedRateModelWithPores,
        BindingModel=StericMassAction,
    )


@pytest.fixture
def pulse_process(column_fs, salt_cs):
    return PulseInjection(
        "pulse",
        column_fs,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )


@pytest.fixture
def lwe_process(sma_fs, two_component_cs):
    return LWE(
        "lwe",
        sma_fs,
        c_buffer_a=[20.0, 0.0],
        c_buffer_b=[1000.0, 0.0],
        c_sample=[20.0, 0.5],
        delta_t_wash=120.0,
        delta_t_elute=600.0,
        delta_t_final_wash=120.0,
        flow_rate_wash=FLOW_RATE,
    )


@pytest.fixture
def flat_reference(salt_cs):
    time = np.linspace(0, 600, 601)
    solution = np.ones((601, 1))
    ref = ReferenceIO("flat", time, solution, flow_rate=FLOW_RATE)
    ref.component_system = salt_cs
    return ref


@pytest.fixture
def comparator(flat_reference):
    comp = Comparator("pulse")
    comp.add_reference(flat_reference)
    comp.add_difference_metric("NRMSE", flat_reference, "column.outlet.outlet[0]")
    return comp


@pytest.fixture
def mock_simulator():
    """Minimal callable that satisfies add_evaluator."""
    def sim(process):
        pass
    sim.__str__ = lambda self: "mock_simulator"
    return sim


# ---------------------------------------------------------------------------
# setup_comparators
# ---------------------------------------------------------------------------


def test_setup_comparators_returns_list(pulse_process, flat_reference):
    result = setup_comparators(
        pulse_process, flat_reference, "column.outlet.outlet[0]", ["NRMSE"]
    )
    assert isinstance(result, list)
    assert len(result) == 1


def test_setup_comparators_comparator_name(pulse_process, flat_reference):
    result = setup_comparators(
        pulse_process, flat_reference, "column.outlet.outlet[0]", ["NRMSE"]
    )
    assert result[0].name == pulse_process.name


def test_setup_comparators_n_metrics(pulse_process, flat_reference):
    result = setup_comparators(
        pulse_process, flat_reference, "column.outlet.outlet[0]", ["NRMSE"]
    )
    assert result[0].n_metrics == 1


def test_setup_comparators_multi_process(pulse_process, flat_reference):
    p2 = PulseInjection(
        "pulse2",
        pulse_process.flow_sheet,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )
    result = setup_comparators(
        [pulse_process, p2],
        [flat_reference, flat_reference],
        "column.outlet.outlet[0]",
        ["NRMSE"],
    )
    assert len(result) == 2
    assert result[0].name == "pulse"
    assert result[1].name == "pulse2"


def test_setup_comparators_per_process_start_end(pulse_process, flat_reference):
    p2 = PulseInjection(
        "pulse2",
        pulse_process.flow_sheet,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )
    # Should not raise even with per-process windows
    result = setup_comparators(
        [pulse_process, p2],
        [flat_reference, flat_reference],
        "column.outlet.outlet[0]",
        ["NRMSE"],
        start=[100.0, 200.0],
        end=[500.0, 550.0],
    )
    assert len(result) == 2


def test_setup_comparators_wrong_reference_count_raises(pulse_process, flat_reference):
    with pytest.raises(ValueError, match="references"):
        setup_comparators(
            [pulse_process],
            [flat_reference, flat_reference],
            "column.outlet.outlet[0]",
            ["NRMSE"],
        )


def test_setup_comparators_wrong_start_list_length_raises(pulse_process, flat_reference):
    p2 = PulseInjection(
        "pulse2",
        pulse_process.flow_sheet,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )
    with pytest.raises(ValueError, match="start"):
        setup_comparators(
            [pulse_process, p2],
            [flat_reference, flat_reference],
            "column.outlet.outlet[0]",
            ["NRMSE"],
            start=[100.0],
        )


# ---------------------------------------------------------------------------
# CharacterizeBase
# ---------------------------------------------------------------------------


def test_characterize_base_no_default_variables(pulse_process, comparator, mock_simulator):
    prob = CharacterizeBase("test", pulse_process, comparator, mock_simulator)
    assert prob.variable_names == []


def test_characterize_bed_per_variable_override(pulse_process, comparator, mock_simulator):
    prob = CharacterizeBed(
        "bed", pulse_process, comparator, mock_simulator,
        bed_porosity={"lb": 0.35, "ub": 0.45},
    )
    bp = prob.variables_dict["bed_porosity"]
    assert bp.lb == pytest.approx(0.35)
    assert bp.ub == pytest.approx(0.45)


def test_characterize_base_single_comparator_coercion(
    pulse_process, comparator, mock_simulator
):
    """A bare Comparator is coerced to a one-element list."""
    prob = CharacterizeBase(
        "test", pulse_process, comparator, mock_simulator
    )
    assert prob.n_objectives == comparator.n_metrics


def test_characterize_base_mismatched_comparators_raises(
    pulse_process, comparator, mock_simulator
):
    with pytest.raises(ValueError, match="comparators"):
        CharacterizeBase(
            "test",
            pulse_process,
            [comparator, comparator],
            mock_simulator,
        )


def test_characterize_base_multi_process_registers_unique_objectives(
    pulse_process, comparator, mock_simulator
):
    """Two processes must not collide on the shared 'Comparator' metric name."""
    p2 = PulseInjection(
        "pulse2",
        pulse_process.flow_sheet,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )
    prob = CharacterizeBase(
        "test",
        [pulse_process, p2],
        [comparator, comparator],
        mock_simulator,
    )
    assert prob.n_objectives == 2 * comparator.n_metrics


@pytest.mark.parametrize(
    "process_name, expected_objective_name",
    [
        ("ok_name", "ok_name"),
        ("pulse 30 CV", "pulse_30_CV"),
        ("pulse-30", "pulse_30"),
        ("5", "_5"),
    ],
)
def test_characterize_base_sanitizes_process_name_for_objective(
    column_fs, comparator, mock_simulator, process_name, expected_objective_name
):
    """Process names are free-form, but objective names must be identifiers."""
    process = PulseInjection(
        process_name,
        column_fs,
        c_buffer_a=[100.0],
        c_sample=[0.0],
        cycle_time=600.0,
        flow_rate=FLOW_RATE,
    )
    prob = CharacterizeBase("test", process, comparator, mock_simulator)

    assert prob.objective_names == [expected_objective_name]


def test_characterize_base_names_unnamed_comparator_after_its_process(
    pulse_process, flat_reference, mock_simulator
):
    comp = Comparator()
    comp.add_reference(flat_reference)
    comp.add_difference_metric("NRMSE", flat_reference, "column.outlet.outlet[0]")

    CharacterizeBase("test", pulse_process, comp, mock_simulator)

    assert comp.name == pulse_process.name


def test_characterize_base_callback_writes_plot_file(
    pulse_process, comparator, mock_simulator, tmp_path, caplog
):
    """Regression test: the registered callback used to reference
    ``individual.id``, which doesn't exist on ``IndividualView``.  The
    AttributeError was swallowed by evaluate_callbacks into a logged
    warning, so no comparison plot was ever written and nobody noticed.

    Uses CharacterizeBed (rather than a bare CharacterizeBase) because a
    zero-variable problem can't build a one-row population: with no
    parameter columns, Population has nothing to infer its row count from.
    """
    prob = CharacterizeBed("test", pulse_process, comparator, mock_simulator)

    calls = []
    comparator.plot_comparison = lambda *args, **kwargs: calls.append(kwargs)

    pop = prob.create_population([[0.4, 1e-7]])
    prob.evaluate_callbacks(pop, current_iteration=0, callbacks_dir=tmp_path)

    assert "failed" not in caplog.text
    assert len(calls) == 1
    file_name = calls[0]["file_name"]
    assert file_name.endswith("_pulse_comparison.png")
    assert pop[0].id_short in file_name


# ---------------------------------------------------------------------------
# CharacterizeTubing
# ---------------------------------------------------------------------------


def test_characterize_tubing_variables(pulse_process, comparator, mock_simulator):
    prob = CharacterizeTubing(
        "tubing", pulse_process, "tubing_pre_injection", comparator, mock_simulator
    )
    assert "tubing_pre_injection_length" in prob.variable_names
    assert "tubing_pre_injection_axial_dispersion" in prob.variable_names


def test_characterize_tubing_bounds(pulse_process, comparator, mock_simulator):
    prob = CharacterizeTubing(
        "tubing", pulse_process, "tubing_pre_injection", comparator, mock_simulator
    )
    var = prob.variables_dict["tubing_pre_injection_length"]
    assert var.lb == pytest.approx(1e-2)
    assert var.ub == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# CharacterizePreInjection
# ---------------------------------------------------------------------------


def test_characterize_pre_injection_variables(pulse_process, comparator, mock_simulator):
    prob = CharacterizePreInjection(
        "pre_injection", pulse_process, comparator, mock_simulator
    )
    assert "tubing_pre_injection_length" in prob.variable_names
    assert "mixer_volume" in prob.variable_names


# ---------------------------------------------------------------------------
# CharacterizeBed
# ---------------------------------------------------------------------------


def test_characterize_bed_variables(pulse_process, comparator, mock_simulator):
    prob = CharacterizeBed("bed", pulse_process, comparator, mock_simulator)
    assert "bed_porosity" in prob.variable_names
    assert "axial_dispersion" in prob.variable_names


def test_characterize_bed_bounds(pulse_process, comparator, mock_simulator):
    prob = CharacterizeBed("bed", pulse_process, comparator, mock_simulator)
    bp = prob.variables_dict["bed_porosity"]
    assert bp.lb == pytest.approx(0.2)
    assert bp.ub == pytest.approx(0.6)


def test_characterize_bed_excludes_particle_porosity(
    pulse_process, comparator, mock_simulator
):
    prob = CharacterizeBed("bed", pulse_process, comparator, mock_simulator)
    assert "particle_porosity" not in prob.variable_names


def test_characterize_bed_particle_porosity(pulse_process, comparator, mock_simulator):
    prob = CharacterizeBed(
        "bed", pulse_process, comparator, mock_simulator,
        include_particle_porosity=True,
    )
    assert "particle_porosity" in prob.variable_names


def test_characterize_bed_particle_porosity_bounds(
    pulse_process, comparator, mock_simulator
):
    prob = CharacterizeBed(
        "bed", pulse_process, comparator, mock_simulator,
        include_particle_porosity=True,
    )
    pp = prob.variables_dict["particle_porosity"]
    assert pp.lb == pytest.approx(0.6)
    assert pp.ub == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# CharacterizeParticles
# ---------------------------------------------------------------------------


def test_characterize_particles_no_flags_raises(
    pulse_process, comparator, mock_simulator
):
    with pytest.raises(ValueError, match="at least one"):
        CharacterizeParticles(
            "p", pulse_process, comparator, mock_simulator, components=["Salt"]
        )


def test_characterize_particles_film_diffusion(
    pulse_process, comparator, mock_simulator
):
    prob = CharacterizeParticles(
        "p",
        pulse_process,
        comparator,
        mock_simulator,
        components=["Salt"],
        include_film_diffusion=True,
    )
    assert "Salt_film_diffusion" in prob.variable_names


def test_characterize_particles_lrmp_flags(pulse_process, comparator, mock_simulator):
    """LRMP supports axial_dispersion, particle_porosity, film_diffusion (not pore_diffusion)."""
    prob = CharacterizeParticles(
        "p",
        pulse_process,
        comparator,
        mock_simulator,
        components=["Salt"],
        include_axial_dispersion=True,
        include_particle_porosity=True,
        include_film_diffusion=True,
    )
    assert "Salt_axial_dispersion" in prob.variable_names
    assert "particle_porosity" in prob.variable_names
    assert "Salt_film_diffusion" in prob.variable_names


# ---------------------------------------------------------------------------
# CharacterizeCapacity
# ---------------------------------------------------------------------------


def test_characterize_capacity_variable(lwe_process, lwe_comparator, mock_simulator):
    prob = CharacterizeCapacity("cap", lwe_process, lwe_comparator, mock_simulator)
    assert "capacity" in prob.variable_names


# ---------------------------------------------------------------------------
# CharacterizeAdsorptionParameters
# ---------------------------------------------------------------------------


@pytest.fixture
def lwe_comparator(lwe_process, flat_reference):
    cs = lwe_process.flow_sheet.component_system
    time = np.linspace(0, 840, 841)
    sol = np.ones((841, len(cs)))
    ref = ReferenceIO("lwe_ref", time, sol, flow_rate=FLOW_RATE)
    ref.component_system = cs
    comp = Comparator("lwe")
    comp.add_reference(ref)
    comp.add_difference_metric("NRMSE", ref, "column.outlet.outlet[0]")
    return comp


def test_characterize_adsorption_equilibrium_variables(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        is_kinetic=False,
    )
    assert "Protein_characteristic_charge" in prob.variable_names
    assert "Protein_adsorption_rate" in prob.variable_names
    assert "Protein_equilibrium_constant" not in prob.variable_names


def test_characterize_adsorption_kinetic_variables(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        is_kinetic=True,
    )
    assert "Protein_characteristic_charge" in prob.variable_names
    assert "Protein_equilibrium_constant" in prob.variable_names
    assert "Protein_kinetic_constant" in prob.variable_names


def test_characterize_adsorption_kinetic_has_dependencies(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        is_kinetic=True,
    )
    dep_names = [v.name for v in prob.dependent_variables]
    assert "Protein_adsorption_rate" in dep_names
    assert "Protein_desorption_rate" in dep_names


def test_characterize_adsorption_equilibrium_sets_desorption_rate(
    lwe_process, lwe_comparator, mock_simulator
):
    CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        is_kinetic=False,
    )
    bm = lwe_process.flow_sheet.column.binding_model
    assert np.all(np.asarray(bm.desorption_rate) == pytest.approx(1.0))


def test_characterize_adsorption_optional_film_diffusion(
    lwe_process, lwe_comparator, mock_simulator
):
    """film_diffusion is available in LRMP; pore_diffusion requires GRM."""
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        include_film_diffusion=True,
    )
    assert "Protein_film_diffusion" in prob.variable_names


def test_characterize_adsorption_optional_steric_factor(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        include_steric_factor=True,
    )
    assert "Protein_steric_factor" in prob.variable_names


def test_characterize_adsorption_steric_factor_off_by_default(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
    )
    assert "Protein_steric_factor" not in prob.variable_names


def test_characterize_adsorption_kinetic_preserves_capacity_reference(
    lwe_process, lwe_comparator, mock_simulator
):
    lwe_process.flow_sheet.column.binding_model.capacity = 540.0
    CharacterizeAdsorptionParameters(
        "ads",
        lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["Protein"],
        is_kinetic=True,
    )
    bm = lwe_process.flow_sheet.column.binding_model
    assert bm.reference_liquid_phase_conc == pytest.approx(bm.capacity)
    assert bm.reference_solid_phase_conc == pytest.approx(bm.capacity)


@pytest.fixture
def three_component_lwe_process():
    cs = ComponentSystem(["Salt", "Monomer", "HMW"])
    flow_sheet = LCFlowSheet(
        cs,
        sample_loop_volume=LOOP_VOLUME,
        ColumnModel=LumpedRateModelWithPores,
        BindingModel=StericMassAction,
    )
    bm = flow_sheet.column.binding_model
    bm.characteristic_charge = [0.0, 5.0, 10.0]
    bm.adsorption_rate = [0.0, 0.1, 0.1]
    return LWE(
        "lwe",
        flow_sheet,
        c_buffer_a=[20.0, 0.0, 0.0],
        c_buffer_b=[1000.0, 0.0, 0.0],
        c_sample=[20.0, 0.5, 0.05],
        delta_t_wash=120.0,
        delta_t_elute=600.0,
        delta_t_final_wash=120.0,
        flow_rate_wash=FLOW_RATE,
    )


@pytest.fixture
def three_component_lwe_comparator(three_component_lwe_process):
    cs = three_component_lwe_process.flow_sheet.component_system
    time = np.linspace(0, 840, 841)
    ref = ReferenceIO("lwe_ref", time, np.ones((841, len(cs))), flow_rate=FLOW_RATE)
    ref.component_system = cs
    comp = Comparator("lwe")
    comp.add_reference(ref)
    comp.add_difference_metric("NRMSE", ref, "column.outlet.outlet[0]")
    return comp


def test_characterize_adsorption_multiple_components_prefix_variables(
    three_component_lwe_process, three_component_lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        three_component_lwe_process,
        three_component_lwe_comparator,
        mock_simulator,
        components=["Monomer", "HMW"],
        HMW_characteristic_charge={"ub": 40},
    )
    assert prob.variable_names == [
        "Monomer_characteristic_charge",
        "Monomer_adsorption_rate",
        "HMW_characteristic_charge",
        "HMW_adsorption_rate",
    ]
    variables = {v.name: v for v in prob.variables}
    assert variables["HMW_characteristic_charge"].ub == 40
    assert variables["Monomer_characteristic_charge"].ub == 9

    prob.set_variables([6.0, 0.01, 30.0, 0.5])
    bm = three_component_lwe_process.flow_sheet.column.binding_model
    assert list(bm.characteristic_charge[1:]) == pytest.approx([6.0, 30.0])
    assert list(bm.adsorption_rate[1:]) == pytest.approx([0.01, 0.5])


def test_characterize_adsorption_multiple_components_kinetic_dependencies(
    three_component_lwe_process, three_component_lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads",
        three_component_lwe_process,
        three_component_lwe_comparator,
        mock_simulator,
        is_kinetic=True,
        components=["Monomer", "HMW"],
    )
    dep_names = {v.name for v in prob.dependent_variables}
    assert dep_names == {
        "Monomer_adsorption_rate",
        "Monomer_desorption_rate",
        "HMW_adsorption_rate",
        "HMW_desorption_rate",
    }


@pytest.fixture
def grouped_lwe_process():
    cs = ComponentSystem(["Salt"])
    cs.add_component("Protein", species=["Monomer", "Fragment"])
    cs.add_component("HMW")
    fs = LCFlowSheet(
        cs,
        sample_loop_volume=LOOP_VOLUME,
        ColumnModel=GeneralRateModel,
        BindingModel=StericMassAction,
    )
    bm = fs.column.binding_model
    fs.column.film_diffusion = [7e-6] * 4
    fs.column.pore_diffusion = [4e-10] * 4
    fs.column.axial_dispersion = [8e-7] * 4
    bm.capacity = 540.0
    bm.steric_factor = [0.0] * 4
    bm.characteristic_charge = [0.0, 5.0, 6.0, 10.0]
    bm.adsorption_rate = [0.0, 0.1, 0.2, 0.3]
    bm.desorption_rate = [1.0, 1.0, 1.0, 1.0]
    return LWE(
        "grouped_lwe",
        fs,
        c_buffer_a=[20.0, 0.0, 0.0, 0.0],
        c_buffer_b=[1000.0, 0.0, 0.0, 0.0],
        c_sample=[20.0, 0.4, 0.1, 0.05],
        delta_t_wash=120.0,
        delta_t_elute=600.0,
        delta_t_final_wash=120.0,
        flow_rate_wash=FLOW_RATE,
    )


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
def test_component_groups_fit_every_species_on_every_process(
    helper, grouped_lwe_process, lwe_comparator, mock_simulator
):
    other = copy.deepcopy(grouped_lwe_process)
    other.name = "other"
    processes = [grouped_lwe_process, other]
    kwargs = dict(include_film_diffusion=True, include_pore_diffusion=True)
    if helper is CharacterizeParticles:
        kwargs.update(include_axial_dispersion=True, include_particle_porosity=True)
    else:
        kwargs.update(include_steric_factor=True, is_kinetic=True)
    prob = helper(
        "joint",
        processes,
        [lwe_comparator, lwe_comparator],
        mock_simulator,
        components=["HMW", "Protein"],
        Fragment_film_diffusion={"ub": 2e-4},
        **kwargs,
    )
    species = ["Monomer", "Fragment", "HMW"]
    assert prob.variables_dict["Fragment_film_diffusion"].ub == 2e-4
    values = {}
    for i, name in enumerate(species, start=1):
        values[f"{name}_film_diffusion"] = i * 1e-6
        values[f"{name}_pore_diffusion"] = i * 1e-10
        if helper is CharacterizeParticles:
            values[f"{name}_axial_dispersion"] = i * 1e-7
        else:
            values[f"{name}_characteristic_charge"] = 4.0 + i
            values[f"{name}_steric_factor"] = 10.0 * i
            values[f"{name}_equilibrium_constant"] = 0.1 * i
            values[f"{name}_kinetic_constant"] = 0.01 * i
    if helper is CharacterizeParticles:
        assert prob.variable_names.count("particle_porosity") == 1
        values["particle_porosity"] = 0.75
    assert set(prob.independent_variable_names) == set(values)
    # Selection follows component-system order, like slice_solution.
    per_species_names = [
        n for n in prob.independent_variable_names if n != "particle_porosity"
    ]
    assert per_species_names[0].startswith("Monomer_")
    prob.set_variables([values[n] for n in prob.independent_variable_names])
    for process in processes:
        column = process.flow_sheet.column
        assert list(column.film_diffusion[1:]) == pytest.approx([1e-6, 2e-6, 3e-6])
        assert list(column.pore_diffusion[1:]) == pytest.approx([1e-10, 2e-10, 3e-10])
        if helper is CharacterizeParticles:
            assert list(column.axial_dispersion[1:]) == pytest.approx(
                [1e-7, 2e-7, 3e-7]
            )
            assert column.particle_porosity == 0.75
        else:
            bm = column.binding_model
            assert list(bm.characteristic_charge) == pytest.approx([0, 5, 6, 7])
            assert list(bm.steric_factor[1:]) == pytest.approx([10, 20, 30])
            assert list(bm.adsorption_rate) == pytest.approx([0, 10, 10, 10])
            assert list(bm.desorption_rate[1:]) == pytest.approx([100, 50, 100 / 3])
            assert bm.reference_liquid_phase_conc == 540.0
            assert bm.reference_solid_phase_conc == 540.0


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
@pytest.mark.parametrize("components", ["Protein", ["Protein"]])
def test_component_name_expands_to_its_species(
    helper, components, grouped_lwe_process, lwe_comparator, mock_simulator
):
    prob = helper(
        "group",
        grouped_lwe_process,
        lwe_comparator,
        mock_simulator,
        components=components,
        include_film_diffusion=True,
    )
    assert "Monomer_film_diffusion" in prob.variable_names
    assert "Fragment_film_diffusion" in prob.variable_names
    assert not any(n.startswith("HMW_") for n in prob.variable_names)


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
@pytest.mark.parametrize(
    "components, message",
    [
        (None, "distinct"),
        ([], "distinct"),
        (["Protein", "Protein"], "distinct"),
        (["Missing"], "Unknown components"),
        (["Monomer"], "Unknown components"),
    ],
)
def test_invalid_component_selection_is_rejected(
    helper, components, message, grouped_lwe_process, lwe_comparator, mock_simulator
):
    with pytest.raises(ValueError, match=message):
        helper(
            "invalid",
            grouped_lwe_process,
            lwe_comparator,
            mock_simulator,
            components=components,
            include_film_diffusion=True,
        )


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
def test_public_index_selection_is_rejected(
    helper, grouped_lwe_process, lwe_comparator, mock_simulator
):
    with pytest.raises(TypeError, match="component_index"):
        helper(
            "invalid",
            grouped_lwe_process,
            lwe_comparator,
            mock_simulator,
            components=["Protein"],
            component_index=1,
            include_film_diffusion=True,
        )


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
@pytest.mark.parametrize("mismatch", ["names", "indices"])
def test_named_selection_rejects_misaligned_process_species(
    helper, mismatch, grouped_lwe_process, lwe_comparator, mock_simulator
):
    other = copy.deepcopy(grouped_lwe_process)
    other.name = "other"
    if mismatch == "names":
        other.component_system.components_dict["Protein"].species[
            0
        ].name = "OtherMonomer"
    else:
        other.component_system.remove_component("Salt")
        other.component_system.add_component("Salt")
    with pytest.raises(ValueError, match="same names and indices"):
        helper(
            "invalid",
            [grouped_lwe_process, other],
            [lwe_comparator, lwe_comparator],
            mock_simulator,
            components=["Protein"],
            include_film_diffusion=True,
        )


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
def test_named_selection_writes_only_the_selected_component(
    helper, grouped_lwe_process, lwe_comparator, mock_simulator
):
    prob = helper(
        "selected",
        grouped_lwe_process,
        lwe_comparator,
        mock_simulator,
        components=["HMW"],
        include_film_diffusion=True,
    )
    assert "HMW_film_diffusion" in prob.variable_names
    assert all(name.startswith("HMW_") for name in prob.variable_names)
    values = {
        name: 5.0 if name == "HMW_characteristic_charge" else 0.1
        for name in prob.independent_variable_names
    }
    values["HMW_film_diffusion"] = 4e-6
    before = list(grouped_lwe_process.flow_sheet.column.film_diffusion)
    prob.set_variables([values[name] for name in prob.independent_variable_names])
    actual = grouped_lwe_process.flow_sheet.column.film_diffusion
    assert list(actual[:3]) == before[:3]
    assert actual[3] == 4e-6


@pytest.mark.parametrize(
    "helper", [CharacterizeAdsorptionParameters, CharacterizeParticles]
)
def test_component_selection_is_required(
    helper, grouped_lwe_process, lwe_comparator, mock_simulator
):
    with pytest.raises(TypeError, match="components"):
        helper(
            "missing_selection",
            grouped_lwe_process,
            lwe_comparator,
            mock_simulator,
            include_film_diffusion=True,
        )
