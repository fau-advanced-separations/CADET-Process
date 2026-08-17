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


# ---------------------------------------------------------------------------
# CharacterizeParticles
# ---------------------------------------------------------------------------


def test_characterize_particles_no_flags_raises(pulse_process, comparator, mock_simulator):
    with pytest.raises(ValueError, match="at least one"):
        CharacterizeParticles("p", pulse_process, comparator, mock_simulator)


def test_characterize_particles_film_diffusion(pulse_process, comparator, mock_simulator):
    prob = CharacterizeParticles(
        "p", pulse_process, comparator, mock_simulator,
        include_film_diffusion=True,
    )
    assert "film_diffusion" in prob.variable_names


def test_characterize_particles_lrmp_flags(pulse_process, comparator, mock_simulator):
    """LRMP supports axial_dispersion, particle_porosity, film_diffusion (not pore_diffusion)."""
    prob = CharacterizeParticles(
        "p", pulse_process, comparator, mock_simulator,
        include_axial_dispersion=True,
        include_particle_porosity=True,
        include_film_diffusion=True,
    )
    assert "axial_dispersion" in prob.variable_names
    assert "particle_porosity" in prob.variable_names
    assert "film_diffusion" in prob.variable_names


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
        "ads", lwe_process, lwe_comparator, mock_simulator, is_kinetic=False
    )
    assert "characteristic_charge" in prob.variable_names
    assert "adsorption_rate" in prob.variable_names
    assert "equilibrium_constant" not in prob.variable_names


def test_characterize_adsorption_kinetic_variables(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads", lwe_process, lwe_comparator, mock_simulator, is_kinetic=True
    )
    assert "characteristic_charge" in prob.variable_names
    assert "equilibrium_constant" in prob.variable_names
    assert "kinetic_constant" in prob.variable_names


def test_characterize_adsorption_kinetic_has_dependencies(
    lwe_process, lwe_comparator, mock_simulator
):
    prob = CharacterizeAdsorptionParameters(
        "ads", lwe_process, lwe_comparator, mock_simulator, is_kinetic=True
    )
    dep_names = [v.name for v in prob.dependent_variables]
    assert "adsorption_rate" in dep_names
    assert "desorption_rate" in dep_names


def test_characterize_adsorption_equilibrium_sets_desorption_rate(
    lwe_process, lwe_comparator, mock_simulator
):
    CharacterizeAdsorptionParameters(
        "ads", lwe_process, lwe_comparator, mock_simulator, is_kinetic=False
    )
    bm = lwe_process.flow_sheet.column.binding_model
    assert np.all(np.asarray(bm.desorption_rate) == pytest.approx(1.0))


def test_characterize_adsorption_optional_film_diffusion(
    lwe_process, lwe_comparator, mock_simulator
):
    """film_diffusion is available in LRMP; pore_diffusion requires GRM."""
    prob = CharacterizeAdsorptionParameters(
        "ads", lwe_process, lwe_comparator, mock_simulator,
        include_film_diffusion=True,
    )
    assert "film_diffusion" in prob.variable_names
