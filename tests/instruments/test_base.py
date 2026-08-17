import math

import numpy as np
import pytest
from CADETProcess.instruments import (
    LWE,
    Breakthrough,
    LCFlowSheet,
    LCProcess,
    Phase,
    PhasedProcess,
    PulseInjection,
    Step,
    StepElution,
    ValveEvent,
)
from CADETProcess.processModel import ComponentSystem, Linear, LumpedRateModelWithPores

sample_loop_volume = 50e-9   # m³
sample_loop_diameter = 0.75e-3     # m
flow_rate = 8.3e-9           # m³/s


@pytest.fixture
def salt_cs():
    return ComponentSystem(["Salt"])


@pytest.fixture
def no_column_fs(salt_cs):
    return LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
    )


@pytest.fixture
def column_fs(salt_cs):
    return LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        ColumnModel=LumpedRateModelWithPores,
        BindingModel=Linear,
    )


# --- LCFlowSheet topology ---

def test_full_topology_has_all_units(column_fs):
    expected = {
        "buffer_a", "buffer_b", "buffer_c", "buffer_d", "feed_inlet",
        "mixer", "tubing_pre_injection", "sample_loop",
        "tubing_pre_column", "column", "tubing_post_column", "tubing_detectors",
        "outlet", "waste",
    }
    assert {u.name for u in column_fs.units} == expected


def test_feed_inlet_always_present(no_column_fs):
    assert "feed_inlet" in {u.name for u in no_column_fs.units}


def test_waste_outlet_always_present(no_column_fs):
    assert "waste" in {u.name for u in no_column_fs.units}


def test_bypass_removes_specified_units(no_column_fs):
    unit_names = {u.name for u in no_column_fs.units}
    for bypassed in ["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"]:
        assert bypassed not in unit_names
    assert "outlet" in unit_names
    assert "buffer_c" in unit_names
    assert "buffer_d" in unit_names
    assert "feed_inlet" in unit_names


def test_bypass_tubing_pre_injection_removes_unit(salt_cs):
    fs = LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        bypass_units=["tubing_pre_injection", "tubing_pre_column", "column",
                      "tubing_post_column", "tubing_detectors"],
    )
    assert "tubing_pre_injection" not in {u.name for u in fs.units}
    assert fs.valve_junction_name == "mixer"


def test_bypass_mixer_sets_small_volume(salt_cs):
    fs = LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        bypass_units=["mixer", "tubing_pre_column", "column",
                      "tubing_post_column", "tubing_detectors"],
    )
    assert "mixer" in {u.name for u in fs.units}
    assert fs.mixer.init_liquid_volume == pytest.approx(1e-9)


def test_unknown_bypass_unit_raises(salt_cs):
    with pytest.raises(ValueError, match="Unexpected bypass"):
        LCFlowSheet(
            salt_cs,
            sample_loop_volume=sample_loop_volume,
            sample_loop_diameter=sample_loop_diameter,
            ColumnModel=LumpedRateModelWithPores,
            bypass_units=["not_a_real_unit"],
        )


def test_missing_column_model_raises(salt_cs):
    with pytest.raises(ValueError, match="ColumnModel"):
        LCFlowSheet(
            salt_cs,
            sample_loop_volume=sample_loop_volume,
            sample_loop_diameter=sample_loop_diameter,
        )


def test_sample_loop_diameter_without_volume_raises(salt_cs):
    with pytest.raises(ValueError, match="sample_loop_diameter requires sample_loop_volume"):
        LCFlowSheet(
            salt_cs,
            sample_loop_diameter=sample_loop_diameter,
            bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
        )


# --- LCFlowSheet topology: no loop ---

@pytest.fixture
def no_loop_fs(salt_cs):
    return LCFlowSheet(
        salt_cs,
        bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
    )


def test_no_loop_omits_sample_loop_unit(no_loop_fs):
    assert "sample_loop" not in {u.name for u in no_loop_fs.units}


def test_no_loop_still_has_feed_inlet(no_loop_fs):
    assert "feed_inlet" in {u.name for u in no_loop_fs.units}


def test_loop_diameter_derived_from_volume(salt_cs):
    fs = LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
    )
    expected_id = math.sqrt(4 * sample_loop_volume / math.pi)
    assert fs.sample_loop.diameter == pytest.approx(expected_id)
    assert fs.sample_loop.length == pytest.approx(1.0)


# --- Valve events ---

def test_valve_inject_routes_loop_to_column(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("inject", t=0.0)
    names = {e.name for e in proc.events}
    assert "valve_tubing_pre_injection_inject" in names
    assert "valve_sample_loop_inject" in names


def test_valve_run_routes_direct(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("run", t=0.0)
    ev = next(e for e in proc.events if "tubing_pre_injection" in e.name)
    assert "outlet" in ev.state or list(ev.state.values()) == [1.0]


def test_valve_load_routes_loop_to_waste(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("load", t=0.0)
    ev = next(e for e in proc.events if "sample_loop" in e.name)
    assert ev.state == {"waste": 1.0}


def test_valve_direct_inject_routes_tubing_to_waste(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("direct_inject", t=0.0)
    ev = next(e for e in proc.events if "tubing_pre_injection" in e.name)
    assert ev.state == {"waste": 1.0}


def test_valve_duplicate_position_gets_unique_name(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 1000.0
    proc.add_valve_event("inject", t=0.0)
    proc.add_valve_event("inject", t=700.0)
    names = [e.name for e in proc.events]
    assert len(names) == len(set(names))


def test_valve_unknown_position_raises(no_column_fs):
    proc = LCProcess("p", no_column_fs)
    proc.cycle_time = 100.0
    with pytest.raises(ValueError, match="Unknown valve position"):
        proc.add_valve_event("spin", t=0.0)


def test_valve_inject_degrades_to_run_without_loop(no_loop_fs):
    proc = LCProcess("p", no_loop_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("inject", t=0.0)
    # Only one event: tubing_pre_injection (no sample_loop event)
    assert len(proc.events) == 1
    assert "tubing_pre_injection" in proc.events[0].name


def test_valve_load_degrades_to_run_without_loop(no_loop_fs):
    proc = LCProcess("p", no_loop_fs)
    proc.cycle_time = 100.0
    proc.add_valve_event("load", t=0.0)
    assert len(proc.events) == 1
    assert "tubing_pre_injection" in proc.events[0].name


# --- Templates requiring a loop ---

def test_pulse_injection_requires_loop(no_loop_fs):
    with pytest.raises(ValueError, match="requires a sample loop"):
        PulseInjection("p", no_loop_fs, c_buffer_a=[0.0], c_sample=[1.0],
                       cycle_time=600.0, flow_rate=flow_rate)


def test_lwe_requires_loop(no_loop_fs):
    with pytest.raises(ValueError, match="requires a sample loop"):
        LWE("lwe", no_loop_fs, c_buffer_a=[0.0], c_buffer_b=[1000.0], c_sample=[0.0],
            delta_t_wash=200.0, delta_t_elute=400.0, delta_t_final_wash=100.0,
            flow_rate_wash=flow_rate)


def test_step_elution_requires_loop(no_loop_fs):
    with pytest.raises(ValueError, match="requires a sample loop"):
        StepElution("se", no_loop_fs, c_buffer_a=[0.0], c_buffer_b=[1000.0], c_sample=[0.0],
                    delta_t_wash=200.0, delta_t_elute=400.0, delta_t_final_wash=100.0,
                    flow_rate_wash=flow_rate)


# --- Phase ---

def test_phase_step_fills_composition_end():
    p = Phase(100.0, flow_rate, {"A": 1.0})
    assert p.composition_end == {"A": 1.0}


def test_phase_gradient_preserves_composition_end():
    p = Phase(100.0, flow_rate, {"A": 1.0}, {"B": 1.0})
    assert p.composition_end == {"B": 1.0}


def test_phase_f_with_other_buffer_raises():
    with pytest.raises(ValueError, match="Buffer F cannot coexist"):
        Phase(100.0, flow_rate, {"F": 1.0, "A": 0.5})


def test_phase_f_zero_with_others_allowed():
    # F present but zero — not a real conflict
    p = Phase(100.0, flow_rate, {"F": 0.0, "A": 1.0})
    assert p.composition_start["A"] == 1.0


def test_phase_negative_duration_raises():
    with pytest.raises(ValueError, match="duration must be positive"):
        Phase(-1.0, flow_rate, {"A": 1.0})


def test_phase_negative_flow_rate_raises():
    with pytest.raises(ValueError, match="flow_rate must be non-negative"):
        Phase(100.0, -1.0, {"A": 1.0})


def test_phase_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown buffer keys"):
        Phase(100.0, flow_rate, {"X": 1.0})


def test_phase_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        Phase(100.0, flow_rate, {"A": 1.5})


def test_phase_fractions_not_summing_to_one_raises():
    with pytest.raises(ValueError, match="must sum to 1"):
        Phase(100.0, flow_rate, {"A": 0.5, "B": 0.3})


def test_phase_is_immutable():
    p = Phase(100.0, flow_rate, {"A": 1.0})
    with pytest.raises((AttributeError, TypeError)):
        p.duration = 200.0


def test_phase_composition_start_is_copy():
    original = {"A": 1.0}
    p = Phase(100.0, flow_rate, original)
    original["A"] = 0.5
    assert p.composition_start["A"] == 1.0


# --- PhasedProcess ---

@pytest.fixture
def two_phase_process(no_column_fs):
    phases = [
        Phase(300.0, flow_rate, {"A": 1.0}),
        Phase(500.0, flow_rate, {"B": 1.0}),
    ]
    return PhasedProcess("two_phase", no_column_fs, phases)


def test_phased_cycle_time_is_sum_of_phases(two_phase_process):
    assert two_phase_process.cycle_time == pytest.approx(800.0)


def test_phased_buffer_a_event_at_phase_0(two_phase_process):
    event_names = {e.name for e in two_phase_process.events}
    assert "phase_0_A" in event_names
    assert "phase_1_A" in event_names


def test_phased_buffer_b_zero_at_phase_0(two_phase_process):
    ev = next(e for e in two_phase_process.events if e.name == "phase_0_B")
    assert ev.state == pytest.approx(0.0)


def test_phased_gradient_produces_slope(salt_cs):
    fs = LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
    )
    phases = [Phase(400.0, flow_rate, {"A": 1.0}, {"B": 1.0})]
    proc = PhasedProcess("grad", fs, phases)
    ev = next(e for e in proc.events if e.name == "phase_0_A")
    # A ramps from flow_rate to 0; value should be [v_start, slope]
    assert ev.state[0] == pytest.approx(flow_rate)
    assert ev.state[1] == pytest.approx(-flow_rate / 400.0)


def test_valve_event_fires_at_cumulative_phase_time(no_column_fs):
    steps = [
        Phase(200.0, flow_rate, {"A": 1.0}),
        ValveEvent("inject"),
        Phase(400.0, flow_rate, {"A": 1.0}),
    ]
    proc = PhasedProcess("with_valve", no_column_fs, steps)
    valve_evs = [e for e in proc.events if "valve" in e.name]
    assert len(valve_evs) >= 1
    assert valve_evs[0].time == pytest.approx(200.0)


def test_valve_event_at_start_fires_at_zero(no_column_fs):
    steps = [
        ValveEvent("direct_inject"),
        Phase(600.0, flow_rate, {"A": 1.0}),
    ]
    proc = PhasedProcess("valve_first", no_column_fs, steps)
    valve_evs = [e for e in proc.events if "valve" in e.name]
    assert valve_evs[0].time == pytest.approx(0.0)


def test_consecutive_valve_events_raise(no_column_fs):
    steps = [
        ValveEvent("inject"),
        ValveEvent("run"),
        Phase(600.0, flow_rate, {"A": 1.0}),
    ]
    with pytest.raises(ValueError, match="consecutive ValveEvents"):
        PhasedProcess("bad", no_column_fs, steps)


def test_pulse_injection_has_inject_valve_event_at_zero(no_column_fs):
    proc = PulseInjection(
        "pulse", no_column_fs,
        c_buffer_a=[0.0],
        c_sample=[100.0],
        cycle_time=600.0,
        flow_rate=flow_rate,
    )
    valve_evs = [e for e in proc.events if "inject" in e.name and "valve" in e.name]
    assert len(valve_evs) >= 1
    assert valve_evs[0].time == pytest.approx(0.0)


# --- Breakthrough ---

@pytest.fixture
def breakthrough(no_column_fs):
    return Breakthrough(
        "bt", no_column_fs,
        c_sample=[100.0],
        flow_rate=flow_rate,
        cycle_time=600.0,
    )


def test_breakthrough_uses_feed_by_default(breakthrough):
    np.testing.assert_allclose(breakthrough.flow_sheet.feed_inlet.flow_rate[0], flow_rate)


def test_breakthrough_other_buffers_zero(breakthrough):
    for key in ["buffer_a", "buffer_b", "buffer_c", "buffer_d"]:
        assert breakthrough.flow_sheet[key].flow_rate[0] == 0


def test_breakthrough_feed_concentration(breakthrough):
    assert breakthrough.flow_sheet.feed_inlet.c[0, 0] == pytest.approx(100.0)


def test_breakthrough_cycle_time(breakthrough):
    assert breakthrough.cycle_time == pytest.approx(600.0)


def test_breakthrough_with_regular_buffer(salt_cs):
    fs = LCFlowSheet(
        salt_cs,
        sample_loop_volume=sample_loop_volume,
        sample_loop_diameter=sample_loop_diameter,
        bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
    )
    proc = Breakthrough(
        "bt_b", fs,
        c_sample=[50.0],
        flow_rate=flow_rate,
        cycle_time=300.0,
        sample_buffer="B",
    )
    np.testing.assert_allclose(proc.flow_sheet.buffer_b.flow_rate[0], flow_rate)
    assert proc.flow_sheet.buffer_b.c[0, 0] == pytest.approx(50.0)


# --- StepElution ---

@pytest.fixture
def step_elution(column_fs):
    return StepElution(
        "se", column_fs,
        c_buffer_a=[20.0],
        c_buffer_b=[1000.0],
        c_sample=[20.0],
        delta_t_wash=200.0,
        delta_t_elute=400.0,
        delta_t_final_wash=100.0,
        flow_rate_wash=flow_rate,
    )


def test_step_elution_cycle_time_is_sum_of_phases(step_elution):
    assert step_elution.cycle_time == pytest.approx(700.0)


def test_step_elution_has_three_phase_events(step_elution):
    event_names = {e.name for e in step_elution.events}
    # three phases × two buffer keys (A and B)
    assert {"phase_0_A", "phase_1_B", "phase_2_A"}.issubset(event_names)


def test_step_elution_injects_sample_loop_at_t0(step_elution):
    inject_events = [
        e for e in step_elution.events if e.name.endswith("_inject")
    ]
    assert inject_events
    assert all(e.time == pytest.approx(0.0) for e in inject_events)


def test_step_elution_elute_phase_is_constant(step_elution):
    # phase_1_B should be a step (scalar), not a gradient (list)
    ev = next(e for e in step_elution.events if e.name == "phase_1_B")
    state = ev.state
    assert np.isscalar(state) or (hasattr(state, "__len__") and len(state) == 1)


def test_step_elution_a_zero_during_elute(step_elution):
    ev = next(e for e in step_elution.events if e.name == "phase_1_A")
    state = ev.state
    v = state[0] if hasattr(state, "__len__") else state
    assert v == pytest.approx(0.0)


# --- PulseInjection ---

@pytest.fixture
def pulse_injection(no_column_fs):
    return PulseInjection(
        "pulse", no_column_fs,
        c_buffer_a=[0.0],
        c_sample=[100.0],
        cycle_time=600.0,
        flow_rate=flow_rate,
    )


def test_pulse_buffer_a_carries_flow(pulse_injection):
    np.testing.assert_allclose(pulse_injection.flow_sheet.buffer_a.flow_rate[0], flow_rate)


def test_pulse_buffer_b_has_no_flow(pulse_injection):
    assert pulse_injection.flow_sheet.buffer_b.flow_rate[0] == 0


def test_pulse_buffers_c_d_have_no_flow(pulse_injection):
    assert pulse_injection.flow_sheet.buffer_c.flow_rate[0] == 0
    assert pulse_injection.flow_sheet.buffer_d.flow_rate[0] == 0


def test_pulse_sample_loop_concentration(pulse_injection):
    assert pulse_injection.flow_sheet.sample_loop.c == pytest.approx([100.0])


def test_pulse_cycle_time(pulse_injection):
    assert pulse_injection.cycle_time == 600.0


# --- Step ---

@pytest.fixture
def step_process(no_column_fs):
    return Step(
        "step", no_column_fs,
        c_buffer_a=[0.0],
        c_buffer_b=[1000.0],
        cycle_time=800.0,
        flow_rate=flow_rate,
    )


def test_step_buffer_b_carries_flow(step_process):
    np.testing.assert_allclose(step_process.flow_sheet.buffer_b.flow_rate[0], flow_rate)


def test_step_buffer_a_has_no_flow(step_process):
    assert step_process.flow_sheet.buffer_a.flow_rate[0] == 0


def test_step_buffer_b_concentration(step_process):
    assert step_process.flow_sheet.buffer_b.c[0, 0] == pytest.approx(1000.0)


# --- LWE ---

@pytest.fixture
def lwe_process(column_fs):
    return LWE(
        "lwe", column_fs,
        c_buffer_a=[20.0],
        c_buffer_b=[1000.0],
        c_sample=[20.0],
        delta_t_wash=200.0,
        delta_t_elute=400.0,
        delta_t_final_wash=100.0,
        flow_rate_wash=flow_rate,
    )


def test_lwe_cycle_time_is_sum_of_phases(lwe_process):
    assert lwe_process.cycle_time == pytest.approx(700.0)


def test_lwe_has_events_for_all_three_phases(lwe_process):
    event_names = {e.name for e in lwe_process.events}
    assert event_names == {
        "valve_tubing_pre_injection_inject", "valve_sample_loop_inject",
        "phase_0_A", "phase_0_B",
        "phase_1_A", "phase_1_B",
        "phase_2_A", "phase_2_B",
    }


def test_lwe_injects_sample_loop_at_t0(lwe_process):
    inject_events = [
        e for e in lwe_process.events if e.name.endswith("_inject")
    ]
    assert inject_events
    assert all(e.time == pytest.approx(0.0) for e in inject_events)
