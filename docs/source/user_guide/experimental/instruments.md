---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../../')
%matplotlib inline
```

(instruments_guide)=
# Instruments

The {mod}`~CADETProcess.instruments` module provides ready-made LC system flow sheet templates and experiment process classes.
Rather than manually assembling a {class}`~CADETProcess.processModel.FlowSheet` and adding events by hand, you pick a pre-built template or compose phases declaratively.

The design follows the same two-step pattern as plain CADET-Process objects: construct the flow sheet first, then pass it to the process.

## LC system topology

All templates share the same physical topology, modelled by {class}`~CADETProcess.instruments.LCFlowSheet`.
The topology and valve nomenclature follow typical preparative LC instruments such as the Äkta (Cytiva) and Knauer Azura series.

```{figure} instruments/figures/flowsheet_lcsystem.png
:alt: LC system flow sheet
:width: 100%

Physical topology of the modeled LC system.
Internally, each tubing segment, the mixer, and the sample loop are represented as explicit unit operations with configurable dimensions; see the units table below.
Pumps and inline detectors are not modeled as separate units: flow rates are set directly on the inlet units.
```

**Inlets**

| Name                     | Purpose                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| `buffer_a` to `buffer_d` | Eluent reservoirs; flow through the mixer for gradient formation |
| `feed_inlet`             | Feed inlet; connects directly to the sample loop                 |

**Internal units**

| Name                   | Model                                              | Purpose                                          |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------ |
| `mixer`                | {class}`~CADETProcess.processModel.Cstr`           | Mixes buffers A to D                             |
| `tubing_pre_injection` | {class}`~CADETProcess.processModel.TubularReactor` | Dead volume between mixer and injection point    |
| `sample_loop`          | {class}`~CADETProcess.processModel.TubularReactor` | Injection loop; content set as initial condition |
| `tubing_pre_column`    | {class}`~CADETProcess.processModel.TubularReactor` | Pre-column dead volume                           |
| `column`               | configurable                                       | Chromatographic column                           |
| `tubing_post_column`   | {class}`~CADETProcess.processModel.TubularReactor` | Post-column dead volume                          |
| `tubing_detectors`     | {class}`~CADETProcess.processModel.TubularReactor` | Detector cell dead volume                        |

**Outlets**: `outlet` (product) and `waste` (for loop loading and pump waste routing).

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem, LumpedRateModelWithPores, Linear
from CADETProcess.instruments import LCFlowSheet

cs = ComponentSystem(["Salt", "Protein"])
Q = 1.0e-8  # m³/s

fs = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    ColumnModel=LumpedRateModelWithPores,
    BindingModel=Linear,
)
print("units:", [u.name for u in fs.units])
```

Units can be excluded from the flow path to characterize the system sequentially, starting from the simplest configuration and adding components one at a time:

```{code-cell} ipython3
fs_no_col = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
)
print("units:", [u.name for u in fs_no_col.units])
```

## Process templates

Each template takes a pre-constructed {class}`~CADETProcess.instruments.LCFlowSheet` as its second argument.

### PulseInjection

The system is pre-equilibrated with buffer A.
The sample loop content is injected at t=0 and washed through with buffer A.

```{code-cell} ipython3
from CADETProcess.instruments import PulseInjection

pulse = PulseInjection(
    "pulse", fs_no_col,
    c_buffer_a=[0.0, 0.0],
    c_sample=[0.0, 1.0],
    cycle_time=600.0,
    flow_rate=Q,
)
print("cycle_time:", pulse.cycle_time, "s")
print("buffer_a flow rate:", pulse.flow_sheet.buffer_a.flow_rate[0], "m³/s")
pulse.plot_events();
```

### Step

Switch from buffer A to buffer B at t=0.
Useful for measuring system dead volumes and mixing dynamics.

```{code-cell} ipython3
from CADETProcess.instruments import Step

step = Step(
    "step", fs_no_col,
    c_buffer_a=[0.0, 0.0],
    c_buffer_b=[1000.0, 0.0],
    cycle_time=800.0,
    flow_rate=Q,
)
step.plot_events();
```

### LWE

Load-wash-elute with a linear salt gradient.
After the wash phase, buffer B ramps up linearly while buffer A ramps down.

```{code-cell} ipython3
from CADETProcess.instruments import LWE

lwe = LWE(
    "lwe", fs,
    c_buffer_a=[20.0, 0.0],
    c_buffer_b=[1000.0, 0.0],
    c_sample=[20.0, 1.0],
    delta_t_wash=200.0,
    delta_t_elute=400.0,
    delta_t_final_wash=100.0,
    flow_rate_wash=Q,
)
print("cycle_time:", lwe.cycle_time, "s")
lwe.plot_events();
```

### StepElution

Like {class}`~CADETProcess.instruments.LWE` but with an instantaneous step to buffer B
instead of a gradient.

```{code-cell} ipython3
from CADETProcess.instruments import StepElution

se = StepElution(
    "se", fs,
    c_buffer_a=[20.0, 0.0],
    c_buffer_b=[1000.0, 0.0],
    c_sample=[20.0, 1.0],
    delta_t_wash=200.0,
    delta_t_elute=400.0,
    delta_t_final_wash=100.0,
    flow_rate_wash=Q,
)
print("cycle_time:", se.cycle_time, "s")
se.plot_events();
```

### Breakthrough

Sample flows continuously from t=0 through the column.
By default the sample is delivered via the feed inlet (`feed_inlet`).
Pass `sample_buffer="B"` (or any key A to D) to use a main buffer instead.

```{code-cell} ipython3
from CADETProcess.instruments import Breakthrough

bt = Breakthrough(
    "bt", fs,
    c_sample=[20.0, 1.0],
    flow_rate=Q,
    cycle_time=600.0,
)
print("feed flow rate:", bt.flow_sheet.feed_inlet.flow_rate[0], "m³/s")
bt.plot_events();
```

## PhasedProcess

{class}`~CADETProcess.instruments.PhasedProcess` lets you compose arbitrary phase sequences.
Each {class}`~CADETProcess.instruments.Phase` specifies a duration, flow rate, and buffer fractions at the start (and optionally end) of the phase.

```{code-cell} ipython3
from CADETProcess.instruments import Phase, ValveEvent, PhasedProcess

# Three-phase wash / gradient / final-wash
steps = [
    Phase(200.0, Q, {"A": 1.0}),                    # wash: 100 % A
    Phase(400.0, Q, {"A": 1.0}, {"B": 1.0}),        # gradient: A to 0, B to 1
    Phase(100.0, Q, {"A": 1.0}),                     # final wash
]
proc = PhasedProcess("custom", fs_no_col, steps)
print("cycle_time:", proc.cycle_time, "s")
print("events:", [e.name for e in proc.events])
proc.plot_events();
```

A **step** is a phase whose composition does not change (`composition_end=None`):

```{code-cell} ipython3
step_phases = [
    Phase(200.0, Q, {"A": 1.0}),   # 100 % A
    Phase(400.0, Q, {"B": 1.0}),   # step to 100 % B
    Phase(100.0, Q, {"A": 1.0}),   # step back to A
]
```

The **feed inlet** can appear as key `"F"` in a phase composition, but cannot be mixed with buffers A to D in the same phase:

```{code-cell} ipython3
# Deliver sample via feed inlet for 60 s, then switch to running buffer
feed_phases = [
    Phase(60.0,  Q, {"F": 1.0}),    # sample via feed
    Phase(540.0, Q, {"A": 1.0}),    # running buffer
]
```

## Valve events

{class}`~CADETProcess.instruments.ValveEvent` is an instantaneous valve position change.
Its time is determined by the cumulative duration of all preceding phases in the step sequence.
The default valve state before the first step is `"run"`.

Valve events and phases are passed together as a single `steps` list to {class}`~CADETProcess.instruments.PhasedProcess`.
This example injects at t=0 and returns to run after 30 s:

```{code-cell} ipython3
fs_pulse = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
)
pulse_steps = [
    ValveEvent("inject"),
    Phase(30.0,  Q, {"A": 1.0}),
    ValveEvent("run"),
    Phase(570.0, Q, {"A": 1.0}),
]
proc_pulse = PhasedProcess("pulse", fs_pulse, pulse_steps)
proc_pulse.plot_events();
```

The four named positions, with two aliases:

## Valve positions

SyP (system pump) is the buffer line: buffers A to D flow through the mixer into `tubing_pre_injection`.
SaP (sample pump) is the feed line: `feed_inlet` connects directly to the sample loop (or `first_unit` when no loop is present).

Four named positions are available, with two aliases:

| Position          | Alias                 | SyP (`tubing_pre_injection`) | SaP / loop     | Requires loop             |
| ----------------- | --------------------- | ---------------------------- | -------------- | ------------------------- |
| `"run"`           |                       | → `first_unit`               | → waste        | no                        |
| `"load"`          |                       | → `first_unit`               | → loop → waste | yes (degrades to `"run"`) |
| `"inject"`        | `"sample_pump_waste"` | → loop → `first_unit`        | → waste        | yes (degrades to `"run"`) |
| `"direct_inject"` | `"system_pump_waste"` | → waste                      | → `first_unit` | no                        |

**Single injection then run** (using {class}`~CADETProcess.instruments.LCProcess` directly with {meth}`~CADETProcess.instruments.LCProcess.add_valve_event`):

```{code-cell} ipython3
from CADETProcess.instruments import LCProcess

fs_valve = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
)
proc = LCProcess("single_inj", fs_valve)
proc.cycle_time = 600.0
proc.add_valve_event("inject", t=0.0)   # push loop contents through column
proc.add_valve_event("run",    t=30.0)  # return to normal run after loop is cleared
proc.plot_events();
```

**Successive injections:** reload the loop during the run:

```{code-cell} ipython3
fs_multi = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
)
proc2 = LCProcess("multi_inj", fs_multi)
proc2.cycle_time = 1400.0

proc2.add_valve_event("inject", t=0.0)    # first injection: loop → column
proc2.add_valve_event("load",   t=30.0)   # feed_inlet fills loop; column on direct path
proc2.add_valve_event("inject", t=700.0)  # second injection
proc2.add_valve_event("load",   t=730.0)  # reload again

proc2.plot_events();
```

**System equilibration:** waste system pump output before the column:

```{code-cell} ipython3
fs_equil = LCFlowSheet(
    cs,
    sample_loop_volume=50e-9,
    sample_loop_diameter=0.75e-3,
    bypass_units=["tubing_pre_column", "column", "tubing_post_column", "tubing_detectors"],
)
proc3 = LCProcess("equil", fs_equil)
proc3.cycle_time = 600.0

proc3.add_valve_event("system_pump_waste", t=0.0)   # system pump to waste while equilibrating
proc3.add_valve_event("run",               t=60.0)  # switch to column path
proc3.plot_events();
```
