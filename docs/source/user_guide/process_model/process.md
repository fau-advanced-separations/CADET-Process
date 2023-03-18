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
sys.path.append('../../../../')
```

(process_guide)=
# Process
The {class}`~CADETProcess.processModel.Process` class is used to define dynamic changes to of the units and connections.
To instantiate a {class}`~CADETProcess.processModel.Process`, a {class}`~CADETProcess.processModel.FlowSheet` needs to be passed as argument, as well as a string to name that process.
For this guide, a {ref}`flow_sheet_batch_elution` {class}`~CADETProcess.processModel.FlowSheet` is used.
For more information on the {class}`~CADETProcess.processModel.FlowSheet`, refer to {ref}`flow_sheet_guide`.

```{code-cell} ipython3
:tags: [remove-cell]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)

from CADETProcess.processModel import Inlet, Outlet
feed = Inlet(component_system, 'feed')
eluent = Outlet(component_system, 'eluent')
outlet = Outlet(component_system, 'outlet')

from CADETProcess.processModel import FlowSheet
flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(feed)
flow_sheet.add_unit(eluent)
flow_sheet.add_unit(outlet)
```

```{code-cell} ipython3
from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch_elution')
```

After instantiation, it is important to also set the overall duration of the process.
Since **CADET-Process** is also designed for cyclic processes (see {ref}`stationarity_guide`), the corresponding attribute is called {attr}`~CADETProcess.processModel.Process.cycle_time`.

```{code-cell} ipython3
process.cycle_time = 10     # s
```

The {meth}`~CADETProcess.processModel.Process.add_event` method requires the following arguments:
- `name`: Name of the event.
- `parameter_path`: Path of the parameter that is changed in dot notation. E.g. to specify the `flow_rate` parameter of {class}`~CADETProcess.processModel.Inlet` unit named `eluent`, of the {class}`~CADETProcess.processModel.FlowSheet` would read as `flow_sheet.eluent.flow_rate`.
- `state`: Value of the attribute that is changed at Event execution.
- `time`: Time at which the event is executed in $s$.

For example, to add an event called `inject_on` which should modify `flow_rate` of unit `feed` to value `1` at time `0`, and an event called `inject_off` which sets the flow rate back to `0` at $t = 1~s$, specify the following:

```{code-cell} ipython3
:tags: [remove-output]

process.add_event('inject_on', 'flow_sheet.feed.flow_rate', 1, 0)
process.add_event('inject_off', 'flow_sheet.feed.flow_rate', 0, 1)
```

If the event shall only modify a single entry of an array, an additional `entry_index` can be added.
For example, the following will only modify the concentration of the first component.

```{code-cell} ipython3
:tags: [remove-output]

process.add_event('conc_high', 'flow_sheet.feed.c', 1, 0, entry_index=0)
process.add_event('conc_low', 'flow_sheet.feed.c', 0, 1, entry_index=0)
```

All events can are stored in the {attr}`~CADETProcess.processModel.Process.events` attribute.
To visualize the trajectory of the parameter state over the entire cycle, the {class}`~CADETProcess.processModel.Process` provides a {meth}`~CADETProcess.processModel.Process.plot_events` method.

```{code-cell} ipython3
_ = process.plot_events()
```

(event_dependencies_guide)=
## Event Dependencies
In order to reduce the complexity of process configurations, {class}`~CADETProcess.dynamicEvents.Event` dependencies can be specified that define the time at which an {class}`~CADETProcess.dynamicEvents.Event` occurs as a function of other {class}`~CADETProcess.dynamicEvents.Event` times.
Especially for more advanced processes, this reduces the degrees of freedom and facilitates the overall handiness.
For this purpose, {class}`Durations <CADETProcess.dynamicEvents.Duration>` can also be defined to describe the time between the execution of two {class}`Events <CADETProcess.dynamicEvents.Event>`.

The time of a dependent event is calculated using the following equation

$$
t = \sum_i^{n_{dep}} \lambda_i \cdot f_i(t_{dep,i}),
$$

where $n_{dep}$ denotes the number of dependencies of a given {class}`~CADETProcess.dynamicEvents.Event`, $\lambda_i$ denotes a linear factor, and $f_i$ is some transform function.

To add a dependency in **CADET-Process**, use the {meth}`~CADETProcess.processModel.Process.add_event_dependency` method of the {class}`~CADETProcess.processModel.Process` class which requires the following arguments:
- `dependent_event` : Name of the event whose value will depend on other events.
- `independent_events` : List of other events on which event depends
- `factors` : List with factors for linear combination of dependencies.

Consider the batch elution process (see {ref}`here <batch_elution_example>`).

```{figure} ../../../examples/batch_elution/figures/event_dependencies.svg
Events of batch elution process.
```

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.batch_elution import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch elution')
process.cycle_time = 600
Q = 60e-6/60
```

Here, every time the feed is switched on, the elution buffer should be switched off and vice versa.

```{code-cell} ipython3
:tags: [remove-output]

process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)

process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
process.add_event_dependency('eluent_off', ['feed_on'])
```

Alternatively, the dependencies can also already be added in the {meth}`~CADETProcess.processModel.Process.add_event` method when creating the {class}`~CADETProcess.dynamicEvents.Event` in the first place.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.batch_elution import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch elution')
Q = 60e-6/60
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
```

```{code-cell} ipython3
:tags: [remove-output]

process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q, dependencies=['feed_off'])
```

Now, only one of the event times needs to be adjusted, e.g. in a process optimization setting.

```{code-cell} ipython3
process.feed_off.time = 10
print(f'feed_off: {process.feed_off.time}')
print(f'eluent_on: {process.eluent_on.time}')
```

For a more complex scenario refer to {ref}`SSR process <ssr_example>`.


## Inlet Profile from Existing Data
In some cases, it is desirable to use an existing concentration profile as inlet profile (e.g. the output of some upstream process).
**CADET-Process** provides a method to automatically fit a piecewise cubic polynomial to a given profile and add the corresponding events to a specific inlet.

E.g. consider this (arbitrary) sinusoidal profile:

```{code-cell} ipython3
:tags: [remove-input]

from examples.operating_modes.lwe_concentration import flow_sheet, inlet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0

import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, process.cycle_time, 1001)
c = np.sin(time/1000) + 1
_ = plt.plot(time/60, c)
plt.ylabel('c / mM')
plt.xlabel('time / min')
```

Consider that the time points are stored in a variable `t`, and the concentration in the variable `c`.
To add the profile to the unit called `inlet`, use the {meth}`~CADETProcess.processModel.Process.add_concentration_profile` method of the {class}`~CADETProcess.processModel.Process` with the following arguments:
- `unit_operation` : Inlet unit operation
- `time` : Time points
- `c` : Concentration values
- `component_index`: Component index to which concentration profile shall be set. If `-1`, the same profile is set for all components.

```{code-cell} ipython3
process.add_concentration_profile('inlet', time, c, component_index=-1)
```

When inspecting the {meth}`~CADETProcess.processModel.Process.plot_events` method, the concentration profile can now be seen.

```{code-cell} ipython3
_ = process.plot_events()
```

Similarly, flow rate profiles can be added to {class}`Inlets <CADETProcess.processModel.Inlet>` and {class}`Cstrs <CADETProcess.processModel.Cstr>`.
Note that for obvious reasons, the component index is omitted.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.lwe_concentration import flow_sheet, inlet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0
```

```{code-cell} ipython3
process.add_flow_rate_profile('inlet', time, c)
_ = process.plot_events()
```

(sensitivity_guide)=
## Parameter Sensitivities
```{warning}
This functionality is still work in progress.
Changes in the interface are to be expected.
```

Parameter sensitivities $s = \partial y / \partial p$ of a solution $y$ with respect to some parameter $p$ are required for various tasks, for example, parameter estimation, process design, and process analysis.
The CADET simulator implements the forward sensitivity approach which creates a linear companion DAE for each sensitive parameter.
To add a parameter sensitivity to the process, use the {meth}`~CADETProcess.processModel.Process.add_parameter_sensitivity` method.
The first argument is the parameter path in dot notation.
Since currently parameter sensitivities cannot be computed for {attr}`CADETProcess.processModel.FlowSheet.output_states` or event times, it is not required to specify `flow_sheet` in the path.
Optionally, a `name` can be provided.
If none is provided, the parameter path is used.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.batch_elution import process
```

```{code-cell} ipython3
process.add_parameter_sensitivity('column.total_porosity')
```

For parameters that are specific to a component, reaction, bound index, etc., this must also be specified in the method.

```{code-cell} ipython3
process.add_parameter_sensitivity(
    'feed.c',
    components=['A'],
    polynomial_coefficients=0,
    section_indices=0,
)
```

Multiple parameters can also be linearly combined.
For each of the arguments, a list of parameter needs to be passed.
In this case, a `name` also must be provided.

For more information on the computed sensitivities, refer to {ref}`simulation_results_guide`.
