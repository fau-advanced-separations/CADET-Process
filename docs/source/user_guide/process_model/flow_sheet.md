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

(flow_sheet_guide)=
# Flow Sheet
The connectivity of {ref}`unit operations <unit_operation_guide>` is defined in the {class}`~CADETProcess.processModel.FlowSheet` class.
This class provides a directed graph structure that allows for the simple definition of configurations for multiple columns or reactor-separator networks, even when they are cyclic.
Furthermore, unit operation models can be used to model tubing and other external volumes.

To ensure that all unit operations in the process have a consistent components, a {class}`~CADETProcess.processModel.ComponentSystem` is required to instantiate a {class}`~CADETProcess.processModel.FlowSheet`.
For more information about the {class}`~CADETProcess.processModel.ComponentSystem`, refer to {ref}`component_system_guide`.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)
```

```{code-cell} ipython3
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)
```

## Adding Unit Operations
To add unit operations, use {meth}`~CADETProcess.processModel.FlowSheet.add_unit`.
For demonstration purposes, consider an {class}`~CADETProcess.processModel.Inlet` and an {class}`~CADETProcess.processModel.Outlet` unit which are connected.

```{figure} ./figures/io.svg
---
width: 60%
name: flow_sheet_io
---
Simple Flow Sheet.
```

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import Inlet, Outlet
inlet = Inlet(component_system, 'inlet')
outlet = Outlet(component_system, 'outlet')
```

```{code-cell} ipython3
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(outlet)
```

Note that all unit operations in the {class}`~CADETProcess.processModel.FlowSheet` must have a unique name since the name is later used for defining connections (and events).
The {class}`~CADETProcess.processModel.FlowSheet` can be indexed with the unit operation name to access that unit.

```{code-cell} ipython3
print(flow_sheet['inlet'])
```

To list all units of a {class}`~CADETProcess.processModel.FlowSheet`, use the {attr}`~CADETProcess.processModel.FlowSheet.units`.

```{code-cell} ipython3
print(flow_sheet.units)
```

## Adding Connections
Every unit operation can have any number of in- and output streams except for {class}`Inlets <CADETProcess.processModel.Inlet>` that represent streams entering the system, and {class}`Outlets <CADETProcess.processModel.Outlet>` that represent those exiting.
To allow connecting the units use {meth}`~CADETProcess.processModel.FlowSheet.add_connection`.

```{code-cell} ipython3
flow_sheet.add_connection(inlet, outlet)
```

If a unit operation has more than one input, all ingoing streams are mixed before entering the unit.

```{figure} ./figures/mixer.svg
---
width: 60%
name: flow_sheet_mixer
---
Multiple Input Streams.
```

```{code-cell} ipython3
:tags: [hide-cell]

inlet_a = Inlet(component_system, 'inlet_a')
inlet_b = Inlet(component_system, 'inlet_b')
outlet = Outlet(component_system, 'outlet')

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet_a)
flow_sheet.add_unit(inlet_b)
flow_sheet.add_unit(outlet)
```

```{code-cell} ipython3
flow_sheet.add_connection(inlet_a, outlet)
flow_sheet.add_connection(inlet_b, outlet)
```

Note that it is straightforward to also include internal recycles in the {class}`~CADETProcess.processModel.FlowSheet`, which is important for systems such as SSR (see example in Section {ref}`stationarity_guide`) or SMB (see {ref}`here <carousel_tutorial>`).

## Output State
If a unit operation has multiple outputs, the distribution of those streams needs to be specified.
Consider the following {class}`~CADETProcess.processModel.FlowSheet`.

```{figure} ./figures/splitter.svg
---
width: 60%
name: flow_sheet_splitter
---
Multiple Output Streams.
```

```{code-cell} ipython3
:tags: [hide-cell]

inlet = Inlet(component_system, 'inlet')
inlet.flow_rate = 1e-6
outlet_a = Outlet(component_system, 'outlet_a')
outlet_b = Outlet(component_system, 'outlet_b')

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(outlet_a)
flow_sheet.add_unit(outlet_b)

flow_sheet.add_connection(inlet, outlet_a)
flow_sheet.add_connection(inlet, outlet_b)
```

To set the output state, use the {meth}`~CADETProcess.processModel.FlowSheet.set_output_state` method.
The first argument of the method is the corresponding unit.
The second argument is a dictionary of outgoing units with the fraction of the stream that is transported to that destination.
Note that the sum of fractions must add up to $1$.
For example, if $100 \%$ of the flow should be directed to `outlet_a`, set the following:

```{code-cell} ipython3
flow_sheet.set_output_state('inlet', {'outlet_a': 1})
```

For a 50/50 split, set the following:

```{code-cell} ipython3
flow_sheet.set_output_state('inlet', {'outlet_a': 0.5, 'outlet_b': 0.5})
```

The current state of all units in the system is stored in the {class}`~CADETProcess.processModel.FlowSheet.output_states` attribute:

```{code-cell} ipython3
print(flow_sheet.output_states)
```

Note that the output_state can also be changed dynamically during simulation using {class}`Events <CADETProcess.dynamicEvents.Event>`.
For more information, refer to {ref}`process_guide`.

## Flow Rates
In **CADET-Process**, the {class}`~CADETProcess.processModel.Inlet` model acts as source unit that "generates" flow.
This flow is then transported to subsequent unit operations downstream.
Since all fluids in **CADET-Process** are considered incompressible, all flow that enters a unit must also leave that same unit.
The exception is the {class}`~CADETProcess.processModel.Cstr` which can have a variable volume.
If the `flow_rate` of a {class}`~CADETProcess.processModel.Cstr` is not `None`, the outgoing streams can be decoupled from the ingoing streams.
This can be useful to e.g. model holdup volumes.
However, it is important that the volume of a {class}`~CADETProcess.processModel.Cstr` never becomes $0$ or **CADET** will raise an `Exception`.
If the `flow_rate` of a {class}`~CADETProcess.processModel.Cstr` is `None`, the unit is treated like all other unit operations models and the outgoing flow rate equals the ingoing flow rate.
This can be useful when e.g. modelling valves.

Since internal recycles are also possible in **CADET-Process**, the actual flow rates for every unit operation can be calculated with the {meth}`~CADETProcess.processModel.FlowSheet.get_flow_rates`.
Considering the previous example depicted {ref}`above <flow_sheet_splitter>` and assuming a flow rate of $1~mL\cdot s^{-1}$, the flow rate is evenly split:

```{code-cell} ipython3
print(flow_sheet.get_flow_rates())
```

Note that this calculation is performed automatically for all time sections before running a simulation to account for dynamic changes of flow rates and output states.

## Marking Unit Operations for Performance Indicators
To properly determine process performance indicators such as yield or eluent consumption, {class}`~CADETProcess.processModel.Inlet` unit operations can be registered as `feed_inlet` or `eluent_inlet`.
This way, the total mass injected and the total eluent volume used are automatically determined from those units.
For more information, also refer to {ref}`fractionation_guide`.

```{figure} ./figures/batch_elution.svg
---
width: 60%
name: flow_sheet_batch_elution
---
Batch Elution Flow Sheet
```

```{code-cell} ipython3
:tags: [hide-cell]

feed = Inlet(component_system, 'feed')
eluent = Inlet(component_system, 'eluent')
outlet = Outlet(component_system, 'outlet')

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed)
flow_sheet.add_unit(eluent)
flow_sheet.add_unit(outlet)
```

```
flow_sheet.add_feed_inlet('feed')
flow_sheet.add_eluent_inlet('eluent')
flow_sheet.add_product_outlet('outlet')
```

Alternatively, this flag can also be set when adding the unit operations.

```{code-cell} ipython3
:tags: [hide-cell]

feed = Inlet(component_system, 'feed')
eluent = Inlet(component_system, 'eluent')
outlet = Outlet(component_system, 'outlet')

flow_sheet = FlowSheet(component_system)
```

```{code-cell} ipython3
flow_sheet.add_unit(feed, feed_inlet=True)
flow_sheet.add_unit(eluent, eluent_inlet=True)
flow_sheet.add_unit(outlet, product_outlet=True)
```
