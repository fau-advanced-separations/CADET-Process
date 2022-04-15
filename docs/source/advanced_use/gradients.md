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
sys.path.append('../../../')
```

(gradients_tutorial)=
# Gradients and Inlet Profiles
In many chromatographic processes, concentration gradients are used to improve separation performance (e.g. salt concentrations or the $pH$ are continuously increased or decreased).

In **CADET-Process**, the temporal profile of inlet concentrations, as well as of flow rates can be modelled using piecewise cubic polynomials.
While in many cases, both approaches can be used to achieve identical numerical results, there are some subtle conceptual differences which will be demonstrated in this tutorial.

Before investigating a real example, the interface to set these polynomial parameters is introduced.
For this purpose, consider a 2 component system.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(2)

from CADETProcess.processModel import Source
inlet = Source(component_system, 'inlet')
```

In previous tutorials, only constant values were considered for inlets.

```{code-cell} ipython3
# Constant value for all components
inlet.c = 1

# Constant value per component
inlet.c = [1, 2]
```

To specify the polynomial coefficients, a list of lists needs to be set.
For each component, the coefficients are added in ascending order where the first entry is the constant term, the second is the linear term etc.
E.g. consider a gradient where the first component concentration has a constant term of $0~mM$ and increases linearly with slope $1 mM \cdot s^{-1}$, and the second component starts at $2~mM$ and decreases with a quadratic term of $-1~mM \cdot s^{-2}$.

```{code-cell} ipython3
# Polynomial value per component
inlet.c = [[0, 1], [2, 0, -1]]
print(inlet.c)
```

Similarly, the polynomial coefficients for the unit flow rate can be set for `Sources` and `Cstrs`.

```{code-cell} ipython3
# Constant flow rate
inlet.flow_rate = 1
print(inlet.flow_rate)
```

```{code-cell} ipython3
# Polynomial coefficients
inlet.flow_rate = [1,2]
print(inlet.flow_rate)
```

Since these parameters are mostly used in dynamic process models, they are usually modified using `Events`.
To demonstrate this feature in a real process, consider a load-wash-elude process.
First, the column is loaded with a feed solution containing a protein with $0.1~mM$ protein and salt with $180~mM$.
Then, the column is washed with a $70~mM$ salt buffer.
Finally, the salt concentration is linearly increased to $500~mM$.

```{figure} ../examples/operating_modes/figures/lwe_inlet_profile.png
:name: lwe_inlet_profile

Inlet profile for load-wash-elute process.
```

## Concentration Gradients
With this approach, a single `Source` is used to directly describe the concentration profile at the column inlet.
To model a gradient, `Events` are added to the process which modify the polynomial coefficients of the inlet concentration parameters at a given time.

```{figure} ../examples/operating_modes/figures/lwe_concentration_flow_sheet.svg
:name: lwe_concentration_flow_sheet

Flow sheet for load-wash-elute process using a single inlet.
```

```{figure} ../examples/operating_modes/figures/lwe_concentration_events.svg
:name: lwe_concentration_events

Events of load-wash-elute process using a single inlet and modifying its concentration.
```

For the full process configuration, see {ref}`here <lwe_example_concentration>`.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np

from examples.operating_modes.lwe_concentration import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0
```

```{code-cell} ipython3
wash_start = 7500.0
gradient_start = 9500.0
gradient_duration = process.cycle_time - gradient_start

process.add_event('load', 'flow_sheet.inlet.c', [180.0, 0.1])
process.add_event('wash', 'flow_sheet.inlet.c', [70.0, 0.0], wash_start)
process.add_event('elute', 'flow_sheet.inlet.c', [[70.0, (500-70)/gradient_duration], [0]], gradient_start)
_ = process.plot_events()
```

## Flow Rate Gradients

Alternatively, multiple `Source` units can be considered with constant concentrations where the gradient at the column inlet results from dynamically changing the ratio of ingoing flow rates.
E.g., for a linear gradient, the flow rate of one unit operation is continuously increased while the flow rate of another unit operation is continuously decreased, effectively modelling the behaviour of a mixing pump connected to two containers.
While this approach requires the definition of more unit operations and events, it is usually a better representation of the actual process in which different buffers with constant concentrations are prepared and to create concentration gradients, the mixing ratio is continuously modified.

```{figure} ../examples/operating_modes/figures/lwe_flow_rate_flow_sheet.svg
:name: lwe_flow_rate_flow_sheet

Flow sheet for load-wash-elute process using a separate inlets for buffers.
```

```{figure} ../examples/operating_modes/figures/lwe_flow_rate_events.svg
:name: lwe_flow_rate_events

Events of load-wash-elute process using multiple inlets and mofifying their flow rates.
```

For the full configuration, please refer to the {ref}`example <lwe_example_flow_rate>`.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.lwe_flow_rate import flow_sheet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0
```

```{code-cell} ipython3
Q = 2.88e-8
load_duration = 7500.0
gradient_start = 9500.0
gradient_slope = Q/(process.cycle_time - gradient_start)

process.add_event('load_on', 'flow_sheet.load.flow_rate', Q)
process.add_event('load_off', 'flow_sheet.load.flow_rate', 0.0)
process.add_duration('load_duration', time=load_duration)
process.add_event_dependency('load_off', ['load_on', 'load_duration'], [1, 1])

process.add_event('wash_off', 'flow_sheet.wash.flow_rate', 0)
process.add_event('elute_off', 'flow_sheet.elute.flow_rate', 0)

process.add_event('wash_on', 'flow_sheet.wash.flow_rate', Q, time=load_duration)
process.add_event_dependency('wash_on', ['load_off'])

process.add_event('wash_gradient', 'flow_sheet.wash.flow_rate', [Q, -gradient_slope], gradient_start)
process.add_event('elute_gradient', 'flow_sheet.elute.flow_rate', [0, gradient_slope])
process.add_event_dependency('elute_gradient', ['wash_gradient'])

_ = process.plot_events()
```

## Inlet Profile from Existing Data
In some cases, it is desirable to use an existing concentration profile as inlet profile (e.g. the output of some upstream process).
**CADET-Process** provides a method to automatically fit a piecewise cubic polynomial to a given profile and add the corresponding events to a specific inlet.

E.g. consider this (arbitrary) sinusoidal profile:

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, process.cycle_time, 1001)
c = np.sin(time/1000) + 1
_ = plt.plot(time/60, c)
plt.ylabel('c / mM')
plt.xlabel('time / min')
```

Consider that the time points are stored in a variable `t`, and the concentration in the variable `c`.
To add the profile to the unit called `inlet`, use the `add_concentration_profile` method of the `Process` with the following arguments:
- `unit_operation` : Inlet unit operation
- `time` : Time points
- `c` : Concentration values
- `component_index`: Component index to which concentration profile shall be set. If `-1`, the same profile is set for all components.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.lwe_concentration import flow_sheet, inlet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0
```

```{code-cell} ipython3
process.add_concentration_profile(inlet, time, c, component_index=-1)
```

When inspecting the `plot_events` method, the concentration profile can now be seen.

```{code-cell} ipython3
_ = process.plot_events()
```

Similarly, flow rate profiles can be added to `Sources` and `Cstrs`.
Note that for obvious reasons, the component index is omitted.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.operating_modes.lwe_concentration import flow_sheet, inlet

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0
```

```{code-cell} ipython3
process.add_flow_rate_profile(inlet, time, c)
_ = process.plot_events()
```
