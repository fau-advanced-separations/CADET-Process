---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(stationarity_tutorial)=
# Cyclic Stationarity
Preparative chromatographic separations are operated in a repetitive fashion.
In particular processes that incorporate the recycling of streams, like steady-state-recycling (SSR) or simulated moving bed (SMB), have a distinct startup behavior that takes multiple cycles until a periodic steady state is reached.
But also in conventional batch chromatography several cycles are needed to attain stationarity in optimized situations where there is a cycle-to-cycle overlap of the elution profiles of consecutive injections.
However, it is not known beforehand how many cycles are required until cyclic stationarity is established.

For this reason, the `Simulator` can simulate a `ProcessModel` for a fixed number of cycles, or continue simulating until the corresponding module in {ref}`framework_overview` confirms that cyclic stationarity is reached.
Different criteria can be specified such as the maximum deviation of the concentration profiles or the peak areas of consecutive cycles {cite}`Holmqvist2015`.
The simulation terminates if the corresponding difference is smaller than a specified value.
For the evaluation of the process (see {ref}`fractionation_tutorial`), only the last cycle is examined, as it yields a representative `Performance` of the process in all later cycles.

## Demonstration
<!-- To demonstrate this, a SSR process is considered (see {ref}`ssr`.) -->
```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../')

from examples.operating_modes.ssr import process
```

A first strategy is to simulte multiple cycles at once.
For this purpose, we can specify `n_cycles` for the `ProcessSimulator`.

```{code-cell} ipython3
from CADETProcess.simulation import Cadet
process_simulator = Cadet()
process_simulator.n_cycles = 10
simulation_results = process_simulator.simulate(process)
_ = simulation_results.solution.column.outlet.plot()
```
However, it is hard to anticipate, when steady state is reached.
To automatically simulate until stationarity is reached, set `evaluate_stationarity` in the `Solver` to `True`.

```{code-cell} ipython3
process_simulator.evaluate_stationarity = True
simulation_results = process_simulator.simulate(process)
_ = simulation_results.solution.column.outlet.plot()
```
Here, the simulator ran 27 cycles.

It is possible to access the solution of any of the cycles.
For the last cycle, use the index `-1`.

```{code-cell} ipython3
_ = simulation_results.solution_cycles.column.outlet[-1].plot()
```

