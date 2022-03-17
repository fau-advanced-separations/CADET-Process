---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(fractionation_tutorial)=
# Product Fractionation
Key information for evaluating the process `Performance` (cf. {ref}`framework_overview`) of a chromatographic process are the amounts of the target components in the collected product fractions.

In this work, the `Fractionation` module automatically sets up an `OptimizationProblem`.
For every component, different purity requirements can be specified, and any function may be applied as objective.

For the objective and constraint functions, fractions are pooled from all `Outlets` of the `FlowSheet` (see equations {eq}`mass` and {eq}`purity`).
As initial values for the optimization, areas of the chromatogram with sufficient local purity are identified, i.e., intervals where $PU_i(t)=c_i(t)/\sum_j c_j(t)\geq PU_{min,i}$ {cite}`Shan2004`.
These initial intervals are then expanded by the optimizer towards regions of lower purity while meeting the cumulative purity constraints.
In the current implementation, **COBYLA** {cite}`Powell1994` of the **SciPy** {cite}`SciPyContributors2020` library is used as `OptimizationSolver`
Yet, any other solver or heuristic algorithm may be implemented.

## Demonstration
To demonstrate the strategy, we can reuse the example from the previous {ref}`tutorial <simulation_tutorial>`.

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../')

from examples.operating_modes.batch_elution import process
```
To enable the calculation of the process parameters, it is necessary to specify which of the inlets should be considered for the feed and eluent consumption.
Moreover, the outlet(s) which are used for evaluation need to be defined.

```
flow_sheet.add_feed_source('feed')
flow_sheet.add_eluent_source('eluent')
flow_sheet.add_chromatogram_sink('outlet')
```

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.simulation import Cadet
process_simulator = Cadet()
simulation_results = process_simulator.simulate(process)
```

After simulation, we can now setup a `FractionationOptimizer`.
By default, the mass of the components is maximized.
However, other objective functions can be used.
Depending on the target component, we can specify a required purity. 
For example, if only the first component is relevant, and requires a purity $\ge 95 \%$, we can specify the following:

```{code-cell} ipython3
from CADETProcess.fractionation import FractionationOptimizer
purity_required = [0.95, 0]
fractionation_optimizer = FractionationOptimizer(purity_required)
```

To call the procedure, we now need to pass the chromatograms and some information which is stored in `process_meta`.
```{code-cell} ipython3
fractionation = fractionation_optimizer.optimize_fractionation(
    simulation_results.chromatograms,
    process.process_meta,
)
```
The results are stored in a `Performance` object.
```{code-cell} ipython3
print(fractionation.performance)
```
The chromatogram can also be plotted with the fraction times overlaid:
```{code-cell} ipython3
fractionation.plot_fraction_signal()
```

For comparison, if only the second component is relevant

```{code-cell} ipython3
purity_required = [0, 0.95]
fractionation_optimizer = FractionationOptimizer(purity_required)
fractionation = fractionation_optimizer.optimize_fractionation(
    simulation_results.chromatograms,
    process.process_meta,
)
print(fractionation.performance)
fractionation.plot_fraction_signal()
```

But of course, also both components can be valuable.
Here, the required purity is also reduced to demonstrate that overlapping fractions are automatically avoided by internally introducing linear constraints.

```{code-cell} ipython3
purity_required = [0.8, 0.8]
fractionation_optimizer = FractionationOptimizer(purity_required)
fractionation = fractionation_optimizer.optimize_fractionation(
    simulation_results.chromatograms,
    process.process_meta,
)
print(fractionation.performance)
fractionation.plot_fraction_signal()
```

