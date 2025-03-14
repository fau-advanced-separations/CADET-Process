---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
execution:
  timeout: 300
---

%matplotlib notebook

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```

(fractionation_guide)=
# Product Fractionation
Key information for evaluating the separation performance of a chromatographic process are the amounts of the target components in the collected product fractions.
To define corresponding fractionation intervals, the chromatograms, i.e., the concentration profiles $c_{i,k}\left(t\right)$ at the outlet(s) of the process must be evaluated.
In a strict sense, a chromatogram is only given at the outlet of a single column. Note that here this term is used more generally for the concentration profiles at the outlets of a flow sheet, which only accounts for material leaving the process.
The times for the start, $t_{start, j}$, and the end, $t_{end, j}$, of a product fraction $j$ have to be chosen such that constraints on product purity are met.
It is important to note, that in advanced chromatographic process configurations, outlet chromatograms can be much more complex than the example shown below and that multiple sections of the chromatogram may represent suitable fractions $j$ for collecting one target component $i$.
Moreover, flow sheets can have multiple outlets $k$ that have to be fractionated simultaneously.
Also, the volumetric flow rate $Q_k$ at the outlets may depend on time and needs to be considered in the integral.
These aspects are considered by defining the total product amount of a component $i$ as

```{math}
:label: mass
m_{i} = \sum_{k=1}^{n_{chrom}} \sum_{j=1}^{n_{frac, k}^{i}}\int_{t_{start, j}}^{t_{end, j}} Q_k(t) \cdot c_{i,k}(t) dt,\\
```

where $n_{frac, k}^{i}$ is the number of fractions considered for component $i$ in chromatogram $k$, and $n_{chrom}$ is the number of chromatograms that is evaluated.

Further performance criteria typically used for evaluation and optimization of chromatographic performance are the specific productivity, $PR_i$, the recovery yield, $Y_i$, and the specific solvent consumption, $EC_i$, which all depend on the product amounts:

```{math}
:label: productivity
PR_{i} = \frac{m_i}{V_{solid} \cdot \Delta t_{cycle}},\\
```

```{math}
:label: yield
Y_{i} = \frac{m_i}{m_{feed, i}},\\
```

```{math}
:label: eluent_consumption
EC_{i} = \frac{V_{solvent}}{m_i},\\
```

with $V_{solid}$ being the volume of stationary phase, $V_{solvent}$ that of the solvent introduced during a cycle with duration $\Delta t_{cycle}$, and $m_{feed}$ the injected amount of mixture to be separated. Multiple {class}`Inlets <CADETProcess.processModel.Inlet>` can be considered for the amounts of consumed feed and solvent,

```{math}
:label: solvent_in
V_{solvent} = \sum_{s=1}^{n_{solvents}} \int_{0}^{t_{cycle}} Q_s(t) dt,\\
```

```{math}
:label: feed_in
m_{feed,i} = \sum_{f=1}^{n_{feeds}} \int_{0}^{t_{cycle}} Q_f(t) \cdot c_{f,i}(t) dt.\\
```

For the cumulative product purities $PU_i$ holds

```{math}
:label: purity
PU_{i} = \frac{m_{i}^{i}}{\sum_{l=1}^{n_{comp}} m_{l}^{i}},\\
```

where $n_{comp}$ is the number of mixture components and $m_{l}^{i}$ is the mass of component $l$ in target fraction $i$.

In **CADET-Process**, the {mod}`~CADETProcess.fractionation` module provides methods to calculate these performance indicators.


## Fractionator
The {class}`~CADETProcess.fractionation.Fractionator` allows slicing the solution and pool fractions for the individual components.
It enables evaluating multiple chromatograms at once and multiple fractions per component per chromatogram.

The most basic strategy is to manually set all fractionation times manually.
To demonstrate the strategy, consider a simple{ref}`batch-elution example<batch_elution_example>`.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.batch_elution.process import process
```

To enable the calculation of the process parameters, it is necessary to specify which of the inlets should be considered for the feed and eluent consumption.
Moreover, the outlet(s) which are used for evaluation need to be defined.

```
flow_sheet.add_feed_inlet('feed')
flow_sheet.add_eluent_inlet('eluent')
flow_sheet.add_product_outlet('outlet')
```

```{code-cell} ipython3
:tags: [remove-cell]

from CADETProcess.simulator import Cadet
process_simulator = Cadet()
simulation_results = process_simulator.simulate(process)
```

For reference, this is the chromatogram at the outlet that needs to be fractionated:

```{code-cell} ipython3
---
tags: [remove-input]
mystnb:
  figure:
    caption: |
      Concentration profile at column outlet.
    name: column_outlet
---

_ = simulation_results.solution.outlet.outlet.plot()
```

After import, the {class}`~CADETProcess.fractionation.Fractionator` is instantiated with the simulation results.

```{code-cell} ipython3
from CADETProcess.fractionation import Fractionator
fractionator = Fractionator(simulation_results)
```

To add a fractionation event, the following arguments need to be provided:
- `event_name`: Name of the event.
- `target`: Pool to which fraction is added. `-1` indicates waste.
- `time`: Time of the event
- `chromatogram`: Name of the chromatogram. Optional if only one outlet is set as `product_outlet`.

Here, component $A$ seems to have sufficient purity between $5 \colon 00~min$ and $5 \colon 45~min$ and component $B$ between $6 \colon 30~min$ and $9 \colon 00~min$.

```{code-cell} ipython3
fractionator.add_fractionation_event('start_A', 'A', 5*60, 'outlet')
fractionator.add_fractionation_event('end_A', -1, 5.75*60)
fractionator.add_fractionation_event('start_B', 'B', 6.5*60)
fractionator.add_fractionation_event('end_B', -1, 9*60)
```

The {class}`~CADETProcess.performance.Performance` object of the {class}`~CADETProcess.fractionation.Fractionator` contains the parameters:
```{code-cell} ipython3
print(fractionator.performance)
```
With these fractionation times, the both component fractions reach a purity of $99.7~\%$, and $97.2~\%$  respectively.
The recovery yields are $65.2~\%$ and $63.4~\%$.

The chromatogram can be plotted with the fraction times overlaid:
```{code-cell} ipython3
_ = fractionator.plot_fraction_signal()
```

## Optimization of Fractionation Times
The {mod}`~CADETProcess.fractionation` module also provides a method to set up an {class}`~CADETProcess.optimization.OptimizationProblem` which automatically determines optimal cut times.
For every component, different purity requirements can be specified, and any function may be applied as objective.

For the objective and constraint functions, fractions are pooled from all {class}`Outlets <CADETProcess.processModel.Outlet>` of the {class}`~CADETProcess.processModel.FlowSheet` (see equations {eq}`mass` and {eq}`purity`) that have been marked as `product_outlet`.
For more information about configuring the {class}`~CADETProcess.processModel.FlowSheet`, refer to {ref}`flow_sheet_guide`.

As initial values for the optimization, areas of the chromatogram with sufficient local purity are identified, i.e., intervals where $PU_i(t)=c_i(t)/\sum_j c_j(t)\geq PU_{min,i}$ {cite}`Shan2004`.
These initial intervals are then expanded by the optimizer towards regions of lower purity while meeting the cumulative purity constraints.
In the current implementation, {class}`~CADETProcess.optimization.COBYLA` {cite}`Powell1994` of the **SciPy** {cite}`SciPyContributors2020` library is used as optimizer.
Yet, any other solver or heuristic algorithm may be used.

```{code-cell} ipython3
from CADETProcess.fractionation import FractionationOptimizer
fractionation_optimizer = FractionationOptimizer()
```

By default, the mass of the components is maximized under purity constraints.
However, other objective functions can be used.

To automatically optimize the fractionation times, pass the simulation results to the {meth}`~CADETProcess.fractionation.FractionationOptimizer.optimize_fractionation` method.
Depending on the separation problem at hand, different purity requirements can be specified.
For example, here only the first component is relevant, and requires a purity $\ge 95~\%$:

```{code-cell} ipython3
fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0.95, 0])
```

The results are stored in a {class}`~CADETProcess.performance.Performance` object.

```{code-cell} ipython3
print(fractionator.performance)
```

The chromatogram can also be plotted with the fraction times overlaid:

```{code-cell} ipython3
_ = fractionator.plot_fraction_signal()
```

For comparison, this is the results if only the second component is relevant:

```{code-cell} ipython3
fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0, 0.95])

print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

But of course, also both components can be valuable.
Here, the required purity is also reduced to demonstrate that overlapping fractions are automatically avoided by internally introducing linear constraints.

```{code-cell} ipython3
fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0.8, 0.8])

print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

To set an alternative objective, a function needs to be passed that takes a {class}`~CADETProcess.performance.Performance` as an input.
In this example, not only the total mass is considered important but also the concentration of the fraction.
As previously mentioned, `COBYLA` only handles single objectives.
Hence, a {class}`~CADETProcess.performance.RankedPerformance` is used which transforms the {class}`~CADETProcess.performance.Performance` object by adding a weight $w_i$ to each component.

$$
p = \frac{\sum_i^{n_{comp}}w_i \cdot p_i}{\sum_i^{n_{comp}}(w_i)}
$$

It is important to remember that by default, objectives are minimized.
To register a function that is to be maximized, add the `minimize=False` flag.
Also, the number of objectives the function returns needs to be specified.

```{code-cell} ipython3
from CADETProcess.performance import RankedPerformance
ranking = [1, 1]
def alternative_objective(performance):
	performance = RankedPerformance(performance, ranking)
	return performance.mass * performance.concentration

fractionator = fractionation_optimizer.optimize_fractionation(
	simulation_results, purity_required=[0.95, 0.95],
	obj_fun=alternative_objective,
    n_objectives=1,
    minimize=False,
)

print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

The resulting fractionation times show that in this case, it is advantageous to discard some slices of the peak in order not to dilute the overall product fraction.

## Exclude Components
In some situations, not all components are relevant for fractionation.
For example, salt used for elution usually does not affect the purity of a component.
For this purpose, a subset of components can be specified.

To demonstrate the strategy, consider the {ref}`LWE example<lwe_example>`.
Here, the `Salt` component should not be used for fractionation.

```{code-cell} ipython3
:tags: [remove-cell]

from examples.load_wash_elute.lwe_flow_rate import process

from CADETProcess.simulator import Cadet
process_simulator = Cadet()
simulation_results = process_simulator.simulate(process)
```

```{code-cell} ipython3
fractionator = fractionation_optimizer.optimize_fractionation(
    simulation_results,
    components=['A', 'B', 'C'],
    purity_required=[0.95, 0.95, 0.95]
)
print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

## Sum species
Note that by default the sum-signal of all component {class}`~CADETProcess.processModel.Species` is used for fractionation.
To disable this feature, set `use_total_concentration_components=False`.
For more information on {class}`~CADETProcess.processModel.Species`, refer to {ref}`component_system_guide`.
