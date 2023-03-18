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

(comparison_guide)=
# Comparing Simulation Results with Reference Data
The {mod}`CADETProcess.comparison` module in CADET-Process offers functionality to quantify the difference between simulations and references, such as other simulations or experiments.
The {class}`~CADETProcess.comparison.Comparator` class allows users to compare the outputs of two simulations or simulations with experimental data.
It provides several methods for visualizing and analyzing the differences between the data sets.
Users can choose from a range of metrics to quantify the differences between the two data sets, such as sum squared errors or shape comparison.

```{code-cell} ipython3
from CADETProcess.comparison import Comparator
comparator = Comparator()
```

## References
To properly work with **CADET-Process**, the experimental data needs to be converted to an internal standard.
The {mod}`CADETProcess.reference` module provides different classes for different types of experiments.
For in- and outgoing streams of unit operations, the {class}`~CADETProcess.reference.ReferenceIO` class must be used.

To demonstrate this module, consider a simple dextran pulse injection onto a chromatographic column.
The following (experimental) concentration profile is measured at the column outlet.
Consider that the time and the data of the experiment are stored in the variables `time_experiment`, and `dextran_experiment` respectively which are simply added to the constructor, together with a name for the reference.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
data = np.loadtxt('../../../../examples/parameter_estimation/reference_data/dextran.csv', delimiter=',')
time_experiment = data[:, 0]
dextran_experiment = data[:, 1]
```

```{code-cell} ipython3
from CADETProcess.reference import ReferenceIO
reference = ReferenceIO('dextran experiment', time_experiment, dextran_experiment)
```

Similarly to the {class}`~CADETProcess.solution.SolutionIO` class, the {class}`~CADETProcess.reference.ReferenceIO` class also provides a plot method:

```{code-cell} ipython3
_ = reference.plot()
```

To add the reference to the {class}`~CADETProcess.comparison.Comparator`, use the {meth}`~CADETProcess.comparison.Comparator.add_reference` method.

```{code-cell} ipython3
comparator.add_reference(reference)
```

## Difference Metrics
There are many metrics which can be used to quantify the difference between the simulation and the reference.
Most commonly, the sum squared error ({class}`~CADETProcess.comparison.SSE`) is used.
However, SSE is often not an ideal measurement for chromatography.
Because of experimental non-idealities like pump delays and fluctuations in flow rate there is a tendency for the peaks to shift in time.
This causes the optimizer to favour peak position over peak shape and can lead for example to an overestimation of axial dispersion.

In contrast, the peak shape is dictated by the physics of the physico-chemical interactions while the position can shift slightly due to systematic errors like pump delays.
Hence, a metric which prioritizes the shape of the peaks being accurate over the peak eluting exactly at the correct time is preferable.
For this purpose, **CADET-Process** offers a {class}`~CADETProcess.comparison.Shape` metric {cite}`Heymann2022`.
For an overview of all available difference metrics, refer to {mod}`CADETProcess.comparison`.

To add a difference metric, the following arguments need to be passed to the {meth}`~CADETProcess.comparison.Comparator.add_difference_metric` method:
- `difference_metric`: The type of the metric.
- `reference`: The reference which should be used for the metric.
- `solution_path`: The path to the corresponding solution in the simulation results.

```Python3
comparator.add_difference_metric('SSE', reference, 'column.outlet')
```

Optionally, a start and end time can be specified to only evaluate the difference metric at that slice.
This is particularly useful if system noise (e.g. injection peaks) should be ignored or if certain peaks correspond to certain components.

```{code-cell} ipython3
comparator.add_difference_metric(
    'SSE', reference, 'column.outlet', start=5*60, end=7*60
)
```

## Reference Model
Next to the experimental data, a reference model needs to be configured.
It must include relevant details s.t. it is capable of accurately predicting the experimental system (e.g. tubing, valves etc.).
For this example, the full process configuration can be found {ref}`here <dextran_pulse_example>`.

As an initial guess, the bed porosity is set to $0.4$, and the axial dispersion to $1.0 \cdot 10^{-7}$.
After process simulation, the {meth}`~CADETProcess.comparison.Comparator.evaluate` method is called with the simulation results.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.simulator import Cadet
simulator = Cadet()

from examples.parameter_estimation.reference_simulation.dextran_pulse import process
simulation_results = simulator.simulate(process)
```

```{code-cell} ipython3
metrics = comparator.evaluate(simulation_results)
print(metrics)
```

The difference can also be visualized:

```{code-cell} ipython3
_ = comparator.plot_comparison(simulation_results)
```

The comparison shows that there is still a large discrepancy between simulation and experiment.
Instead of manually adjusting these parameters, an {class}`~CADETProcess.optimization.OptimizationProblem` can be set up which automatically determines the parameter values.
For an example, see {ref}`` [todo]
