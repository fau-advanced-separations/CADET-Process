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

from pathlib import Path
import sys

root_dir = Path('../../../../').resolve()
sys.path.append(root_dir.as_posix())

examples_path = root_dir / 'examples' / 'characterize_chromatographic_system'
sys.path.append(examples_path.as_posix())

import shutil
shutil.rmtree('./experimental_data/', ignore_errors=True)
shutil.copytree(examples_path / 'experimental_data', './experimental_data/')
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

To demonstrate this module, consider a simple tracer pulse injection onto a chromatographic column.
The following (experimental) concentration profile is measured at the column outlet.
Consider that the time and the data of the experiment are stored in the variables `time_experiment`, and `c_experiment` respectively which are simply added to the constructor, together with a name for the reference.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
data = np.loadtxt('../../../../examples/characterize_chromatographic_system/experimental_data/non_pore_penetrating_tracer.csv', delimiter=',')
time_experiment = data[:, 0]
c_experiment = data[:, 1]
```

```{code-cell} ipython3
from CADETProcess.reference import ReferenceIO
reference = ReferenceIO('c experiment', time_experiment, c_experiment)
```

Similarly to the {class}`~CADETProcess.solution.SolutionIO` class, the {class}`~CADETProcess.reference.ReferenceIO` class also provides a plot method:

```{code-cell} ipython3
fig, axes = reference.plot()
assert np.all(np.isfinite(reference.solution))
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

Construct the metric with the reference and pass the instance along with the solution path to {meth}`~CADETProcess.comparison.Comparator.add_difference_metric`.
The `solution_path` identifies the corresponding outlet in the simulation results.

```{code-cell} ipython3
from CADETProcess.comparison import SSE
metric = SSE(reference)
comparator.add_difference_metric(metric, 'column.outlet')
```

Optionally, a start and end time can be specified to only evaluate the difference metric at that slice.
This is particularly useful if system noise (e.g. injection peaks) should be ignored or if certain peaks correspond to certain components.

```{code-cell} ipython3
from CADETProcess.comparison import SSE
metric = SSE(reference, start=3*60, end=6*60)
comparator.add_difference_metric(metric, 'column.outlet')
```

## Reference Model
Next to the experimental data, a reference model needs to be configured.
It must include relevant details s.t. it is capable of accurately predicting the experimental system (e.g. tubing, valves etc.).
For this example, the full process configuration can be found {ref}`here <fit_column_transport>`.

As an initial guess, the bed porosity is set to $0.5$, and the axial dispersion to $1.0 \cdot 10^{-7}$.
After process simulation, the {meth}`~CADETProcess.comparison.Comparator.evaluate` method is called with the simulation results.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.simulator import Cadet
simulator = Cadet()

from column_transport_parameters import process

process.flow_sheet.column.bed_porosity = 0.5
process.flow_sheet.column.axial_dispersion = 1e-7

simulation_results = simulator.simulate(process)
```

```{code-cell} ipython3
metrics = comparator.evaluate(simulation_results)
assert len(metrics) == comparator.n_metrics
assert np.all(np.isfinite(metrics))
print(metrics)
```

The difference can also be visualized:

```{code-cell} ipython3
fig, axes = comparator.plot_comparison(simulation_results)
assert len(axes) == comparator.n_difference_metrics
assert all(len(ax.lines) >= 2 for ax in axes)
```

The comparison shows that there is still a large discrepancy between simulation and experiment.
Instead of manually adjusting these parameters, an {class}`~CADETProcess.optimization.OptimizationProblem` can be set up which automatically determines the parameter values.
For an example, see {ref}`fit_column_transport`.

```{code-cell} ipython3
:tags: [remove-cell]

shutil.rmtree('./experimental_data/', ignore_errors=True)
```


## Comparing collected fractions

Offline measurements describe the average concentration in each collection window.
{class}`~CADETProcess.reference.FractionationReference` stores these windows together with measured amounts and volumes.
{class}`~CADETProcess.comparison.FractionationNRMSE` integrates the simulated outlet over the same windows and compares their flow-weighted concentrations.
Each fraction has equal weight in the RMSE, and each component's RMSE is divided by its maximum measured fraction concentration.
{class}`~CADETProcess.comparison.FractionationSSE` remains available when an unnormalized sum of squared errors is desired.

A continuous {class}`~CADETProcess.reference.ReferenceIO` already inherits {meth}`~CADETProcess.solution.SolutionIO.create_fraction`.
The returned fraction's `concentration` is its integrated amount divided by its collected volume, using the signal's flow rate.
Use consistent units for measured amounts, volumes and simulated concentrations.
Select the desired collection windows in the reference; `FractionationNRMSE` rejects trace slicing with `start` or `end`.
Normalization follows `NRMSE`, including division by zero for an all-zero reference component; select components with a meaningful positive concentration scale.

This example creates one synthetic offline measurement from the simulated outlet, with a deliberately 10 % larger measured amount.
In practice, supply the measured amount, volume and collection times instead.

```{code-cell} ipython3
from CADETProcess.comparison import FractionationNRMSE
from CADETProcess.fractionation import Fraction
from CADETProcess.reference import FractionationReference

outlet = simulation_results.solution.column.outlet
collected = outlet.create_fraction(outlet.time[0], outlet.time[-1])
measured_fraction = Fraction(
    mass=1.1 * collected.mass,
    volume=collected.volume,
    start=collected.start,
    end=collected.end,
)
fraction_reference = FractionationReference(
    'offline measurement', [measured_fraction],
    component_system=outlet.component_system,
)
fraction_metric = FractionationNRMSE(fraction_reference)
fraction_comparator = Comparator()
fraction_comparator.add_difference_metric(fraction_metric, 'column.outlet')
fraction_scores = fraction_comparator.evaluate(simulation_results)
np.testing.assert_allclose(fraction_scores, [1 / 11], rtol=1e-6)
print(fraction_scores)
```
