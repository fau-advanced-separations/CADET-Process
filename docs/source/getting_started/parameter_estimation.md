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

(parameter_estimation_tutorial)=
# Parameter Estimation
One important aspect in modelling is parameter estimation.
For this purpose, model parameters are varied until the simulated output matches some reference (usually experimental data). 
To quantify the difference between simulation and reference, **CADET-Process** provides a `Comparator` module.

To demonstrate this module, consider a simple dextran pulse injection onto a chromatographic column. 
The following (experimental) concentration profile is measured at the column outlet.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
data = np.loadtxt('../../../examples/parameter_estimation/reference_data/dextran.csv', delimiter=',')
time_experiment = data[:, 0]
dextran_experiment = data[:, 1]

import matplotlib.pyplot as plt
_ = plt.plot(time_experiment, dextran_experiment)
```

The goal is to determine the bed porosity of the column, as well the axial dispersion.
Other process parameters like the column geometry and particle sizes are assumed to be known.

To instantiate the `Comparator` import the `comoparison` module.

```{code-cell} ipython3
from CADETProcess.comparison import Comparator
comparator = Comparator()
```

## References
To properly work with **CADET-Process**, the experimental data needs to be converted to an internal standard.
The `reference` module provides different classes for different types of experiments.
For in- and outgoing streams of unit operations, the `ReferenceIO` class must be used.

Consider that the time and the data of the experiment are stored in the variables `time_experiment`, and `dextran_experiment` respectively which are simply added to the constructor, together with a name for the reference.

```{code-cell} ipython3
import numpy as np

# Setup Reference
from CADETProcess.reference import ReferenceIO

reference = ReferenceIO('dextran experiment', time_experiment, dextran_experiment)
```

Similarly to the `SolutionIO` class, the `ReferenceIO` class also provides a plot method:

```{code-cell} ipython3
_ = reference.plot()
```

To add the reference to the comparator, use the `add_reference` method.

```{code-cell} ipython3
comparator.add_reference(reference)
```

## Difference Metrics
There are many metrics which can be used to quantify the difference between the simulation and the reference.
Most commonly, the sum squared error (SSE) is used.
However, SSE is often not an ideal measurement for chromatography.
Because of experimental non-idealities like pump delays and fluctuations in flow rate there is a tendency for the peaks to shift in time.
This causes the optimizer to favour peak position over peak shape and can lead for example to an overestimation of axial dispersion.

In contrast, the peak shape is dictated by the physics of the physico-chemical interactions while the position can shift slightly due to systematic errors like pump delays.
Hence, a metric which prioritizes the shape of the peaks being accurate over the peak eluting exactly at the correct time is preferable.
For this purpose, **CADET-Process** offers a `Shape` metric {cite}`Heymann2022`.

To add a difference metric, the following arguments need to be passed to the `add_difference_metric` method:
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
After process simulation, the `evaluate` method is called with the simulation results.

```{code-cell} ipython3
:tags: [remove-cell]

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
Instead of manually adjusting these parameters, an `OptimizationProblem` can be set up which automatically determines the parameter values.

## Setup as Optimization Problem
As in the previous example, the process is added as an `EvaluationObject`, and `Cadet` is added as `Evaluator`.
Moreover, bed porosity, and axial dispersion are added as optimization variables with some reasonable bounds.


```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('dextran')

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable('flow_sheet.column.bed_porosity', lb=0, ub=1)
optimization_problem.add_variable(
    'flow_sheet.column.axial_dispersion', lb=1e-8, ub=1e-5
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    requires=[simulator]
)

```
