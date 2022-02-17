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

(optimization_tutorial)=
# Process optimization
One of the main features of **CADET-Process** is process optimization. 

an `OptimizationProblem` class is introduced that decouples the problem formulation from the `Optimizer` used for its solution (see {numref}`framework_overview`), allowing for a simple comparison of different optimization approaches.
The `OptimizationVariables` $x$ may refer to any attribute of the `ProcessModel`.
This includes model parameters, as well as `FlowSheet`-events.
As for the latter, not only the time when they are executed can be optimized, but also the value to which the attribute is changed when the `Event` is performed, allowing for structural optimization.
Bound constraints and linear constraints can limit the parameter space and the user is free to define arbitrary function $f(x)$ as the objective, and add arbitrary nonlinear constraint functions $c(x)$ to the `OptimizationProblem`.

The `OptimizerAdapter` provides a unified interface for using external optimization libraries.
It is responsible for converting the `OptimizationProblem` to the specific API of the external `Optimizer`.
Currently, adapters to pymoo {cite}`pymoo2020` and SciPy's optimization suite {cite}`SciPyContributors2020` are implemented, all of which are published under open source licenses that allow for academic as well as commercial use.

In order to facilitate the bi-level approach mentioned beforehand, an intermediate `ProcessEvaluator` is implemented.
It is responsible for passing the `ProcessModel` to the Simulator, checking Stationarity requirements and calling the `Fractionation` subroutine.
The `ProcessEvaluator` returns the `Performance` to the `OptimizationProblem` which in turn determines the value of the objective function at the current point.
The value is passed to the external `Optimizer`, then determines new candidates to be evaluated until a termination criterion is reached.

## Demonstration
In this example, 

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../')

from examples.operating_modes.batch_elution import process
```

First, the `CADET` is configured as usual. 
To also allow for cycle-to-cycle overlaps, cyclic stationarity is also considered.

```{code-cell} ipython3
from CADETProcess.simulation import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True
```

To evaluate the `Process`, again a `FractionationOptimizer` is used.
For the objective function, the mass is maximized. 
In this example, both components are equally valuable.
Hence, a `RankedPerformance` is used which combines the performance of multiple components into a single metric based on a ranking.
Moreover, the minimum required purity is set to $95~\%$ for both components.

```{code-cell} ipython3
from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.common import RankedPerformance

purity_required = [0.95, 0.95]
ranking = [1, 1]
def fractionation_objective(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass

fractionation_optimizer = FractionationOptimizer(purity_required, fractionation_objective)
```

To combine simulation and fractionation, a `ProcessEvaluator` is used. 
```{code-cell} ipython3
from CADETProcess.evaluator import SimulateAndFractionate
evaluator = SimulateAndFractionate(process_simulator, fractionation_optimizer)
```

Now, the `OptimizationProblem` is configured.
For this purpose, we pass the `Process`, as well as the `Evaluator` to the Optimization Problem.
```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem(process, evaluator)
```

First, the objective function is defined.
In this example, a product of mass, recovery yield, and eluent consumption is considered.
For this purpose, the `ProcessPerformance` object which the `SimulateAndFractionate` object returns again is ranked equally for both components. 
```{code-cell} ipython3
def objective_function(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass * performance.recovery * performance.eluent_consumption
optimization_problem.add_objective(objective_function)
```

In this example, we want to find the optimal cycle time and feed duration.
For this, we use the `add_variable` method. 
First, it takes the path to the variable in the evaluation object.
Additionally, lower and upper bounds can be specified.
```{code-cell} ipython3
optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)
```
Moreover, it is useful to add a linear constraint.
Linear constraints are usually defined in the following way

$$
A \cdot x \leq b
$$

In **CADET-Process**, add each row $a$ of the constraint function needs to be added individually.
The `add_linear_constraint` function takes the variables subject to the constraint as first argument.
$a$ and $b$ are then passed as second and third argument. 
It is important to note that the order in $a$ is inferred from the order in which the optimization variables are passed.
```{code-cell} ipython3
optimization_problem.add_linear_constraint(['feed_duration.time', 'cycle_time'], [1,-1], 0)
```

Before the optimization can be run, the `Solver` needs to be initialized and configured.
For this example, `NSGA2` is used, a genetic algorithm.
Important parameters are the population size (`pop_size`), the maximum number of generations `n_max_gen` and the number of cores (`n_cores`) which the optimization can use.

```{code-cell} ipython3
from CADETProcess.optimization import NSGA2

optimization_solver = NSGA2()
optimization_solver.pop_size = 200
optimization_solver.n_max_gen = 200
optimization_solver.n_cores = 4
```
To start the simulation, the `OptimizationProblem` needs to be passed to the `Solver`.
```
optimization_results = opt_solver.optimize(optimization_problem)
```
In the results, the information about the optimization results are stored such as best individual etc.

