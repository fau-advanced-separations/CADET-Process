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

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../')
```

(optimization_tutorial)=
# Process Optimization
One of the main features of **CADET-Process** is process optimization. 

An `OptimizationProblem` class is introduced that decouples the problem formulation from the `Optimizer` used for its solution (see {ref}`overview`), allowing for a simple comparison of different optimization approaches.
The `OptimizationVariables` $x$ may refer to any attribute of the `ProcessModel`.
This includes model parameters, as well as `FlowSheet`-events.
As for the latter, not only the time when they are executed can be optimized, but also the value to which the attribute is changed when the `Event` is performed, allowing for structural optimization.
Bound constraints and linear constraints can limit the parameter space and the user is free to define arbitrary function $f(x)$ as the objective, and add arbitrary nonlinear constraint functions $g(x)$ to the `OptimizationProblem`.

The `OptimizerAdapter` provides a unified interface for using external optimization libraries.
It is responsible for converting the `OptimizationProblem` to the specific API of the external `Optimizer`.
Currently, adapters to pymoo {cite}`pymoo2020` and SciPy's optimization suite {cite}`SciPyContributors2020` are implemented, all of which are published under open source licenses that allow for academic as well as commercial use.

## Optimization Problem
After import, the `OptimizationProblem` is initialized with a name.

```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem(name='batch elution')
```

Then, one or more `EvaluationObjects` can be added to the problem.
These are objects which contain the parameters which should be optimized (e.g. a `Process` or a `Fractionator`).

For this demonstration, the process model from {ref}`this example <batch_elution_example>` is used.

```{code-cell} ipython3
:tags: [remove-cell]
from examples.operating_modes.batch_elution import process
```

```{code-cell} ipython3
optimization_problem.add_evaluation_object(process)
```

In this example, we want to find the optimal cycle time and feed duration.
To specify these variable, the `add_variable` method is used.
First, it takes the path to the variable in the evaluation object.
Lower and upper bounds can be specified and it is possible to specify with which `EvaluationObject` the variable is associated.
By default, the variable is associated with all evaluation objects.

```{code-cell} ipython3
optimization_problem.add_variable('cycle_time', lb=10, ub=600, evaluation_objects=process)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)
```

Moreover, linear constraints can be added to the optimization problem.
In this example, it needs to be ensured that the cycle time is always greater than the feed duration.
Linear constraints are usually defined in the following way

$$
A \cdot x \leq b
$$

In **CADET-Process**, add each row $a$ of the constraint matrix needs to be added individually.
The `add_linear_constraint` function takes the variables subject to the constraint as first argument.
The left-hand side $a$ and the bound $b_a$ are passed as second and third argument. 
It is important to note that the column order in $a$ is inferred from the order in which the optimization variables are passed.

```{code-cell} ipython3
optimization_problem.add_linear_constraint(['feed_duration.time', 'cycle_time'], [1,-1], 0)
```

## Evaluation Toolchain
In many situations, some preprocessing steps are required before the objective function can be evaluated.

In the current example, to evaluate the performance of the process, the following steps need to be performed:
- Simulate the process until stationarity is reached.
- Determine fractionation times under purity constraints.
- Calculate objective functions; Here, two objectives are considered:
	- Productivity,
	- Yield recovery.

To implement these evaluation toolchains, **CADET-Process** provides a mechanism to add `Evaluators` to an `OptimizationProblem` which can be referenced by objective and constraint functions.
Any callable function can be added as `Evaluator`, assuming the first argument is the result of the previous step and it returns a single result object which is then processed by the next step.
Additional arguments and keyword arguments can be passed using `args` and `kwargs` when adding the `Evaluator`.
Optionally, the intermediate results can be cached when different objective and constraint functions require the same preprocessing steps.

To demonstrate this, a `ProcessSimulator` and a `FractionationOptimizer` are added to the `OptimizationProblem`.
First, `CADET` is configured as usual, and the assertion of cyclic stationarity is enabled.

```{code-cell} ipython3
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

optimization_problem.add_evaluator(process_simulator)
```

Note that storing all simulation results requires large amounts of disk space.
Here, only the fractionation results are cached since they will be used by both objectives.
If the evaluator requires additional arguments, they can be specified with `args` and `kwargs`.

```{code-cell} ipython3
from CADETProcess.fractionation import FractionationOptimizer
fractionation_optimization = FractionationOptimizer()

optimization_problem.add_evaluator(fractionation_optimization, cache=True, kwargs={'purity_required': [0.95, 0.95]})
```

When adding the objectives (or nonlinear constraint) functions, the evaluators are added as requirements.

```{code-cell} ipython3
from CADETProcess.performance import Recovery, Productivity

recovery = Recovery()
optimization_problem.add_objective(recovery, n_objectives=2, requires=[process_simulator, fractionation_optimization])
productivity = Productivity()
optimization_problem.add_objective(productivity, n_objectives=2, requires=[process_simulator, fractionation_optimization])
```

Now, when the objectives are evaluated, the process is only simulated once, and the optimal fractionation times only need to be determined once.

```{code-cell} ipython3
f = optimization_problem.evaluate_objectives([300, 60])
print(f"objectives: {f}")
```

## Optimizer
Before the optimization can be run, the `Optimizer` needs to be initialized and configured.
For this example, `U_NSGA3` is used, a genetic algorithm {cite}`Seada2016`.
Important parameters are the population size (`pop_size`), the maximum number of generations (`n_max_gen`) and the number of cores (`n_cores`) which can be used during optimization.

```{code-cell} ipython3
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
optimizer.pop_size = 100
optimizer.n_max_gen = 50
optimizer.n_cores = 4
```

To start the simulation, the `OptimizationProblem` needs to be passed to the `optimize()` method.
**CADET-Process** automatically stores checkpoints so that an optimization can be interrupted and restarted.
To load from an existing file, set `use_checkpoint=True`.

<!-- ```{code-cell} ipython3 -->
```
optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=True,
)
```

In the results, the information about the optimization results are stored such as best individual etc.

