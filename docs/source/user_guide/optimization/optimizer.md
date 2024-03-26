---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
execution:
  timeout: 600
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```

(optimizer_guide)=
# Optimizer
The {class}`~CADETProcess.optimization.OptimizerBase` provides a unified interface for using external optimization libraries.
It is responsible for converting the {class}`~CADETProcess.optimization.OptimizationProblem` to the specific API of the external optimizer.
Currently, adapters to {class}`Scipy's <CADETProcess.optimization.SciPyInterface>` optimization suite {cite}`SciPyContributors2020`, {class}`Pymoo <CADETProcess.optimization.PymooInterface>` {cite}`pymoo2020` and {class}`Ax <CADETProcess.optimization.AxInterface>` {cite}`ax` are implemented, all of which are published under open source licenses that allow for academic as well as commercial use.

Before the optimization can be run, the optimizer needs to be initialized and configured.
All implementations share the following options:

- {attr}`~CADETProcess.optimization.OptimizerBase.progress_frequency`: Number of generations after which optimizer reports progress.
- {attr}`~CADETProcess.optimization.OptimizerBase.n_cores`: The number of cores that the optimizer should use.
- {attr}`~CADETProcess.optimization.OptimizerBase.cv_tol`: Tolerance for constraint violation.
- {attr}`~CADETProcess.optimization.OptimizerBase.similarity_tol`: Tolerance for individuals to be considered similar.

The individual optimizer implementations provide additional options, including options for termination criteria.
For this example, {class}`~CADETProcess.optimization.U_NSGA3` is used, a genetic algorithm {cite}`Seada2016`.
It has the following additional options:

- {attr}`~CADETProcess.optimization.U_NSGA3.pop_size`: Number of individuals per generation.
- {attr}`~CADETProcess.optimization.U_NSGA3.n_max_gen`: Maximum number of generations.

All options can be displayed the following way:

```{code-cell} ipython3
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
print(optimizer.options)
```
For more information, refer to the reference of the individual implementations (see {mod}`~CADETProcess.optimization`).

To start the optimization, the {class}`~CADETProcess.optimization.OptimizationProblem` needs to be passed to the {meth}`~CADETProcess.optimization.OptimizerBase.optimize()` method.
Note that before running the optimization, a {meth}`check <CADETProcess.optimization.OptimizerBase.check_optimization_problem>` method is called to verify that checks whether the given optimization problem is configured correctly and supported by the optimizer.

Consider this bi-objective function:

$$
f(x) = \begin{bmatrix}x_1^2 + x_2^2, (x_1-1)^2 + x_2^2\end{bmatrix}
$$

Note, by default, results are stored to file.
To disable this, set `save_results=False`.

```{code-cell} ipython3
:tags: [hide-input]

from CADETProcess.optimization import OptimizationProblem

def multi_objective_func(x):
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 1)**2 + x[1]**2
    return f1, f2

optimization_problem = OptimizationProblem('moo')

optimization_problem.add_variable('x_0', lb=-5, ub=5)
optimization_problem.add_variable('x_1', lb=-5, ub=5)

optimization_problem.add_objective(multi_objective_func, n_objectives=2)
```

```{code-cell} ipython3
:tags: [scroll-output]

optimizer.n_cores = 4
optimization_results = optimizer.optimize(optimization_problem, save_results=False)
```

## Initial values
By default, the optimizer automatically tries to generate initial values using the {meth}`~CADETProcess.optimization.OptimizationProblem.create_initial_values` method provided by the {class}`~CADETProcess.optimization.OptimizationProblem` (see {ref}`initial_values_creation_guide`).
To manually specify initial values, pass `x0` to the method.

```python
optimization_results = optimizer.optimize(
    optimization_problem,
    x0=[[0, 1], [1,2]]
)
```

```{Note}
If the optimizer requires additional starting values beyond the ones provided, it will generate new individuals automatically.
Conversely, if more individuals are provided than necessary, the optimizer will ignore the excess values.
```


## Checkpoint
**CADET-Process** automatically stores checkpoints so that an optimization can be interrupted and restarted.
To load from an existing file, set `use_checkpoint=True`.

```python
optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=True
)
```

## Optimization Results
The {class}`~CADETProcess.optimization.OptimizationResults` object contains the results of the optimization.
This includes:
- {attr}`~CADETProcess.optimization.OptimizationResults.exit_flag`: Information about the optimizer termination.
- {attr}`~CADETProcess.optimization.OptimizationResults.exit_message`: Additional information about the optimiz status.
- {attr}`~CADETProcess.optimization.OptimizationResults.n_evals`: Number of evaluations.
- {attr}`~CADETProcess.optimization.OptimizationResults.x`: Optimal points.
- {attr}`~CADETProcess.optimization.OptimizationResults.f`: Optimal objective values.
- {attr}`~CADETProcess.optimization.OptimizationResults.g`: Optimal nonlinear constraint values.

Moreover, multiple plot methods are provided to visualize the results.
The {meth}`~CADETProcess.optimization.OptimizationResults.plot_objectives` method shows the values of all objectives as a function of the input variables using a colormap where later generations are plotted with darker blueish colors.
Invalid points, i.e. points where nonlinear constraints are not fulfilled, are also plotted using reddish colors, where also darker shades represent later generations.

```{code-cell} ipython3
:tags: [remove-cell]

# Restore matplotlib backend
%matplotlib inline
```

```{code-cell} ipython3
optimization_results.plot_objectives()
```

The {meth}`~CADETProcess.optimization.OptimizationResults.plot_pareto` method shows a pairwise Pareto plot, where each objective is plotted against every other objective in a scatter plot, allowing for a visualization of the trade-offs between the objectives.

```{code-cell} ipython3
optimization_results.plot_pareto()
```

The {meth}`~CADETProcess.optimization.OptimizationResults.plot_convergence` method is a tool for visualizing the convergence of the optimization over time, where the objective value is plotted against the number of function evaluations.

```{code-cell} ipython3
_ = optimization_results.plot_convergence()
```

The {meth}`~CADETProcess.optimization.OptimizationResults.plot_corner` method plots each evaluated variable value against every other variable in a set of scatter plots.
The corner plot is particularly useful when exploring high-dimensional data or parameter spaces, as it allows us to identify correlations and dependencies between variables, and to visualize the marginal distributions of each variable.
It is also useful when we want to compare the distribution of variables across different subsets of the data or parameter space.

```{code-cell} ipython3
optimization_results.plot_corner()
```

```{code-cell} ipython3
:tags: [remove-cell]

import os
from glob import glob
from shutil import rmtree

path = os.getcwd()
results_pattern = os.path.join(path, "results_*")
diskcache_pattern = os.path.join(path, "diskcache_*")

for item in glob(results_pattern) + glob(diskcache_pattern):
    if not os.path.isdir(item):
        continue
    rmtree(item)
```
