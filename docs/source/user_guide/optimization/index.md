(optimization_guide)=
# Optimization
One of the main applications of **CADET-Process** is performing optimization studies.
Optimization refers to the selection of a solution with regard to some criterion.
In the simplest case, an optimization problem consists of minimizing some function $f(x)$ by systematically varying the input values $x$ and computing the value of that function.

$$
\min_x f(x)
$$

In the context of physico-chemical processes, examples for the application of optimization studies include scenarios such as process optimization (see {ref}`batch_elution_optimization_single`) and parameter estimation (see {ref}`fit_column_transport`).
Here, often many variables are subject to optimization, multiple criteria have to be balanced, and additional linear and nonlinear constraints need to be considered.

$$
\min_x f(x) \\

s.t. \\
    &g(x) \le 0, \\
    &h(x) = 0, \\
    &x \in \mathbb{R}^n \\
$$

where $g$ summarizes all inequality constraint functions, and $h$ equality constraints.


In the following, the optimization module of CADET-Process is introduced.
To decouple the problem formulation from the problem solution, two classes are provided:
An {class}`~CADETProcess.optimization.OptimizationProblem` class to specify optimization variables, objectives and constraints.
And an {class}`~CADETProcess.optimization.OptimizerBase` class which allows interfacing different external optimizers to solve the problem.

```{toctree}
:maxdepth: 2

optimization_problem
optimizer
```

## Advanced Configuration
```{toctree}
:maxdepth: 2

parallel_evaluation
evaluator
multi_objective_optimization
variable_dependencies
```
