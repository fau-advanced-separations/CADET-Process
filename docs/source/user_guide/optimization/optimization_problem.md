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

(optimization_problem_guide)=
# Optimization Problem
The {class}`~CADETProcess.optimization.OptimizationProblem` class is used to specify optimization variables, objectives and constraints.
To instantiate a {class}`~CADETProcess.optimization.OptimizationProblem`, name needs to be passed as argument.

```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('single_objective')
```

By default, the {class}`~CADETProcess.optimization.OptimizationProblem` uses a [DiskCache](https://grantjenks.com/docs/diskcache/) to store intermediate evaluation results.
In contrast to using a simple python dictionary, this also allows for multi-core parallelization.

(optimization_variables_guide)=
## Optimization Variables
Any number of variables can be added to the {class}`~CADETProcess.optimization.OptimizationProblem`.
To add a variable, use the {meth}`~CADETProcess.optimization.OptimizationProblem.add_variable` method.
The first argument is the name of the variable.
Optionally, lower and upper bounds can be specified.

```{code-cell} ipython3
optimization_problem.add_variable('var_0', lb=0, ub=100)
```

Note that for many situations, it can be advantageous to normalize the optimization variables to ensure that all parameters are on a similar scale.
For more information, refer to {ref}`parameter_normalization_guide`.

The total number of variables is stored in {attr}`~CADETProcess.optimization.OptimizationProblem.n_variables` and the names in {attr}`~CADETProcess.optimization.OptimizationProblem.variable_names`.

```{code-cell} ipython3
print(optimization_problem.n_variables)
print(optimization_problem.variable_names)
```

In order to reduce the complexity of the optimization problem, dependencies between individual variables can be defined.
For more information, refer to {ref}`variable_dependencies_guide`.


(objectives_guide)=
## Objectives
Any callable function that takes an input $x$ and returns objectives $f$ can be added to the {class}`~CADETProcess.optimization.OptimizationProblem`.
Consider a quadratic function which expects a single input and returns a single output:

```{code-cell} ipython3
def objective(x):
	return x**2
```

To add this function as objective, use the {meth}`~CADETProcess.optimization.OptimizationProblem.add_objective` method.

```{code-cell} ipython3
optimization_problem.add_objective(objective)
```

It is also possible to use **CADET-Process** for multi-objective optimization problems.
Consider the following function which takes two inputs and returns two outputs.

$$
f(x,y) = [x^2 + y^2, (x-2)^2 + (y-2)^2]
$$


Either two callables are added as objective

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('multi_objective')
optimization_problem.add_variable('x', lb=0, ub=10)
optimization_problem.add_variable('y', lb=-10, ub=10)
```

```{code-cell} ipython3
import numpy as np

def objective_1(x):
	return x[0]**2 + x[1]**2

optimization_problem.add_objective(objective_1)

def objective_2(x):
	return (x[0] - 2)**2 + (x[1] - 2)**2

optimization_problem.add_objective(objective_2)
```

Alternatively, a single function that returns both objectives can be added.
In this case, the number of objectives the function returns needs to be specified by adding `n_objectives` as argument.

```{code-cell} ipython3
:tags: [remove-cell]

optimization_problem = OptimizationProblem('multi_objective')
optimization_problem.add_variable('x', lb=0, ub=10)
optimization_problem.add_variable('y', lb=-10, ub=10)
```

```{code-cell} ipython3
def multi_objective(x):
	f_1 = x[0]**2 + x[1]**2
	f_2 = (x[0] - 2)**2 + (x[1] - 2)**2
	return np.hstack((f_1, f_2))

optimization_problem.add_objective(multi_objective, n_objectives=2)
```

In both cases, the total number of objectives is stored in {attr}`~CADETProcess.optimization.OptimizationProblem.n_objectives`.

```{code-cell} ipython3
optimization_problem.n_objectives
```

For more information on multi-objective optimization, also refer to {ref}`moo_guide`.

The objective(s) can be evaluated with the {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_objectives` method.

```{code-cell} ipython3
optimization_problem.evaluate_objectives([1, 1])
```

It is also possible to evaluate multiple sets of input variables at once by passing a 2D array to the {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_objectives_population` method.

```{code-cell} ipython3
optimization_problem.evaluate_objectives_population([[0, 1], [1, 1], [2, -1]])
```

For more complicated scenarios that require (multiple) preprocessing steps, refer to {ref}`evaluation_toolchains_guide`.


## Linear constraints
Linear constraints are a common way to restrict the feasible region of an optimization problem.
They are typically defined using linear functions of the optimization:

$$
A \cdot x \leq b,
$$

where $A$ is an $m \times n$ coefficient matrix and $b$ is an $m$-dimensional vector and $m$ denotes the number of constraints, and $n$ the number of variables, respectively.

In **CADET-Process**, each row $a$ of the constraint matrix needs to be added individually.
The {meth}`~CADETProcess.optimization.OptimizationProblem.add_linear_constraint` function takes the variables subject to the constraint as first argument.
The left-hand side $a$ and the bound $b_a$ are passed as second and third argument.
It is important to note that the column order in $a$ is inferred from the order in which the optimization variables are passed.

For example, consider the following linear inequalities as constraints:

$$
x_1 + x_3 ≤ 4, \\
2x_2 – x_3 ≥ –2, \\
x_1 – x_2 + x_3 – x_4 ≥ 9.
$$

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('linear_constraints')

optimization_problem.add_variable('var_1')
optimization_problem.add_variable('var_2')
optimization_problem.add_variable('var_3')
optimization_problem.add_variable('var_4')
```

```{code-cell} ipython3
optimization_problem.add_linear_constraint(['var_1', 'var_3'], [1, 1], 4)
optimization_problem.add_linear_constraint(['var_2', 'var_3'], [-2, 1], 2)
optimization_problem.add_linear_constraint(['var_1', 'var_2', 'var_3', 'var_4'], [-1, 1, -1, 1], -9)
```

The combined coefficient matrix $A$ is stored in the attribute {attr}`~CADETProcess.optimization.OptimizationProblem.A`, and the right-hand side vector $b$ in {attr}`~CADETProcess.optimization.OptimizationProblem.b`.
To evaluate linear constraints, use {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_linear_constraints`.

```{code-cell} ipython3
optimization_problem.evaluate_linear_constraints([0, 0, 0, 0])
```

Any value larger than $0$ means the constraint is not met.
Alternatively, use {meth}`~CADETProcess.optimization.OptimizationProblem.check_linear_constraints` which returns `True` if all constraints are met (`False` otherwise).

```{code-cell} ipython3
optimization_problem.check_linear_constraints([0, 0, 0, 0])
```

(nonlinear_constraints_guide)=
## Nonlinear constraints
In addition to linear constraints, nonlinear constraints can be added to the optimization problem.
To add nonlinear constraints, use the {meth}`~CADETProcess.optimization.OptimizationProblem.add_nonlinear_constraint` method.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('nonlinear_constraints')
optimization_problem.add_variable('x', lb=0, ub=10)
optimization_problem.add_variable('y', lb=0, ub=10)
```

Consider this function which returns the product of all input values.

```{code-cell} ipython3
def nonlincon(x):
    return np.prod(x)
```

To add this function as nonlinear constraint, use the {meth}`~CADETProcess.optimization.OptimizationProblem.add_nonlinear_constraint` method.

```{code-cell} ipython3
optimization_problem.add_nonlinear_constraint(nonlincon)
```

As with objectives, it is also possible to add multiple nonlinear constraint functions or a function that returns more than a single value.
For the latter case, add `n_nonlinear_constraints` to the method.

To evaluate the nonlinear constraints, use {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_nonlinear_constraints`.

```{code-cell} ipython3
optimization_problem.evaluate_nonlinear_constraints([0.5, 0.5])
```

Alternatively, use {meth}`~CADETProcess.optimization.OptimizationProblem.check_nonlinear_constraints` which returns `True` if all constraints are met (`False` otherwise).

```{code-cell} ipython3
optimization_problem.check_nonlinear_constraints([0.5, 0.5])
```

By default, any value larger than $0$ means the constraint is not met.
If other bounds are required, they can be specified when registering the function.
For example if the product must be smaller than 1, set `bounds=1` when adding the constraint.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('nonlinear_constraints')
optimization_problem.add_variable('x', lb=0, ub=10)
optimization_problem.add_variable('y', lb=0, ub=10)
```

```{code-cell} ipython3
optimization_problem.add_nonlinear_constraint(nonlincon, bounds=1)
```

Note that {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_nonlinear_constraints` still returns the same value.
```{code-cell} ipython3
optimization_problem.evaluate_nonlinear_constraints([0.5, 0.5])
```

To only compute the constraint violation, use {meth}`~CADETProcess.optimization.OptimizationProblem.evaluate_nonlinear_constraints_violation`.

```{code-cell} ipython3
optimization_problem.evaluate_nonlinear_constraints_violation([0.5, 0.5])
```

In this context, a negative value denotes that the constraint is met.

The {meth}`~CADETProcess.optimization.OptimizationProblem.check_nonlinear_constraints` method also takes bounds into account.

```{code-cell} ipython3
optimization_problem.check_nonlinear_constraints([0.5, 0.5])
```

For more complicated scenarios that require (multiple) preprocessing steps, refer to {ref}`evaluation_toolchains_guide`.

(initial_values_creation_guide)=
## Initial Values
Initial values in optimization refer to the starting values of the decision variables for the optimization algorithm.
These values are typically set by the user and serve as a starting point for the optimization algorithm to begin its search for the optimal solution.
The choice of initial values can have a significant impact on the success of the optimization, as a poor choice may lead to the algorithm converging to a suboptimal solution or failing to converge at all.
In general, good initial values should be as close as possible to the true optimal values, and should take into account any known constraints or bounds on the decision variables.

To facilitate the definition of starting points, the {class}`~CADETProcess.optimization.OptimizationProblem` provides the {meth}`~CADETProcess.optimization.OptimizationProblem.create_initial_values`.

```{note}
This method only works if all optimization variables have defined lower and upper bounds.

Moreover, this method only guarantees that linear constraints are fulfilled.
Any nonlinear constraints may not be satisfied by the generated samples, and nonlinear parameter dependencies can be challenging to incorporate.
```

```{code-cell} ipython3
:tags: [hide-input]

optimization_problem = OptimizationProblem('linear_constraints')

optimization_problem.add_variable('x_0', lb=-10, ub=10)
optimization_problem.add_variable('x_1', lb=-10, ub=10)

optimization_problem.add_linear_constraint(['x_0', 'x_1'], [-1, 1], 0)
```

By default, the method returns a random point from the feasible region of the parameter space.

```{code-cell} ipython3
optimization_problem.create_initial_values()
```

Alternatively, the Chebyshev center of the polytope can be computed, which is the center of the largest Euclidean ball that is fully contained within that polytope.

```{code-cell} ipython3
optimization_problem.create_initial_values(method='chebyshev')
```

It is also possible to generate multiple samples at once.
For this purpose, [hopsy](https://modsim.github.io/hopsy/) is used to efficiently (uniformly) sample the parameter space.

```{code-cell} ipython3
x = optimization_problem.create_initial_values(n_samples=1000)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
fig.tight_layout()
```

## Callbacks
A callback function is a user defined function that is called periodically by the optimizer in order to allow the user to query the state of the optimization.
For example, a simple user callback function might be used to plot results.

All callback functions will be called with the best individual or with every member of the Pareto front in case of multi-objective optimization.
For more information on controlling the members of the Pareto front, refer to {ref}`mcdm_guide`.

```{figure} ./figures/callbacks.svg
:name: callbacks
```

The callback signature may include any of the following arguments:
- results : obj
    x or final result of evaluation toolchain.
- individual : {class}`~CADETProcess.optimization.Individual`, optional
    Information about current step of optimzer.
- evaluation_object : obj, optional
    Current evaluation object.
- callbacks_dir : Path, optional
    Path to store results.

Introspection is used to determine which of the signatures above to invoke.

```{code-cell} ipython3
def callback(individual, evaluation_object, callbacks_dir):
    print(individual.x, individual.f)
    print(evaluation_object)
    print(callbacks_dir)
```

For more information about evaluation toolchains, refer to {ref}`evaluation_toolchains_guide`.

To add the function to the {class}`~CADETProcess.optimization.OptimizationProblem`, use the {meth}`~CADETProcess.optimization.OptimizationProblem.add_callback` method.

```{code-cell} ipython3
optimization_problem.add_callback(callback)
```

By default, the callback is called after every iteration/generation.
To change the frequency, add a `frequency` argument which denotes the number of iterations/generations after which the callback function(s) is/are evaluated.

```{code-cell} ipython3
:tags: [remove-cell]

import os
from glob import glob
from shutil import rmtree

path = os.getcwd()
pattern = os.path.join(path, "results_*")

for item in glob(pattern):
    if not os.path.isdir(item):
        continue
    rmtree(item)
```
