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

(variable_dependencies_guide)=
# Variable Dependencies
In many optimization problems, a large number of variables must be considered simultaneously, leading to high complexity.
For more advanced problems, reducing the degrees of freedom can greatly simplify the optimization process and lead to faster convergence and better results.
One way to achieve this is to define dependencies between individual variables.

When defining dependencies, it is important that the optimizer is not exposed to them directly.
Instead, the dependencies should be integrated into the model in a way that is transparent to the optimizer.
Different mechanisms can be used to define dependencies, including linear combinations and custom functions.
With linear combinations, variables are combined using weights or coefficients, while custom functions allow for more complex relationships between variables to be defined.

For example, consider a process where the same parameter is used in multiple unit operations.
To reduce the number of variables that the optimizer needs to consider, it is possible to add a single variable, which is then set on both evaluation objects in pre-processing.
In other cases, the ratio between model parameters may be essential for the optimization problem.
For instance, consider the equilibrium constant $k_{eq} = k_a / k_d$ for an adsorption process with adsorption rate $k_a$ and desorption rate $k_d$.
Instead of exposing both $k_a$ and $k_d$ to the optimizer, it is usually beneficial to expose $k_a$ and $k_{eq}$.
This way, the values for the equilibrium and the kinetics of the reaction can be found independently.


```{figure} ./figures/transform_dependency.svg
:name: transform_dependency
```

For example, consider an {class}`~CADETProcess.optimization.OptimizationProblem` where the ratio of two variables should be considered as a third variable.
First, all variables need to be added.

```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('transform_demo')

optimization_problem.add_variable('var_0')
optimization_problem.add_variable('var_1')
optimization_problem.add_variable('var_2')
```

To add the dependency, the dependent variable needs to be specified, as well as a list of the independent variables.
Finally, a callable that takes the independent variables and returns the value of the dependent variable value needs to be added.
For more information refer to {meth}`~CADETProcess.optimization.OptimizationProblem.add_variable_dependency`.

```{code-cell} ipython3
def transform_fun(var_0, var_1):
    return var_0/var_1

optimization_problem.add_variable_dependency('var_2', ['var_0', 'var_1'], transform=transform_fun)
```

Note that generally bounds and linear constraints can still be specified independently for all variables.
