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

## Adsorption rates example

For instance, consider an adsorption proces with an adsorption rate $k_a$ and a desorption rate $k_d$.
Both influence the strength of the interaction as well as the dynamics of the interaction.
By using the transformation $k_{eq} = k_a / k_d$ to calculate the equilibrium constant and $k_{kin} = 1 / k_d$ to calculate the kinetics constant, the values for the equilibrium and the kinetics of the reaction can be identified independently.
First, the dependent variables $k_a$ and $k_d$ must be added as they are implemented in the underlying model.

```{code-cell} ipython3
optimization_problem.add_variable(
    name='adsorption_rate',
    parameter_path='flow_sheet.column.binding_model.adsorption_rate',
    lb=1e-3, ub=1e3,
    transform='auto',
    indices=[1]  # modify only the protein (component index 1) parameter
)

optimization_problem.add_variable(
    name='desorption_rate',
    parameter_path='flow_sheet.column.binding_model.desorption_rate',
    lb=1e-3, ub=1e3,
    transform='auto',
    indices=[1]
)
```

Then, the independent variables $k_{eq}$ and $k_{kin}$ are added. To ensure, that CADET-Process does not try to write
these variables into the CADET-Core model, where they do not have a place, `evaluation_objects` is set to `None`.

```{code-cell} ipython3
optimization_problem.add_variable(
    name='equilibrium_constant',
    evaluation_objects=None,
    lb=1e-4, ub=1e3,
    transform='auto',
    indices=[1]
)

optimization_problem.add_variable(
    name='kinetic_constant',
    evaluation_objects=None,
    lb=1e-4, ub=1e3,
    transform='auto',
    indices=[1]
)
```

Lasty, the dependency between the variables is added with the `.add_variable_dependency()` method.

```{code-cell} ipython3
optimization_problem.add_variable_dependency(
    dependent_variable="desorption_rate",
    independent_variables=["kinetic_constant", ],
    transform=lambda k_kin: 1 / k_kin
)

optimization_problem.add_variable_dependency(
    dependent_variable="adsorption_rate",
    independent_variables=["kinetic_constant", "equilibrium_constant"],
    transform=lambda k_kin, k_eq: k_eq / k_kin
)
```
