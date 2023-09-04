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

(variable_normalization_guide)=
# Variable Normalization
Most optimization algorithms struggle when optimization variables spread over multiple orders of magnitude.
This is important because the magnitude or range of the parameters can impact the optimization process, and the relative importance of each parameter can be difficult to determine without normalization.
Normalizing parameters makes the optimization process more efficient and helps ensure that the results are more accurate and reliable.
Additionally, normalization can prevent the optimization process from becoming biased towards certain parameters, which could lead to suboptimal or inefficient solutions.

**CADET-Process** provides several transformation methods which can help to soften these challenges.

In the following `x` will refer to the value exposed to the optimizer, whereas `variable` refers to the actual parameter value.

```{figure} ./figures/transform.svg
:name: transform
```

## Linear Normalization
The linear normalization maps the variable space from the lower and upper bound to a range between $0$ and $1$ by applying the following transformation:

$$
x^\prime = \frac{x - x_{lb}}{x_{ub} - x_{lb}}
$$

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('normalization_demo')
```

```{code-cell} ipython3
optimization_problem.add_variable('var_norm_lin', lb=-100, ub=100, transform='linear')
```

## Log Normalization
The log normalization maps the variable space from the lower and upper bound to a range between $0$ and $1$ by applying the following transformation:

$$
x^\prime = \frac{log \left( \frac{x}{x_{lb}} \right) }{log \left( \frac{x_{ub} }{x_{lb}} \right) }
$$

```{code-cell} ipython3
optimization_problem.add_variable('var_norm_log', lb=-100, ub=100, transform='log')
```

## Auto Transform
This transform will automatically switch between a linear and a log transform if the ratio of upper and lower bounds is larger than some value ($1000$ by default).


```{code-cell} ipython3
optimization_problem.add_variable('var_norm_auto', lb=-100, ub=100, transform='auto')
```
