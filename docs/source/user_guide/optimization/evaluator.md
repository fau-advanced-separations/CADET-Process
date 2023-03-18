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

(evaluation_toolchains_guide)=
# Evaluation Toolchains
In the context of **CADET-Process**, "Evaluation Toolchains" refer to a sequence of preprocessing steps that are necessary to calculate performance indicators for a process, followed by the calculation of the objective function in an optimization problem.
The toolchains involve two types of objects: evaluation objects and evaluators.

(evaluation_objects_guide)=
## Evaluation Objects
In the context of **CADET-Process**, optimization variables usually represent attributes of a {class}`~CADETProcess.processModel.Process` such as model parameters values or event times, but also fractionation times of the {class}`~CADETProcess.fractionation.Fractionator` can be optimized.
or attributes of custom evaluation objects can be used as optimization variables.

An evaluation object is an object that manages the value of an optimization variable in an optimization problem.
It acts as an interface between the optimization problem and the object whose attribute(s) need to be optimized.
The evaluation object provides the optimization problem with the current value of the optimization variable, and when the optimization problem changes the value of the optimization variable, the evaluation object updates the attribute(s) of the associated object accordingly.

```{figure} ./figures/single_evaluation_object.svg
:name: single_evaluation_object
```

For this purpose, the evaluation object must implement a `parameter` property that returns a (potentially nested) dictionary with the current values of all model parameters, as well as a setter for that property.

```{note}
Currently, custom evaluation objects also need to provide a property for `polynomial_parameters`.
This will be improved in a future release.
For reference, see [here](https://github.com/fau-advanced-separations/CADET-Process/issues/20).
```

To associate an optimization variable with an evaluation object, the evaluation object must first be added to the optimization problem using {meth}`~CADETProcess.optimization.OptimizationProblem.add_evaluation_object`.
For demonstration purposes, consider a simple {ref}`batch-elution example<batch_elution_example>`.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('evaluation_object_demo')

from examples.batch_elution.process import process
```

```{code-cell} ipython3
optimization_problem.add_evaluation_object(process)
```

Note that multiple evaluation objects can be added, which for example allows for simultaneous optimization of multiple operating conditions.

When adding variables, it is now possible to specify with which evaluation object the variable is associated.
Moreover, the path to the variable in the evaluation object needs to be specified.

```{code-cell} ipython3
optimization_problem.add_variable('var_0', evaluation_objects=[process], parameter_path='flow_sheet.column.total_porosity', lb=0, ub=1)
```

By default, the variable is associated with all evaluation objects.
If no path is provided, the name is also used as path.
Hence, the variable definition can be simplified to:

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('evaluation_object_demo')
optimization_problem.add_evaluation_object(process)
```

```{code-cell} ipython3
optimization_problem.add_variable('flow_sheet.column.total_porosity', lb=0, ub=1)
```

To demonstrate the flexibility of this approach, consider two evaluation objects and two optimization variables where one variable is associated with a single evaluation object, and the other with both.

```{figure} ./figures/multiple_evaluation_objects.svg
:name: multiple_evaluation_objects
```

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('evaluation_object_demo_multi')

import copy

process_a = copy.deepcopy(process)
process_a.name = 'process_a'
process_b = copy.deepcopy(process)
process_b.name = 'process_b'
```

```{code-cell} ipython3
optimization_problem.add_evaluation_object(process_a)
optimization_problem.add_evaluation_object(process_b)
optimization_problem.add_variable('flow_sheet.column.total_porosity')
optimization_problem.add_variable('flow_sheet.column.length', evaluation_objects=[process_a])
```

(evaluators_guide)=
## Evaluators
In many cases, it is necessary to perform preprocessing steps before evaluating the objective function in an optimization problem.
For example, to calculate performance indicators of a process, several steps may be required, such as simulating the process until stationarity is reached, determining fractionation times under purity constraints, and calculating objective function values based on productivity and yield recovery.

To implement these evaluation toolchains, **CADET-Process** provides a mechanism to add `Evaluators` to an {class}`~CADETProcess.optimization.OptimizationProblem` which can be referenced by objective and constraint functions.
Any callable function can be added as `Evaluator`, assuming the first argument is the result of the previous step and it returns a single result object which is then processed by the next step.
Additional arguments and keyword arguments can be passed using `args` and `kwargs` when adding the `Evaluator`.
The intermediate results are also automatically cached when different objective and constraint functions require the same preprocessing steps.

```{figure} ./figures/single_objective_evaluators.svg
:name: single_objective_evaluators
```

Consider the following example:

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('evaluator_demo')
optimization_problem.add_variable('x')
```
To add the evaluator, use {meth}`~CADETProcess.optimization.OptimizationProblem.add_evaluator`.

```{code-cell} ipython3
def evaluator(x):
    print(f'Running evaluator with {x}')
    intermed_result = x**2
    return intermed_result

optimization_problem.add_evaluator(evaluator)
```

This evaluator can now be referenced when adding objectives, nonlinear constraints, or callbacks.
For this purpose, add the required evaluators (in order) to the corresponding method (here, {meth}`~CADETProcess.optimization.OptimizationProblem.add_objective`).

```{code-cell} ipython3
def objective(intermed_result):
    print(f'Running objective with {intermed_result}')
    return intermed_result**2

optimization_problem.add_objective(objective, requires=evaluator)
```

When evaluating objectives, the evaluator is also called.

```{code-cell} ipython3
optimization_problem.evaluate_objectives(1)
```

Intermediate results are automatically cached s.t. other objectives or constraints that require the same evaluation steps do not need to recompute the pre-processing steps.
