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

(evaluation_pipeline_guide)=
# Evaluation Pipeline

Optimization often requires preprocessing steps before an objective or constraint can be computed.
For example, calculating process performance may involve simulating the process, determining fractionation times under purity constraints, and then computing yield and productivity from the fractionation result.

**CADET-Process** represents these steps as a directed acyclic graph (DAG) of evaluators.
Each evaluator is a named callable that receives the output of upstream evaluators and produces a named result.
Objectives and constraints declare which evaluator outputs they require; the pipeline ensures each evaluator runs exactly once per evaluation, regardless of how many objectives or constraints depend on it.

```{figure} ./figures/single_objective_evaluators.svg
:name: single_objective_evaluators
```

(evaluation_objects_guide)=
## Evaluation Objects

Optimization variables usually represent attributes of a {class}`~CADETProcess.processModel.Process`, such as model parameter values or event times, but any Python object with gettable and settable attributes can serve as an evaluation object.

```{figure} ./figures/single_evaluation_object.svg
:name: single_evaluation_object
```

To associate variables with an evaluation object, add it to the optimization problem first.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('evaluation_object_demo')

from examples.batch_elution.process import process
```

```{code-cell} ipython3
optimization_problem.add_evaluation_object(process)
```

Multiple evaluation objects can be added, which allows simultaneous optimization of multiple operating conditions.
When adding variables, specify which evaluation objects the variable targets and the path to the attribute.

```{code-cell} ipython3
optimization_problem.add_variable(
    'var_0',
    evaluation_objects=[process],
    parameter_path='flow_sheet.column.total_porosity',
    lb=0, ub=1,
)
```

By default, a variable targets all evaluation objects.
If no path is provided, the variable name is used as the path.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('evaluation_object_demo')
optimization_problem.add_evaluation_object(process)
```

```{code-cell} ipython3
optimization_problem.add_variable('flow_sheet.column.total_porosity', lb=0, ub=1)
```

Multiple evaluation objects with different variable associations:

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

To register a preprocessing step, use {meth}`~CADETProcess.optimization.OptimizationProblem.add_evaluator`.
Any callable can be an evaluator; its first argument receives the input (the evaluation object, or the output of an upstream evaluator) and it returns a result that downstream nodes consume.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('evaluator_demo')
optimization_problem.add_variable('x')
```

```{code-cell} ipython3
def evaluator(x):
    return x**2

optimization_problem.add_evaluator(evaluator)
```

To wire an objective to this evaluator, pass it via the `requires` argument on {meth}`~CADETProcess.optimization.OptimizationProblem.add_objective`.

```{code-cell} ipython3
def objective(result):
    return result + 1

optimization_problem.add_objective(objective, requires=[evaluator])
```

When evaluating objectives, the evaluator runs first and its output is passed to the objective.

```{code-cell} ipython3
optimization_problem.evaluate_objectives(2)
```

### Shared evaluator outputs

When multiple objectives or constraints depend on the same evaluator, the evaluator runs once and its result is shared.
This is structural, not a cache accident: the pipeline knows the graph and executes each node exactly once.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('shared_evaluator_demo')
optimization_problem.add_variable('x')
```

```{code-cell} ipython3
def simulate(x):
    print(f"simulate called with {x}")
    return {"yield": x * 0.8, "pressure": x * 1.2}

optimization_problem.add_evaluator(simulate)

def yield_objective(sim_result):
    return sim_result["yield"]

def pressure_constraint(sim_result):
    return sim_result["pressure"]

optimization_problem.add_objective(yield_objective, requires=[simulate])
optimization_problem.add_nonlinear_constraint(pressure_constraint, requires=[simulate])
```

```{code-cell} ipython3
print("objectives:", optimization_problem.evaluate_objectives(5))
print("constraints:", optimization_problem.evaluate_nonlinear_constraints(5))
```

Note that `simulate` is called once: both the objective and the constraint receive the same result.

### Chaining evaluators

Evaluators can be chained: the output of one feeds into the next.
Only declare the immediate upstream dependency; transitive dependencies are resolved automatically.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('chain_demo')
optimization_problem.add_variable('x')
```

```{code-cell} ipython3
def simulate(x):
    return {"chromatogram": x * 2}

def fractionate(sim_result):
    return {"yield": sim_result["chromatogram"] * 0.9}

optimization_problem.add_evaluator(simulate)
optimization_problem.add_evaluator(fractionate)

def compute_yield(frac_result):
    return frac_result["yield"]

optimization_problem.add_objective(compute_yield, requires=[simulate, fractionate])
optimization_problem.evaluate_objectives(3)
```

## Caching

The evaluation pipeline caches intermediate results so that repeated evaluations at the same parameter vector are free.
This is particularly useful during gradient approximation, where the same point may be evaluated multiple times.

By default, an in-memory hybrid cache is used (frequency- and cost-weighted eviction, so expensive simulation nodes are kept over cheap postprocessing nodes).
To persist results across runs or share them between parallel workers, pass a `cache_directory` when creating the optimization problem.

```python
optimization_problem = OptimizationProblem('cached', cache_directory='/tmp/my_cache')
```

The disk cache stores results as pickled files and survives process restarts.
For most interactive workflows, the default in-memory cache is sufficient.
