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

The order in which evaluation objects are listed in a declaration carries no meaning: results and labels always follow the order in which the objects were added to the problem, so `evaluation_objects=[process_b, process_a]` and `evaluation_objects=[process_a, process_b]` declare the same thing.

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

(aggregation_guide)=
## Aggregating over evaluation objects

With multiple evaluation objects, every evaluator and objective runs once per object by default: an objective declared on two processes contributes two entries to the objective vector, and the problem is multi-objective.

Sometimes the individual values are not the point and only a combination matters, such as the mean, a weighted sum, or the worst case across operating conditions.
Declaring an objective with `per_object=False` turns it into a collector: it runs once, receives the complete array of its upstream evaluator's per-object results, and reduces it.
The problem then stays single-objective, so cheaper single-objective optimizers apply where a multi-objective formulation would otherwise be needed.

```{code-cell} ipython3
:tags: [hide-cell]

from dataclasses import dataclass

@dataclass
class OperatingPoint:
    name: str
    scale: float
    concentration: float = 1.0

    def __str__(self):
        return self.name

point_a = OperatingPoint('point_a', scale=1.0)
point_b = OperatingPoint('point_b', scale=0.5)

optimization_problem = OptimizationProblem('aggregation_demo')
optimization_problem.add_evaluation_object(point_a)
optimization_problem.add_evaluation_object(point_b)
optimization_problem.add_variable('concentration', lb=0.1, ub=2.0)
```

```{code-cell} ipython3
def simulate(operating_point):
    return operating_point.concentration * operating_point.scale * 10

optimization_problem.add_evaluator(simulate)

def worst_yield(yields):
    return min(yields)

optimization_problem.add_objective(
    worst_yield, requires=[simulate], minimize=False, per_object=False,
)
```

`simulate` still runs once per evaluation object; only the objective collects.
Despite two evaluation objects, the problem has a single objective:

```{code-cell} ipython3
optimization_problem.n_objectives
```

```{code-cell} ipython3
optimization_problem.evaluate_objectives(1.0)
```

A collector needs an upstream evaluator to collect from, so `requires` is mandatory with `per_object=False`.

```{note}
When the evaluation of one object fails, the aggregated objective currently fails as a whole; the per-object fallback (`bad_metrics`) policy for aggregated objectives follows in a later release.
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

### Bypassing the cache

A cached result is occasionally suspect: a simulation may have failed for a transient reason (a solver hiccup, a full disk) that would not reproduce on retry, yet the failure is cached like any other result.
`bypass_cache` on the underlying {class}`~CADETProcess.evaluation_pipeline.EvaluationPipeline` forces a fresh computation for one call, reachable from an {class}`~CADETProcess.optimization.OptimizationProblem` via its `backend` property.

```{code-cell} ipython3
:tags: [hide-cell]

optimization_problem = OptimizationProblem('bypass_cache_demo')
optimization_problem.add_variable('x')

calls = []

def flaky_simulate(x):
    calls.append(x)
    return x * 2

optimization_problem.add_evaluator(flaky_simulate)

def objective(result):
    return result

optimization_problem.add_objective(objective, requires=[flaky_simulate])
```

Repeated calls at the same point hit the cache; `flaky_simulate` runs only once.

```{code-cell} ipython3
optimization_problem.evaluate_objectives(3)
optimization_problem.evaluate_objectives(3)
len(calls)
```

Passing `bypass_cache=True` to the pipeline's `evaluate` forces this one call to recompute every node, without touching the existing cache entry: later calls at the same point still hit the cache as before.

```{code-cell} ipython3
optimization_problem.backend.evaluate({'x': 3}, bypass_cache=True)
len(calls)
```

## Standalone use

Everything above goes through {class}`~CADETProcess.optimization.OptimizationProblem`'s convenience wrappers: `add_evaluator`, `add_evaluation_object`, and `add_objective` all register nodes on an internal {class}`~CADETProcess.evaluation_pipeline.EvaluationPipeline` (reachable via `backend`, as seen above).
The pipeline is also directly usable with a bare {class}`~CADETProcess.parameter_space.ParameterSpace` (see {ref}`parameter_space_guide`), with no objectives, constraints, or optimizer involved.
This is useful for a one-off evaluation, or for evaluating samples drawn from the space (see {ref}`problem_guide` for the batch entry point built on top of it).

```{code-cell} ipython3
from dataclasses import dataclass

from CADETProcess.parameter_space import ParameterSpace, RangedParameter
from CADETProcess.evaluation_pipeline import EvaluationPipeline

@dataclass
class Column:
    length: float = 0.5

column = Column()
space = ParameterSpace()
space.add_evaluation_object(column)
space.add_parameter(RangedParameter('length', float, lb=0.1, ub=10.0), path='length')

pipeline = EvaluationPipeline(space)
pipeline.add_evaluator(lambda col: col.length ** 2, output_name='length_squared')
pipeline.evaluate({'length': 3})
```

Unlike `OptimizationProblem.add_evaluator`, which derives `output_name` from the function's own name, {meth}`~CADETProcess.evaluation_pipeline.EvaluationPipeline.add_evaluator` takes it explicitly: standalone use has no problem-level bookkeeping to derive a name from.
