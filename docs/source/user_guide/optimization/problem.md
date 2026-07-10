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

(problem_guide)=
# Problem

{class}`~CADETProcess.optimization.OptimizationProblem` bundles three separate concerns: a {class}`~CADETProcess.parameter_space.ParameterSpace` (the input domain), a {class}`~CADETProcess.metric_space.MetricSpace` (the declared outputs), and an evaluation backend (typically an {class}`~CADETProcess.evaluation_pipeline.EvaluationPipeline`, see {ref}`evaluation_pipeline_guide`).
{class}`~CADETProcess.problem.Problem` is that combination without any optimizer policy: no objective/constraint direction beyond what `MetricSpace` declares, no `bad_metrics` substitution, no callbacks.
It is directly usable wherever a sampler or a surrogate model needs declared outputs for a named assignment, without running an optimization.

## Declaring metrics

A {class}`~CADETProcess.metric_space.MetricSpace` collects named output declarations.
Unlike `OptimizationProblem.add_objective`, which derives a name from the callable when none is given, `MetricSpace.add_metric` always takes an explicit name.

```{code-cell} ipython3
from CADETProcess.metric_space import Metric, MetricSpace

metric_space = MetricSpace()
metric_space.add_metric(Metric('volume'))
```

## Wiring a backend

The parameter space and the evaluation backend are built the same way as standalone use of the {ref}`evaluation pipeline <evaluation_pipeline_guide>`.

```{code-cell} ipython3
import math
from dataclasses import dataclass

from CADETProcess.parameter_space import ParameterSpace, RangedParameter
from CADETProcess.evaluation_pipeline import EvaluationPipeline

@dataclass
class Column:
    length: float = 0.5
    diameter: float = 0.05

column = Column()
space = ParameterSpace()
space.add_evaluation_object(column)
space.add_parameter(RangedParameter('length', float, lb=0.1, ub=1.0), path='length')
space.add_parameter(RangedParameter('diameter', float, lb=0.01, ub=0.2), path='diameter')

pipeline = EvaluationPipeline(space)
pipeline.add_evaluator(
    lambda col: col.length * (col.diameter / 2) ** 2 * math.pi,
    output_name='volume',
)
```

`Problem` combines the three: the parameter space, the metric space, and the backend that computes the declared metrics.

```{code-cell} ipython3
from CADETProcess.problem import Problem

problem = Problem(
    parameter_space=space, metric_space=metric_space, backend=pipeline, name='column_volume',
)
problem.evaluate({'length': 0.5, 'diameter': 0.05})
```

`evaluate` validates the backend's result against each declared metric's shape and drops anything the metric space did not declare, so pipeline intermediates never leak out.

## Sampling and batch evaluation

The batch entry point, {meth}`~CADETProcess.problem.Problem.evaluate_batch`, is the same method optimizers and surrogate trainers dispatch through; used directly, it turns a set of sampled points into evaluated metrics without any optimizer in between.
Draw the samples from the parameter space (see {ref}`parameter_space_guide` for the available sampler backends), then evaluate them as a batch.

```{code-cell} ipython3
from CADETProcess.parameter_space import LatinHypercubeSampler

samples = LatinHypercubeSampler().sample(space, 8, seed=0)
results = problem.evaluate_batch(samples)
results[:2]
```

## Building a Population

{meth}`~CADETProcess.optimization.Population.from_records` turns paired samples and results into a {class}`~CADETProcess.optimization.Population`, without ever running an optimizer.
Parameter and metric values are stored as named columns, {attr}`~CADETProcess.optimization.Population.X` and {attr}`~CADETProcess.optimization.Population.metrics`, rather than only flat arrays.

```{code-cell} ipython3
from CADETProcess.optimization import Population

records = [{'X': sample, 'metrics': result} for sample, result in zip(samples, results)]
population = Population.from_records(records, metric_space=metric_space, parameter_space=space)
population.X['length'][:5]
```

Indexing a `Population` with an integer returns an {class}`~CADETProcess.optimization.IndividualView`, a row lens exposing the same `X`/`metrics` accessors scoped to that one row.

```{code-cell} ipython3
population[0].X
```

```{code-cell} ipython3
population[0].metrics
```

A single sample goes through {meth}`~CADETProcess.optimization.Population.from_sample` instead, without wrapping it in a list.

```{code-cell} ipython3
one = Population.from_sample(
    samples[0], results[0], metric_space=metric_space, parameter_space=space,
)
one[0].metrics
```

The {ref}`optimizer_guide` documents the further accessors a `Population` exposes on optimization results, including the flat `f`/`g` projections, the constraint-violation matrices, and the feasibility filters.

```{note}
Metric names on a bare `MetricSpace` are always explicit, unlike `OptimizationProblem.add_objective`/`add_evaluator`, which derive a default name from the callable and sanitize it into a valid Python identifier (a bare `lambda` becomes `_lambda_`).
A metric declared with an `evaluation_object` dimension carries its object name as a label suffix even when only one object is registered, since the dimension is part of the metric's shape, not a special case for multiple objects.
```
