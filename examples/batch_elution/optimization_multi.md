---
jupytext:
  formats: md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(batch_elution_optimization_multi)=
# Optimize Batch Elution Process (Multi-Objective)

## Setup Optimization Problem

```{code-cell}
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('batch_elution_multi')

from examples.batch_elution.process import process
optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)

optimization_problem.add_linear_constraint(
    ['feed_duration.time', 'cycle_time'], [1, -1]
)
```

## Setup Simulator

```{code-cell}
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

optimization_problem.add_evaluator(process_simulator)
```

## Setup Fractionator

```{code-cell}
from CADETProcess.fractionation import FractionationOptimizer
frac_opt = FractionationOptimizer()

optimization_problem.add_evaluator(
    frac_opt,
    kwargs={
        'purity_required': [0.95, 0.95],
        'ignore_failed': False,
        'allow_empty_fractions': False,
    }
)
```

## Setup Objectives

```{code-cell}
from CADETProcess.performance import Productivity, Recovery, EluentConsumption

productivity = Productivity()
optimization_problem.add_objective(
    productivity,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

recovery = Recovery()
optimization_problem.add_objective(
    recovery,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

eluent_consumption = EluentConsumption()
optimization_problem.add_objective(
    eluent_consumption,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)
```

## Add callback for post-processing

```{code-cell}
def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )

optimization_problem.add_callback(
    callback, requires=[process_simulator, frac_opt]
)
```

## Configure Optimizer

```{code-cell}
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
```

## Run Optimization

```{note}
For performance reasons, the optimization is currently not run when building the documentation.
In future, we will try to sideload pre-computed results to also discuss them here.
```

```
if __name__ == '__main__':
    results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=True,
    )
```
