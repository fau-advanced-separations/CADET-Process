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

(moo_guide)=
# Multi-Objective Optimization

In many practical optimization problems, several competing criteria must be balanced simultaneously.
For example, maximizing product yield while minimizing buffer consumption, or maximizing purity while maximizing productivity.
Because these objectives conflict, there is no single solution that optimizes all of them at once.
Instead, the goal is to find the set of *Pareto-optimal* solutions: those where improving one objective necessarily worsens another.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

# Only plot third quadrant
theta = np.linspace(np.pi, 1.5*np.pi, 100)

# the radius of the circle
r = 1

# compute x1 and x2
x1 = r*np.cos(theta) + 1
x2 = r*np.sin(theta) + 1

# create the figure
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_aspect(1)

ax.set_xlabel("$f_1$")
ax.set_ylabel("$f_2$")


# Plot and Annotate Pareto front
ax.plot(x1, x2)

i = 10
ax.annotate(
    'Pareto Front', xy=(x1[i], x2[i]),  xycoords='data',
    xytext=(0.2, 0.8), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='-|>', facecolor='black'),
    horizontalalignment='left', verticalalignment='bottom',
)
# Plot and annotate non-dominated solutions
indices = [40, 50, 60]
nondominated = np.array([(x1[i], x2[i]) for i in indices])

ax.scatter(nondominated[:,0], nondominated[:,1])

for ind in nondominated:
    ax.annotate(
        'Nondominated Solution', xy=ind,  xycoords='data',
        xytext=(0.05, 0.05), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', facecolor='black'),
        horizontalalignment='left', verticalalignment='bottom',
    )

# Plot and annotate dominated solutions
dominated = 1.1*nondominated

ax.scatter(dominated[:,0], dominated[:,1])

for ind in dominated:
    ax.annotate(
        'Dominated Solution', xy=ind,  xycoords='data',
        xytext=(0.5, 0.5), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', facecolor='black'),
        horizontalalignment='left', verticalalignment='bottom',
    )

fig.tight_layout()
```

A *dominated* solution is one for which there exists another solution that is at least as good on every objective and strictly better on at least one.
The Pareto front is the set of all nondominated solutions.

```{figure} ./figures/multi_objective.svg
:name: multi_objective
```

For details on how to set up a multi-objective problem, see {ref}`objectives_guide`.

(mcdm_guide)=
## Multi-Criteria Decision Making (MCDM)

After optimization, the Pareto front may contain many solutions.
A decision maker must select one, and multi-criteria decision making (MCDM) methods provide systematic ways to do so.

```{figure} ./figures/multi_criteria_decision_function.svg
:name: multi_criteria_decision_function
```

Common approaches include weighted sum (assign importance weights to each objective and select the solution with the best composite score) and weighted product (multiplicative analog).
These are simple to implement but require the decision maker to specify weights, which implicitly defines a trade-off ratio between objectives.

The choice of MCDM method depends on the problem and the decision maker's preferences.
The key requirement is that the method produces a consistent ranking that reflects the actual priorities of the application.

(meta_scores_guide)=
## Meta Scores

When many objectives are used, the Pareto front can grow large and become difficult to interpret.
Meta scores provide a way to reduce the effective dimensionality of the Pareto front by aggregating related objectives into composite scores after the optimization has run.

```{figure} ./figures/meta_scores.svg
:name: meta_scores
```

A meta score function receives the Pareto-optimal individuals and computes a derived score, for example an overall process cost that combines yield, purity, and buffer consumption into a single economic metric.
The optimizer then uses these meta scores to filter or re-rank the Pareto front, producing a smaller set of solutions for the decision maker.

Meta scores run through the same evaluation pipeline as objectives: parameter values are written into evaluation objects and the evaluator chain executes before the meta score function is called.

```{figure} ./figures/meta_scores_evaluator.svg
:name: meta_scores_evaluator
```

To add a meta score, use {meth}`~CADETProcess.optimization.OptimizationProblem.add_meta_score`.
The function signature is the same as for objectives: it receives the evaluation result and returns one or more scalar values.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('meta_score_demo')
optimization_problem.add_variable('x', lb=0, ub=10)
optimization_problem.add_variable('y', lb=0, ub=10)
```

```{code-cell} ipython3
import numpy as np

def objective_yield(x):
    return x[0]

def objective_purity(x):
    return x[1]

optimization_problem.add_objective(objective_yield)
optimization_problem.add_objective(objective_purity)

def overall_cost(x):
    return 0.7 * x[0] + 0.3 * x[1]

optimization_problem.add_meta_score(overall_cost)
```
