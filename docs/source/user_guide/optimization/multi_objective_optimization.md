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
Often, multiple objectives are used:

@todo: formatting

The Pareto front represents the optimal solutions that cannot be improved in one objective without sacrificing another.

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

```{figure} ./figures/multi_objective.svg
:name: multi_objective
```

(mcdm_guide)=
## Multi-criteria decision making (MCDM)
```{figure} ./figures/multi_criteria_decision_function.svg
:name: multi_criteria_decision_function
```

Multi-criteria decision functions can be used in multi-objective optimization to help decision-makers choose the best solution from a set of feasible solutions that meet the multiple objectives.
MCDM methods typically involve a set of criteria that are used to evaluate the performance of each feasible solution, and then rank the solutions based on how well they perform across the criteria.

In multi-objective optimization, the goal is to find the set of solutions that are Pareto-optimal, meaning that there is no feasible solution that is better than any solution in the set across all objectives.
Once a set of Pareto-optimal solutions has been identified, multi-criteria decision functions can be used to rank the solutions based on how well they perform across other criteria that are not explicitly included in the objective functions.

There are several MCDM methods that can be used in multi-objective optimization, including weighted sum, weighted product, and analytical hierarchy process (AHP).
Weighted sum and weighted product are simple linear methods that involve assigning weights to each objective function and then summing or multiplying the weighted objectives to create a composite score for each feasible solution.
AHP is a more complex method that involves comparing pairs of criteria and determining the relative importance of each criterion, and then using those weights to evaluate the performance of each feasible solution.

In general, the choice of MCDM method will depend on the specific problem and the preferences of the decision-makers involved.
It is important to carefully consider the criteria that are important for the decision-making process and to choose a method that is appropriate for the problem at hand.
Additionally, it is important to ensure that the resulting solution set is well-defined and that the chosen MCDM method produces results that are consistent with the goals of the optimization problem.

(meta_scores_guide)=
## Meta Scores
In some situations, it can be advantageous to configure an optimization problem as many objectives.
However, this can lead to many individuals in the final Pareto front.
To limit the

@todo: formatting

```{figure} ./figures/meta_scores.svg
:name: meta_scores
```

```{figure} ./figures/meta_scores_evaluator.svg
:name: meta_scores_evaluator
```

```{figure} ./figures/callbacks.svg
:name: callbacks
```
