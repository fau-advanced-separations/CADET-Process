---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: dev
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```

(variable_indices_guide)=
# Specifying indices of multidimensional parameters for Optimization Variables

Similar to events, sometimes we only want to add individual entries of a parameter as an optimization variable.
In fact, optimization variables can only be scalar.
Consequently, an index *must* be provided if the parameter is not scalar and no other broadcasting rule applies (see below).

For this tutorial, consider the following process model and optimization problem:

```{code-cell} ipython3
import numpy as np

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)

from CADETProcess.processModel import MassActionLaw
reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    indices=[0, 1],
    coefficients=[-1, 1],
    k_fwd=0.1,
    k_bwd=0,
)
reaction_system.add_reaction(
    indices=[1, 0],
    coefficients=[-1, 1],
    k_fwd=0.2,
    k_bwd=0,
)

from CADETProcess.processModel import Inlet, LumpedRateModelWithPores, Outlet
inlet = Inlet(component_system, name='inlet')
column = LumpedRateModelWithPores(component_system, 'column')
column.bulk_reaction_model = reaction_system
outlet = Outlet(component_system, 'outlet')

from CADETProcess.processModel import FlowSheet, Process

flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

def setup_process():
    process = Process(flow_sheet, 'Demo Indices')
    process.cycle_time = 10

    return process

from CADETProcess.optimization import OptimizationProblem

def setup_optimization_problem():
    optimization_problem = OptimizationProblem('Demo Indices', use_diskcache=False)
    optimization_problem.add_evaluation_object(process)

    return optimization_problem
```

Values are written into the evaluation objects with {meth}`~CADETProcess.optimization.OptimizationProblem.set_variables`, which takes a vector ordered like {attr}`~CADETProcess.optimization.OptimizationProblem.independent_variables`.
Each example below registers a single variable, so a one-element vector suffices.

## Specifying indices for (multidimensional) arrays

The procedure of adding indices is similar to that of the {class}`~CADETProcess.dynamicEvents.Event` indices (see {ref}`here <event_indices_guide>`).
For a 1-D array, if no indices are specified, all entries are set to the same value.
E.g. to let the `film_diffusion` coefficients all have the same value, specify the following:

```{code-cell} ipython3
process = setup_process()
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'film_diffusion_all', evaluation_objects=process, parameter_path='flow_sheet.column.film_diffusion'
)
optimization_problem.set_variables([1])
print(process.flow_sheet.column.film_diffusion)
assert np.allclose(process.flow_sheet.column.film_diffusion, [1, 1])
```

To add a variable that only modifies a single entry of the parameter array, add an `indices` flag.
E.g. for the first component of the (1D) `film_diffusion` array:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'film_diffusion_0', evaluation_objects=process, parameter_path='flow_sheet.column.film_diffusion', indices=0
)
optimization_problem.set_variables([2])
print(process.flow_sheet.column.film_diffusion)
assert np.allclose(process.flow_sheet.column.film_diffusion, [2, 1])
```

Note, for polynomial parameters, if no index is specified, only the constant coefficient is set and the rest of the coefficients are set to zero.
So, to add the constant term for the `inlet.flow_rate`, use:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'flow_rate_fill', evaluation_objects=process, parameter_path='flow_sheet.inlet.flow_rate'
)
optimization_problem.set_variables([1])
print(process.flow_sheet.inlet.flow_rate)
assert np.allclose(process.flow_sheet.inlet.flow_rate, [1, 0, 0, 0])
```

However, adding indices still works as expected; the constant coefficient set above is left untouched.
E.g. for the linear coefficient of the `flow_rate`:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'flow_rate_single', evaluation_objects=process, parameter_path='flow_sheet.inlet.flow_rate', indices=1
)
optimization_problem.set_variables([2])
print(process.flow_sheet.inlet.flow_rate)
assert np.allclose(process.flow_sheet.inlet.flow_rate, [1, 2, 0, 0])
```

Unlike 1-D arrays and polynomial parameters, a plain multi-dimensional array (e.g. the 2-D reaction exponents below) has no implicit "fill everything" behavior when no index is given.
To set every entry of such an array to the same value, index it explicitly with a full slice.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'exponents', evaluation_objects=process,
    parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd', indices=np.s_[:, :]
)
optimization_problem.set_variables([1])
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
assert np.allclose(process.flow_sheet.column.bulk_reaction_model.exponents_fwd, [[1, 1], [1, 1]])
```

Multidimensional parameters can also be indexed by specifying a tuple with the index for each of the parameter dimensions.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'exponents_single', evaluation_objects=process,
    parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd', indices=(0, 0)
)
optimization_problem.set_variables([2])
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
assert np.allclose(process.flow_sheet.column.bulk_reaction_model.exponents_fwd, [[2, 1], [1, 1]])
```

Just as with Events, slicing notation is also supported, e.g. to set an entire row:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'exponents_slice', evaluation_objects=process,
    parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd', indices=np.s_[0, :]
)
optimization_problem.set_variables([3])
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
assert np.allclose(process.flow_sheet.column.bulk_reaction_model.exponents_fwd, [[3, 3], [1, 1]])
```

Also the procedure for polynomial parameters is analogous to the 1D case, including a multi-dimensional polynomial (e.g. `inlet.c`, one polynomial per component).
For example, to optimize the constant coefficient of every component of the `inlet` concentration:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'concentration_constant_all', evaluation_objects=process, parameter_path='flow_sheet.inlet.c'
)
optimization_problem.set_variables([1])
print(process.flow_sheet.inlet.c)
assert np.allclose(process.flow_sheet.inlet.c, [[1, 0, 0, 0], [1, 0, 0, 0]])
```

A bare index into a multi-dimensional polynomial parameter still applies the "constant coefficient, rest zeroed" convention to just that one entry, leaving the other entries untouched.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'concentration_fill_values_single', evaluation_objects=process, parameter_path='flow_sheet.inlet.c', indices=0
)
optimization_problem.set_variables([2])
print(process.flow_sheet.inlet.c)
assert np.allclose(process.flow_sheet.inlet.c, [[2, 0, 0, 0], [1, 0, 0, 0]])
```

Explicitly modify the linear coefficient of a single entry with a tuple index:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

optimization_problem.add_variable(
    'concentration_single_entry', evaluation_objects=process, parameter_path='flow_sheet.inlet.c', indices=(0, 1)
)
optimization_problem.set_variables([3])
print(process.flow_sheet.inlet.c)
assert np.allclose(process.flow_sheet.inlet.c, [[2, 3, 0, 0], [1, 0, 0, 0]])
```

## Specifying Indices of Multidimensional Event States

In certain scenarios, optimizing the state of an event becomes essential.
Depending on which parameter dimensions the event modifies, the event state can be *ragged*: different components may have a different number of polynomial coefficients, so the state is not a rectangular array.

Consider the slope of an elution gradient that starts at a specific time.
Here, the first component starts at $c = 0~mM$ with a slope of $1~mM / s$ whereas the second component's concentration is always $0~mM$.

```{code-cell} ipython3
process = setup_process()
optimization_problem = setup_optimization_problem()

evt = process.add_event(
    'c_poly', 'flow_sheet.inlet.c', [[0, 1], 0], time=1
)
print(inlet.c)
assert np.allclose(inlet.c, [[0, 1, 0, 0], [0, 0, 0, 0]])
```

Because `evt.state` (`[[0, 1], 0]`) is ragged rather than a rectangular array, `indices` cannot be used here; `IndexedMapper` raises `NotImplementedError` for a ragged target.
Instead, pass `pre_processing`, which builds the full (still-ragged) state from the scalar optimization value, leaving the rest of the structure as declared:

```{code-cell} ipython3
optimization_problem.add_variable(
    'c_poly_linear', evaluation_objects=process, parameter_path='c_poly.state',
    pre_processing=lambda w: [[0, w], 0]
)
optimization_problem.set_variables([2])
print(evt.state)
assert evt.state == [[0, 2.0], 0]
```
