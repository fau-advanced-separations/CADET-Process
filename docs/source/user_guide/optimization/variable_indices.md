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
Consequently, an index *must* be provided if the parameter is not scalar.

For this tutorial, consider the following process model and optimization problem:

```{code-cell} ipython3
import copy
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

from CADETProcess.processModel import Inlet
inlet = Inlet(component_system, 'inlet')

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

## Specifying indices for (multidimensional) arrays

The procedure of adding indices is similar to that of the {class}`~CADETProcess.dynamicEvents.Event` indices (see {ref}`here <event_indices_guide>`).
By default, if no indices are specified, all entries of that array are set to the same value.
E.g. to let the `film_diffusion` coefficients all have the same value, specify the following:

```{code-cell} ipython3
process = setup_process()
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'film_diffusion_all', evaluation_objects=process, parameter_path='flow_sheet.column.film_diffusion'
)
var.value = 1
print(process.flow_sheet.column.film_diffusion)
```

To add a variable that only modifies a single entry of the parameter array, add an `indices` flag.
E.g. for the first component of the (1D) `film_diffusion` array:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'film_diffusion_0', evaluation_objects=process, parameter_path='flow_sheet.column.film_diffusion', indices=0
)
var.value = 2
print(process.flow_sheet.column.film_diffusion)
```

Note, for polynomial parameters, if no index is specified, only the constant coefficient is set and the rest of the coefficients are set to zero.
So, to add the constant term for the `inlet.flow_rate`, use:

```{code-cell} ipython3
var = optimization_problem.add_variable(
    'flow_rate_fill', evaluation_objects=process, parameter_path='flow_sheet.inlet.flow_rate'
)
var.value = 1
print(process.flow_sheet.inlet.flow_rate)
```

However, adding indices still works as expected.
E.g. for the linear coefficient of the `flow_rate`:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'flow_rate_single', evaluation_objects=process, parameter_path='flow_sheet.inlet.flow_rate', indices=1
)
var.value = 2
print(process.flow_sheet.inlet.flow_rate)
```

Just as with 1D arrays, the value is set to all entries of that array if no indices are specified.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'exponents', evaluation_objects=process, parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd'
)
var.value = 1
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
```

Multidimensional parameters can also be indexed by specifying a tuple with the index for each of the parameter dimensions.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'exponents_single', evaluation_objects=process, parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd', indices=(0, 0)
)
var.value = 2
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
```

Just as with Events, slicing notation is also supported:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'exponents_slice', evaluation_objects=process, parameter_path='flow_sheet.column.bulk_reaction_model.exponents_fwd', indices=np.s_[0, :]
)
var.value = 3
print(process.flow_sheet.column.bulk_reaction_model.exponents_fwd)
```

Also the procedure for polynomial parameters is analogous to the 1D case.
For example, to optimize the linear coefficient of the first component of the `inlet` concentration:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'concentration_constant_all', evaluation_objects=process, parameter_path='flow_sheet.inlet.c'
)
var.value = 1
print(process.flow_sheet.inlet.c)
```

To modify only the constant coefficient of a single entry:

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'concentration_fill_values_single', evaluation_objects=process, parameter_path='flow_sheet.inlet.c', indices=0
)
var.value = 2
print(process.flow_sheet.inlet.c)
```

Explicitly modify linear coefficient of single entry.

```{code-cell} ipython3
optimization_problem = setup_optimization_problem()

var = optimization_problem.add_variable(
    'concentration_single_entry', evaluation_objects=process, parameter_path='flow_sheet.inlet.c', indices=(0, 1)
)
var.value = 3
print(process.flow_sheet.inlet.c)
```

## Specifying Indices of Multidimensional Event States

In certain scenarios, optimizing the state of an event becomes essential.
Depending on which parameter dimensions the event modifies, and the indices chosen for the event state, it might be imperative to include indices in the optimization variable as well.

Consider the slope of an elution gradient that starts at a specific time.
Here, the first component starts at $c = 0~mM$ with a slope of $1~mM / s$ whereas the second component's concentration is always $0~mM$.

```{code-cell} ipython3
process = setup_process()
optimization_problem = setup_optimization_problem()

evt = process.add_event(
    'c_poly', 'flow_sheet.inlet.c', [[0, 1], 0], time=1
)
print(inlet.c)
```

To add an optimization variable that only modifies the linear coefficient, add the event state as `parameter_path` and the corresponding index.

```{code-cell} ipython3
var = optimization_problem.add_variable(
    'concentration_single_entry', evaluation_objects=process, parameter_path='c_poly.state', indices=(0, 1)
)
var.value = 2

print(inlet.c)
```
