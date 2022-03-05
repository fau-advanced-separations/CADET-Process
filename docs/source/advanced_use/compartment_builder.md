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
sys.path.append('../../../')
```

(compartment_tutorial)=
# Compartment Modeling
Computational fluid dynamic (CFD) simulations are often used to model computationally expensive

In the design of bioreactors and separation vessels, computational fluid dynamics models (CFD) are often used to describe phenomena like mixing behavior and transport mechanisms.
However, CFD simulations are computationally very expensive, especially when the goal is to also include kinetic model such as (enzymatic) reactions or biomass growth.
To reduce the complexity of these models, it is possible to use CFD simulations to calibrate a compartmentalized model which assumes that the overall system can be reasonably approximated by a system of interconnected simple, ideal models such as Cstrs and plug flow reactors.

Since **CADET** is also capable of simulating networks of unit operations, it is possible to simulate these kinds of models.
For this purpose, a `CompartmentBuilder` was implemented in **CADET-Process** to facilitate the setup of compartment models.

Generally, the following data are required:
- Compartment volumes
- Compartment connectivity (with flow rates)
- Initial conditions

Features:
- Closed volume and flow-through
- Tracer injection experiments
- Combination with kinetic reaction models

Further possible modes (WIP):
- Instationary exchange between compartments (piecewise constant or cubic polynomial)
- Instationary compartment volume.

In the following, an introduction to the setup and configuration of the different features is given.

## Connectivity
First, the definition of the compartment model and the connectivity is introduced. 

### Example 1
To demonstrate the tool, a simple example with five compartments is considered.

```{figure} ../_static/compartment.svg
:name: compartment_simple

Simple compartment model.
```

Before configuring the `ComparmentBuilder`, a `ComponentSystem` needs to be defined.
It serves to ensure consistent parameter entries for the different modules of **CADET-Process**.
For more information see {ref}`simulation`.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)
```

Then, the `ComparmentBuilder` is imported and configured.
For this purpose, the volume of each compartment needs to be provided, as well as the flow rates between all compartments.
The number of compartments is inferred from the length of the volumes passed in the `__init__` method.
Note that the values for the example are just for demonstration purposes.

The flow rate matrix is structured such that each row contains the outgoing streams of the compartment with the row index.
E.g.row $0$ contains all outgoing streams of compartment $0$.

To instantiate the object, the `ComponentSystem` is passed the `ComponentSystem`, the volume and the flow rates.

```{code-cell} ipython3
from CADETProcess.modelBuilder import CompartmentBuilder

volume = [1, 2, 3, 4, 5]
flow_rate_matrix = [
    0,   0.1, 0.2, 0.3, 0.4,
    0.1, 0,   0,   0,   0,
    0.2, 0,   0,   0,   0,
    0.3, 0,   0,   0,   0,
    0.4, 0,   0,   0,   0,
]

builder_simple = CompartmentBuilder(
    component_system,
    volume, flow_rate_matrix,    
)
```

### Example 2
In this second example, it is considered that there is a constant flow through the system.

```{figure} ../_static/compartment_complex.jpg
:name: compartment_complex

Bioreactor compartment model from {cite:t}`Tobo2020`.
```

To model this more complicated system, the inlet and outlet are treated as additional pseudo-compartments.
For this purpose, it is possible to add the stings `inlet` and `outlet` in the volume list.

Note that in this example the overall content volume of the vessel does not change but it is generally possible to consider a gradual increase or decrease of volume of each compartment over time.
However, it is important that none of the compartment volumes must ever become zero since this situation cannot be handled by the methods used in **CADET**.

```{code-cell} ipython3
volume = ['inlet', 2, 1, 3, 1, 2, 1, 4, 1, 'outlet']

flow_rate_matrix = [
#   0    1    2    3    4    5    6    7    8    9
    0,   0.1, 0,   0,   0,   0,   0,   0,   0,   0,   # 0
    0,   0,   0.3, 0,   0,   0,   0,   0.1, 0,   0,   # 1
    0,   0.1, 0,   0.1, 0,   0,   0,   0.1, 0,   0,   # 2
    0,   0.2, 0,   0,   0,   0.5, 0,   0,   0.1, 0,   # 3
    0,   0,   0,   0.1, 0,   0.1, 0,   0.1, 0,   0,   # 4
    0,   0,   0,   0,   0.3, 0,   0.2, 0.1, 0,   0,   # 5
    0,   0,   0,   0,   0,   0,   0,   0.1, 0,   0.1, # 6
    0,   0,   0,   0.5, 0,   0,   0,   0,   0,   0,   # 7
    0,   0,   0,   0.1, 0,   0,   0,   0,   0,   0,   # 8
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   # 9
]

builder_complex = CompartmentBuilder(
    component_system,
    volume, flow_rate_matrix,    
)
```

## Initial conditions
By default, the concentration of every compartment is set to $0$.
To set the initial concentration manually, there are multiple options.
First, the same concentration can be used for all components and all compartments:

```{code-cell} ipython3
builder_simple.init_c = 0
```

Alternatively, a value for each concentration can be provided which is then set to all compartments:

```{code-cell} ipython3
builder_simple.init_c = [0,1]
```

Finally, a vector of concentrations for each compartment can be provided as a numpy array (assuming example 1):

```{code-cell} ipython3
import numpy as np

builder_simple.init_c = np.array([
    [0,0],
    [1,1],
    [2,2],
    [3,3],
    [4,4]
])
```

## Tracers
To add a tracer, we have to specify the following information:
- the compartment to which the tracer is added
- the tracer concentration,
- the flow rate
- the injection duration
Optionally, it is also possible to provide the start time of the injection.

```{code-cell} ipython3
builder_simple.add_tracer(4, [1, 1], 10, 0.1, t_start=0)
```

$$\require{mhchem}$$
## Kinetic reaction models
It is possible to combine the `CompartmentBuilder` with any of the adsorption and reaction models included in **CADET-Process**.
The models are configured as usual and are then set to the builder instance.

For example, consider this simple reaction:

$$
\ce{A <=>[2][1] B}
$$

It is important to note that the `Cstr` supports reactions in bulk and particle pore phase.
Here, only bulk phase is considered.

```{code-cell} ipython3
from CADETProcess.processModel import MassActionLaw

reaction_model = MassActionLaw(component_system)
reaction_model.add_reaction(
    [0, 1], [-1, 1], k_fwd=2, k_bwd=1
)

builder_simple.bulk_reaction_model = reaction_model
```

## Simulation
To simulate the system, the `process` object of the `can be accessed directly and passed to **CADET**

```{code-cell} ipython3
:tags: [remove-output]

from CADETProcess.simulation import Cadet
process_simulator = Cadet()

proc_results = process_simulator.run(builder_simple.process)
```

## Visualization
Since a regular `ProcessResults` object is returned, it is possible to access the solution of each compartment, as well as the default `plot` method.

```{code-cell} ipython3
:tags: [remove-cell]

builder_simple = CompartmentBuilder(
    component_system,
    volume, flow_rate_matrix,    
)
```

```ipython3
for comp in range(builder_simple.n_compartments):
    proc_results.solution[f'compartment_{comp}'].outlet.plot()
```

