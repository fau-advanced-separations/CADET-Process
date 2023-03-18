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
In the design of bioreactors and separation vessels, computational fluid dynamics models (CFD) are often used to describe phenomena such as transport mechanisms and mixing behavior.
However, CFD simulations can be computationally very expensive, especially when the goal is to also include kinetic models, e.g. to describe (enzymatic) reactions or biomass growth.

To reduce the complexity of the bioreactor models, the reactor volume can be divided into a number of interconnected sub-volumes called compartments.
Each compartment represents a zone which can be approximated with a simplified hydrodynamic model such as a continuous stirred-tank reactor (CSTR) or an ideal plug flow reactor (PFR) {cite}`Levenspiel1999`.

Since **CADET** is capable of simulating networks of unit operations, it is possible to simulate compartment models.
To facilitate the setup of these models, a {class}`~CADETProcess.modelBuilder.CompartmentBuilder` was implemented in **CADET-Process**.
Note that currently only perfectly mixed compartments are implemented using the built-in {class}`~CADETProcess.processModel.Cstr` model.
However, other compartments can easily be added in the future.

Generally, the following data are required:
- Compartment volumes
- Compartment connectivity (including flow rates)
- Initial conditions

Features:
- Tracer injection experiments
- Internal recycle and flow-through mode
- Possibility to define kinetic reaction models

## Connectivity
First, the definition of the compartment model and the connectivity is introduced.

### Example 1
Consider a simple example with five compartments.

```{figure} ./figures/compartment_simple.svg
:name: compartment_simple

Simple compartment model.
```

Before configuring the {class}`~CADETProcess.modelBuilder.CompartmentBuilder`, a {class}`~CADETProcess.processModel.ComponentSystem` needs to be defined.
It serves to ensure consistent parameter entries for the different modules of **CADET-Process**.
For more information see {ref}`component_system_guide`.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)
```

Then, the {class}`~CADETProcess.modelBuilder.ComparmentBuilder` is imported and configured.
For this purpose, the volume of each compartment needs to be provided, as well as the flow rates between all compartments.
The number of compartments is inferred from the length of the vector of volumes passed in the `__init__` method.
Note that the values for the example are arbitrary.

The flow rate matrix is structured such that each row contains the outgoing streams of the compartment with the row index.
E.g.row $0$ contains all outgoing streams of compartment $0$.

To instantiate the object, the {class}`~CADETProcess.processModel.ComponentSystem` is passed the {class}`~CADETProcess.modelBuilder.CompartmentBuilder` as as well as the vector of compartment volumes and the flow rates.

```{code-cell} ipython3
from CADETProcess.modelBuilder import CompartmentBuilder

volume = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
flow_rate_matrix = [
    0,      0.1e-4, 0.2e-4, 0.3e-4, 0.4e-4,
    0.1e-4, 0,      0,      0,      0,
    0.2e-4, 0,      0,      0,      0,
    0.3e-4, 0,      0,      0,      0,
    0.4e-4, 0,      0,      0,      0,
]

builder_simple = CompartmentBuilder(
    component_system,
    volume, flow_rate_matrix,
)
```

### Example 2
In this example, consider a constant flow through the system.

```{figure} ./figures/compartment_complex.png
:name: compartment_complex

Bioreactor compartment model from {cite:t}`Tobo2020`.
```

To model this system, the {class}`~CADETProcess.processModel.Inlet` and {class}`~CADETProcess.processModel.Outlet` are treated as additional pseudo-compartments.
For this purpose, it is possible to add the stings `inlet` and `outlet` in the vector of volumes.

Although in this example, the overall content volume of the bioreactor does not change, it is generally possible to consider a gradual increase or decrease of volume of each compartment over time.
However, it is important that none of the compartment volumes must never become zero during simulation since this cannot be handled by **CADET**.

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
By default, the concentration of every compartment is initially set to $0~mM$.
To override this, there are multiple options.
First, the same concentration can be used for all components and all compartments:

```{code-cell} ipython3
builder_simple.init_c = 1
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
Often, small amounts of tracer are added to a specific location in the system to measure the mixing behavior of the vessel.
To add a tracer injection to the model, the following information needs to be specified:
- the compartment to which the tracer is added,
- the tracer concentration in the feed,
- the flow rate,
- the injection duration,
- the start time of the injection (optional, default is $0$),
- Whether the volume of the compartment should be kept constant (optional, default is True).

Here, a $10~mM$ tracer is added to compartment $4$ for $10~s$ with a flow rate of  $0.1~L \cdot s^{-1}$ units.

```{code-cell} ipython3
builder_simple.add_tracer(4, [10, 10], 10, 0.1e-3, t_start=0)
```

$$\require{mhchem}$$
## Kinetic reaction models
It is possible to combine the {class}`~CADETProcess.modelBuilder.CompartmentBuilder` with any of the {ref}`adsorption<binding_models_guide>` and {ref}`reaction<reaction_models_guide>` models included in **CADET-Process**.
The models are configured as usual and are then added to the builder instance.

For example, consider this simple reaction:

$$
\ce{A <=>[2][1] B}
$$

To configure reactions, import and configure the `{class}~CADETProcess.processModel.MassActionLaw`.
For more information, see {ref}`reaction_models_guide`.

```{code-cell} ipython3
from CADETProcess.processModel import MassActionLaw

reaction_model = MassActionLaw(component_system)
reaction_model.add_reaction(
    [0, 1], [-1, 1], k_fwd=2, k_bwd=1
)

builder_simple.bulk_reaction_model = reaction_model
```

## Simulation
To simulate the system, the {class}`~CADETProcess.processModel.Process` object of the {class}`~CADETProcess.modelBuilder.CompartmentBuilder` can be accessed directly.
Before passing it to the process simulator, the cycle time needs to be set.

```{code-cell} ipython3
:tags: [remove-cell]
builder_simple.init_c = 0
```
```{code-cell} ipython3
:tags: [remove-output]

from CADETProcess.simulator import Cadet
process_simulator = Cadet()

process = builder_simple.process
process.cycle_time = 100

simulation_results = process_simulator.run(process)
```

## Visualization
Since a regular {class}`~CADETProcess.simulationResults.SimulationResults` object is returned, it is possible to access the solution objects of each individual compartment and use their default {meth}`~CADETProcess.solution.SolutionIO.plot` method.
For example, here the concentration of the first and fourth compartment is plotted:

```{code-cell} ipython3
_ = simulation_results.solution[f'compartment_1'].outlet.plot()
_ = simulation_results.solution[f'compartment_4'].outlet.plot()
```

The effects of the tracer, as well as the reaction are clearly visible.
