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

$$\require{mhchem}$$

(reactions_tutorial)=
# Chemical Reactions

Since version 4, it is possible to model chemical reactions with **CADET** using mass action law type reactions (see {ref}`reaction_models`).
The mass action law states that the speed of a reaction is proportional to the product of the concentrations of their reactants.

In **CADET-Process**, a reaction module was implemented to facilitate the setup of these reactions.
There are two different classes: the `MassActionLaw` which is used for bulk phase reactions, as well as `MassActionLawParticle` which is specifically designed to model reactions in particle pore phase.
 
## Forward Reactions
To demonstrate this, the following system is considered:

$$
\ce{1 A ->[k_{AB}] 1 B}
$$

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(['A', 'B'])
```

Then, the `ReactionModel` is defined.
The `add_reaction` method expects the following arguments: 
- indices: The indices of the components that take part in the reaction (useful for bigger systems)
- stoichiometric coefficients in the order of the indices
- forward reaction rate
- backward reaction rate

```{code-cell} ipython3
from CADETProcess.processModel import MassActionLaw
reaction_system = MassActionLaw(component_system) 
reaction_system.add_reaction(
    indices=[0,1], 
    coefficients=[-1, 1],
    k_fwd=0.1,
    k_bwd=0
)
```

To demonstrate this reaction, a `Cstr` is instantiated and the reaction is added to the tank.
Moreover, the initial conditions are set.
In principle, the `Cstr` supports reactions in bulk and particle pore phase.
Since the porosity is $1$ by default, only the bulk phase is considered.

```{code-cell} ipython3
from CADETProcess.processModel import Cstr
reactor = Cstr(component_system, 'reactor')
reactor.volume = 1e-6
reactor.bulk_reaction_model = reaction_system
reactor.c = [1.0, 0.0]
```

Now, the reactor is added to a flow sheet and process is set up.
```{code-cell} ipython3
from CADETProcess.processModel import FlowSheet
flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(reactor)

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'reaction_demo')
process.cycle_time = 100
```

After simulation, the results can be plotted:
```{code-cell} ipython3
from CADETProcess.simulation import Cadet
simulator = Cadet()
sim_results = simulator.run(process)
sim_results.solution.reactor.outlet.plot()
```


## Equilibrium Reactions
It is also possible to consider equilibrium reactions.

$$
\ce{ 2 A <=>[k_{AB}][k_{BA}] B}
$$


```{code-cell} ipython3
reaction_system = MassActionLaw(component_system) 
reaction_system.add_reaction(
    indices=[0,1], 
    coefficients=[-2, 1],
    k_fwd=0.2,
    k_bwd=0.1
)
reactor.bulk_reaction_model = reaction_system
```

After simulation, the results can be plotted:
```{code-cell} ipython3
sim_results = simulator.run(process)
sim_results.solution.reactor.outlet.plot()
```

## Cross Phase Reactions
In some cases, a component from the liquid phase can participate in a reaction with another component that is in a bound state.
This is also possible to model in **CADET-Process**.
It is important to note that this only works for reactions in particle pores.

Consider the following system which is analogous to an ion exchange reaction:

$$
\ce{ NH_4^+(aq) + H^+(s) <=>[k_{AB}][k_{BA}] NH_4^+(s) + H^+(aq) }
$$


```{code-cell} ipython3
component_system = ComponentSystem()
component_system.add_component(
    'H+', 
)
component_system.add_component(
    'Ammonia', 
    species=['NH4+', 'NH3'],
)
```

Now the `add_cross_phase_reaction` function is used.
In addition to the previous function, it also takes the phase indices as argument which assumes the index $0$ for pores, and $1$ for the solid phase:

```{code-cell} ipython3
from CADETProcess.processModel import MassActionLawParticle
reaction_system = MassActionLawParticle(component_system)
reaction_system.add_cross_phase_reaction(
    indices=[1, 0, 1, 0], 
    coefficients=[-1, -1, 1, 1], 
    phases=[0, 1, 0, 1], 
    k_fwd=1.5,
    k_bwd=1
)
```

