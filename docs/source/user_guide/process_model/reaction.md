---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

<!-- ## To do -->
<!-- - [ ] List reactions. -->
<!-- - [ ] Print reactions -->
<!-- - [ ] Add cross-phase reaction demo -->

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../../')
```

$$\require{mhchem}$$

(reaction_models_guide)=
# Chemical Reactions
Since version 4, it is possible to model chemical reactions with CADET using mass action law type reactions (see {ref}`reaction_models`).
The mass action law states that the speed of a reaction is proportional to the product of the concentrations of their reactants.

In CADET-Process, a reaction module was implemented to facilitate the setup of these reactions.
There are two different classes: the {class}`~CADETProcess.processModel.MassActionLaw` which is used for bulk phase reactions, as well as {class}`~CADETProcess.processModel.MassActionLawParticle` which is specifically designed to model reactions in particle pore phase.

## Forward Reactions
As a simple example, consider the following system:

$$
\ce{1 A ->[k_{AB}] 1 B}
$$

Assuming a {class}`~CADETProcess.processModel.ComponentSystem` with components `A` and `B`, configure the {class}`~CADETProcess.processModel.MassActionLaw` reaction model.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(['A', 'B'])
```

To instantiate it, pass the {class}`~CADETProcess.processModel.ComponentSystem`.
Then, add the reaction using the {meth}`~CADETProcess.processModel.MassActionLaw.add_reaction` method.
The following arguments are expected:
- components: The components names that take part in the reaction (useful for bigger systems)
- stoichiometric coefficients in the order of components
- forward reaction rate
- backward reaction rate

```{code-cell} ipython3
from CADETProcess.processModel import MassActionLaw
reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    components=['A', 'B'],
    coefficients=[-1, 1],
    k_fwd=0.1,
    k_bwd=0
)
```

To demonstrate this reaction, a {class}`~CADETProcess.processModel.Cstr` is instantiated and the reaction is added to the tank.
Moreover, the initial conditions are set.
In principle, the {class}`~CADETProcess.processModel.Cstr` supports reactions in bulk and particle pore phase.
Since the porosity is $1$ by default, only the bulk phase is considered.

```{code-cell} ipython3
from CADETProcess.processModel import Cstr

reactor = Cstr(component_system, 'reactor')
reactor.bulk_reaction_model = reaction_system
reactor.V = 1e-6
reactor.c = [1.0, 0.0]
```

## Equilibrium Reactions
It is also possible to consider equilibrium reactions where the product can react back to the educts.

$$
\ce{ 2 A <=>[k_{AB}][k_{BA}] B}
$$

```{code-cell} ipython3
reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    components=['A', 'B'],
    coefficients=[-2, 1],
    k_fwd=0.2,
    k_bwd=0.1
)

reactor.bulk_reaction_model = reaction_system
```
