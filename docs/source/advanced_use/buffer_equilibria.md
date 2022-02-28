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

(equilibrira_tutorial)=
# Buffer Equilibria

In many applications, it is crucial to know the protonation state of a molecule.
For example, in ion exchange chromatography only charged molecules can adsorb on the resin, i.e. positively charged ions on a cation exchanger, or negatively charged ions on an anion exchanger.
However, the charge of the molecule is a function of the pH value of the solution.
For this purpose, the pH, which is often also influenced by other acids or bases, has to be considered when modelling the system.

Since the $pH$ is defined as:

$$
pH = -log_{10}(\ce{[H^+]}),
$$

it can be derived from knowing the proton concentration, which in turn is a result of reaction equilibria of the protonation and deprotonation reactions.
There reactions are characterized by their $pK_a$ value:

$$
\ce{A <=>[K_{a}] A^{-} + H^{+}}
$$

$$
K_{a} = \frac{\ce{[A^{-}]}\ce{[H^{+}]}}{\ce{[HA]}}
$$

$$
pK_{a} = -log_{10}(K_{a})
$$


To model the pH in CADET, we can treat different protonation states of molecules as individual (pseudo)components which are coupled via chemical reactions.
Currently, reactions in rapid equilibrium are not implemented in CADET but can be approximated by scaling up the reaction rate.

## Example: Ammonia
Ammonia is a base and can thusly accept protons:

$$
\ce{NH3 + H^{+} <=>[pK_{a}] NH4^{+}}.
$$

With a $pK_{a}$ value of 9.2, it is mostly positively charged for pH values lower than 9.2

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem()
component_system.add_component(
    'Ammonia', 
    species=['NH4+', 'NH3'],
    charge=[1, 0]
)
component_system.add_component(
    'H+', 
    charge=1
)
```

Now, we add the reaction:
```{code-cell} ipython3
from CADETProcess.processModel import MassActionLaw
reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    [0, 1, 2], [-1, 1, 1], 10**(-9.2)*1e3, is_kinetic=False
)
```
To demonstrate this, we can now examine the charge distribution of ammonia as a function of the $pH$:

```{code-cell} ipython3
from CADETProcess import equilibria
_ = equilibria.plot_charge_distribution(reaction_system)
```
