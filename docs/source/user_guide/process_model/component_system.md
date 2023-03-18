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

(component_system_guide)=
# Component System
The {class}`~CADETProcess.processModel.ComponentSystem` ensures that all parts of the process have the same number of components.
Moreover, components can be named which automatically adds legends to the plot methods.

The easiest way to initiate a {class}`~CADETProcess.processModel.ComponentSystem` is to simply pass the number of components in the constructor.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(2)
```

Alternatively, a list of strings for the component names can be passed:

```{code-cell} ipython3
component_system = ComponentSystem(['A', 'B'])
```

For more complicated systems, it is recommended to add components individually using {meth}`~CADETProcess.processModel.ComponentSystem.add_component`.
For this purpose, add the name, as well as additional properties such as:
- `charge`
- `molecular_weight`

```{code-cell} ipython3
component_system = ComponentSystem()
component_system.add_component(
	'A',
	charge=1
)
```

Moreover, a {class}`~CADETProcess.processModel.Component` can have different {class}`~CADETProcess.processModel.Species`.
For example, in some situations, charged species need to be considered separately.
However, for plotting, only the total concentration might be required.
To register a {class}`~CADETProcess.processModel.Species` in the {class}`~CADETProcess.processModel.ComponentSystem`, add a `species` argument.
Note that this requires the number of entries to match for all properties.

```{code-cell} ipython3
component_system = ComponentSystem()
component_system.add_component(
	'Proton',
	species=['H+'],
	charge=[1]
)
component_system.add_component(
	'Ammonia',
	species=['NH4+', 'NH3'],
	charge=[1, 0]
)
print(component_system.names)
print(component_system.species)
```
