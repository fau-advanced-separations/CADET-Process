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

(binding_models_guide)=
# Binding Models

In **CADET-Process**, a number of binding models can be used to describe the interaction of molecules with a stationary phase.
For an overview of all models in **CADET-Process**, see {mod}`CADETProcess.processModel`.
It's important to note that the adsorption model is defined independently of the unit operation.
This facilitates reusing the same configuration in different unit operations or processes.

To instantiate a binding model, the corresponding class needs to be imported from the {mod}`CADETProcess.processModel` module.
For example, consider the {class}`~CADETProcess.processModel.Langmuir` isotherm with a two-component system.

```{code-cell} ipython3
:tags: [hide-cell]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)
```

```{code-cell} ipython3
from CADETProcess.processModel import Langmuir
binding_model = Langmuir(component_system)
```

All model parameters can be listed using the `parameters` attribute.

```{code-cell} ipython3
print(binding_model.parameters)
```

Note that some parameters might have default values.
To only show required parameters, inspect `required_parameters`.

```{code-cell} ipython3
print(binding_model.required_parameters)
```

## Multi-State Binding
Some binding models support multiple binding sites, for example, the {class}`~CADETProcess.processModel.BiLangmuir` model.
Note that originally, the Bi-Langmuir model is limited to two different binding site types.
In **CADET**, the model has been extended to arbitrary many binding site types.

```{code-cell} ipython3
from CADETProcess.processModel import BiLangmuir

binding_model = BiLangmuir(component_system, n_binding_sites=2, name='langmuir')
```

{attr}`~CADETProcess.processModel.BindingBaseClass.n_binding_sites` denotes the total number of binding sites whereas {attr}`~CADETProcess.processModel.BindingBaseClass.n_bound_states` the total number of bound states.

```{code-cell} ipython3
print(binding_model.n_binding_sites)
print(binding_model.n_bound_states)
```

This means that parameters like `adsorption_rate` now have $4$ entries.

```{code-cell} ipython3
binding_model.adsorption_rate = [1,2,3,4]
```
The order of this linear array is in `state-major` ordering.

*I.e.*, the bound state index changes the slowest and the component index the fastest:
```
comp0bnd0, comp1bnd0, comp0bnd1, comp1bnd1
```

Note that this also effects the number of entries for the initial state of the stationary phase `q` (see {ref}`unit_operation_guide`).
