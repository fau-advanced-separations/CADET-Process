---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
execution:
  timeout: 300
---

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append('../../../')
```

(smb_builder_tutorial)=
# SMB Builder

To facilitate the setup of SMB models, an {class}`~CADETProcess.modelBuilder.SMBBuilder` is provided which automatically configures a {class}`~CADETProcess.modelBuilder.CarouselBuilder` as a 4-Zone SMB.
For it to work, it requires fully configured eluent and feed {class}`Inlets <CADETProcess.processModel.Inlet>`, as well as a column object.
To demonstrate this, consider the following component system, isotherm, and column model:

```{code-cell} ipython3
:tags: [hide-input]
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Linear
from CADETProcess.processModel import Inlet, LumpedRateModelWithPores

# Component System
component_system = ComponentSystem(['A', 'B'])

# Binding Model
binding_model = Linear(component_system)
binding_model.is_kinetic = True
binding_model.adsorption_rate = [2, 3]
binding_model.desorption_rate = [1, 1]

# Column
column = LumpedRateModelWithPores(component_system, name='column')
column.binding_model = binding_model

column.length = 0.6
column.diameter = 0.025
column.bed_porosity = 0.4

column.particle_porosity = 0.7
column.particle_radius = 5.0e-6
column.film_diffusion = component_system.n_comp*[1e-5]

column.axial_dispersion = 1e-8

eluent = Inlet(component_system, name='eluent')
eluent.c = [0, 0]
eluent.flow_rate = 7e-7

feed = Inlet(component_system, name='feed')
feed.c = [10, 10]
feed.flow_rate = 4e-7
```

These objects are now passed to the {class}`~CADETProcess.modelBuilder.SMBBuilder`, which inherits from {class}`~CADETProcess.modelBuilder.CarouselBuilder`.
Consequently, switch time and output states can be specified as before.
Note, that the names of the automatically generated zones are: `zone_I` `zone_II` `zone_III` `zone_IV`.

```{code-cell} ipython3
from CADETProcess.modelBuilder import SMBBuilder

smb_builder = SMBBuilder(feed, eluent, column)
smb_builder.switch_time = 100

w_e = 0.14
smb_builder.set_output_state('zone_I', [w_e, 1-w_e])

w_r = 0.13
smb_builder.set_output_state('zone_III', [w_r, 1-w_r])
```

Again, the {class}`~CADETProcess.processModel.Process` needs to be built before simulation.

```{code-cell} ipython3
process = smb_builder.build_process()

from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.n_cycles = 3

simulation_results = process_simulator.simulate(process)

_ = simulation_results.solution.raffinate.inlet.plot()
_ = simulation_results.solution.extract.inlet.plot()
```

(smb_design_tutorial)=
## SMB Design

```{warning}
This functionality is still work in progress.
Changes in the interface are to be expected.
```

Several shortcut methods exist for the design of SMB systems.
Most famously, the so-called triangle theory can be used to determine flow rates of the individual zones.

However, these methods highly depend on the adsorption behaviour of the components.
In **CADET-Process**, design methods for two-component {class}`~CADETProcess.processModel.Linear` and {class}`~CADETProcess.processModel.Langmuir` models are provided.
For this purpose, specialized {class}`~CADETProcess.modelBuilder.SMBBuilder` classes need to be used.

Considering the same parameters as above, a {class}`~CADETProcess.modelBuilder.LinearSMBBuilder` needs to be imported.
The {meth}`~CADETProcess.modelBuilder.LinearSMBBuilder.triangle_design` method returns the values for `Q_feed`, `Q_eluent`, `w_r`, and `w_e`.
By default, these values are also set automatically in the SMB builder.

```{code-cell} ipython3
from CADETProcess.modelBuilder import LinearSMBBuilder

smb_builder = LinearSMBBuilder(feed, eluent, column)
smb_builder.switch_time = 100

Q_feed, Q_eluent, w_r, w_e = smb_builder.triangle_design()
print(Q_feed, Q_eluent, w_r, w_e)
```

To account for non-idealities, such as axial dispersion or adsorption kinetics, a safety factor `gamma` can also be provided which moves the operating point from the theoretical optimum towards the middle of the triangle, ensuring the purity of the products.

```{code-cell} ipython3
Q_feed, Q_eluent, w_r, w_e = smb_builder.triangle_design(gamma=1.05)
print(Q_feed, Q_eluent, w_r, w_e)
```

Moreover, the triangle region can also be plotted.

```{code-cell} ipython3
_ = smb_builder.plot_triangle(gamma=1.05)
```

Running the simulation is equivalent to before.

```{code-cell} ipython3
process = smb_builder.build_process()

from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.n_cycles = 3

simulation_results = process_simulator.simulate(process)

_ = simulation_results.solution.raffinate.inlet.plot()
_ = simulation_results.solution.extract.inlet.plot()
```
