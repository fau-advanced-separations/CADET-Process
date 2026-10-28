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

(zsphinx-build -b html source build_tutorial)=
# Zonal Rate Model Builder

In membrane chromatography (MC), computational fluid dynamics (CFD) models can describe local flow and mixing behavior in detail, but they are computationally expensive and require detailed device geometry, which is not always available.

To reduce the complexity of MC models, the Zonal Rate Model (ZRM), introduced by Francis et al. (2011, 2012), provides a practical reduced-order alternative. In this model, the membrane device is represented by a membrane region and void regions, i.e., inlet and outlet regions. Each region can be discretized into one or more interconnected virtual zones: membrane zones (MZs) and void zones (VZs). Each zone is assumed to be spatially homogeneous.

- MZs describe transport within the membrane and are implemented using a transport model, which can be coupled with a binding model if applicable.
- VZs describe non-membrane volumes corresponding to the distributor and collector regions and are implemented using a continuous stirred-tank reactor (CSTR) model.

## ZRM connectivity

To demonstrate the concept, a representative virtual partitioning of the axial-flow Mustang XT5 capsule is considered.

```{figure} ./figures/ZRM_CFD_setup_2.png
:name: ZRM_CFD_setup_2

Schematic representation of the virtual partitioning of the XT5 capsule geometry into membrane and void zones. Here, $n$ denotes the number of zones. CAD geometry courtesy of Elena Bull.
```

In a standard ZRM layout, each MZ is connected to one inlet VZ and one outlet VZ. Together, these three connected zones form one zonal block.

```{figure} ./figures/zonal_unit.png
:name: zonal_unit

Schematic representation of one zonal block.
```

A single zonal block is often not sufficient to capture spatial heterogeneity. Therefore, a network of multiple zonal blocks is used to better represent heterogeneous flow. In **CADET-Process**, this ZRM structure can be generated using the {class}`~CADETProcess.modelBuilder.ZRMFlowSheetBuilder`. The builder provides a convenient interface for automatically generating a network of multiple zonal blocks.

Figure {numref}`3` generalizes this layout to $n$ zonal blocks in an axial-flow configuration. Here, $VZ_{\mathrm{in},i}$ and $VZ_{\mathrm{out},i}$ represent the inlet and outlet VZs of zonal block $i$, respectively. The internal flow rate through the corresponding MZ is denoted by $Q_i$.

```{figure} ./figures/ZRM_n_axial.png
:name: 3

Schematic representation of a axial ZRM layout with n zonal blocks for an axial-flow configuration.
```

For radial-flow device configuration, a different ZRM layout is used. In this case, the outlet is connected to $VZ_{\mathrm{out},n}$ rather than $VZ_{\mathrm{out},1}$.

```{figure} ./figures/ZRM_n_radial.png
:name: ZRM_n_radial

Schematic representation of a radial ZRM layout with n zonal blocks for an radial-flow configuration.
```

The internal flow rate $Q_i$ through each MZ is assigned according to the surface-area portion of the corresponding MZ relative to the total membrane surface area.

Generally, the following data are required:

- ZRM layout, i.e., axial or radial.
- VZ volumes for each zonal block. In models with multiple zonal blocks, inlet and outlet VZ volumes are provided as separate lists. If the outlet VZ volumes are not provided, they default to the inlet VZ volumes of the corresponding zonal blocks.
- The MZ surface areas must be specified.
<!-- - MZ geometry. For planar membrane geometry, the surface area of each MZ must be specified. For cylindrical or annular membrane geometry, the cylindrical height assigned to each MZ must be specified, as illustrated in {numref}`5`. -->
- MZ transport model, which is provided as a template and used to generate all MZs in the ZRM layout.
<!-- 
```{figure} ./figures/zrm_membrane_segmentation.png
:name: 5

Schematic representation of surface-area segmentation for planar membrane chromatography geometry and cylindrical-height segmentation for annular membrane geometry.
```

The ZRM layout and the MZ geometry describe different parts of the model. The ZRM layout defines how the zonal blocks are connected, whereas the MZ geometry defines how transport within each membrane zone is parameterized. -->

Features:

<!-- - Automatic generation of MZs and inlet/outlet VZs from compact input lists.
- Support for axial- and radial-flow ZRM layouts.
- Automatic assignment of symmetric outlet VZ volumes if they are not provided. -->
- Creation and connection of multiple ZRM builders with different names, for example in series, in parallel, or as part of an SMB process.
- Compatibility with standard **CADET-Process** workflows, allowing ZRM layouts to be combined with binding models, simulations, and parameter-estimation routines.

## Example: Axial-flow configuration Mustang XT5 capsule by Ghosh et al. {cite}`Ghosh2013`

Before configuring the {class}`~CADETProcess.modelBuilder.ZRMFlowSheetBuilder`, a {class}`~CADETProcess.processModel.ComponentSystem` has to be defined. In this example, a single component is used.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(["simulation"])
```

Next, a template model for the MZ is created. This template defines the transport model and its parameters. The same template is used by the {class}`~CADETProcess.modelBuilder.ZRMFlowSheetBuilder` to generate all MZs and connect them with the corresponding VZs in one flow-sheet layout. 

```{code-cell} ipython3
from CADETProcess.processModel import LumpedRateModelWithoutPores

zone_template = LumpedRateModelWithoutPores(
    component_system,
    "zone_template",
    total_porosity=0.7,
    length=2.2e-3,
    axial_dispersion=7e-9,
)
```
As discussed above, {class}`~CADETProcess.modelBuilder.ZRMFlowSheetBuilder` can be combined with other **CADET-Process** features. For example, a binding model can be assigned to the MZ template as follows:

```{code-block} ipython3
from CADETProcess.processModel import Linear

binding_model = Linear(component_system, "binding_model")
binding_model.adsorption_rate = [1]
binding_model.desorption_rate = [1]

zone_template.binding_model = binding_model
```

In the example shown here, binding is not considered. The code above is only included to illustrate how a binding model can be assigned to the MZ template, and the kinetic parameters were set arbitrarily.

Then, the {class}`~CADETProcess.modelBuilder.ZRMFlowSheetBuilder` is imported and configured. For each zonal block, the corresponding VZ volumes and MZ areas have to be provided. The number of zonal blocks is inferred from the length of the MZ segments area list passed to the `__init__` method.

The following configuration defines an axial-flow ZRM with two zonal blocks. The values used here are taken from a published case study reported by Ghosh et al. {cite}`Ghosh2013`, using the axial-flow Mustang XT5 capsule, but they can also be estimated by fitting the model to experimental data using **CADET-Process** optimization.

```{code-cell} ipython3
from CADETProcess.modelBuilder import ZRMFlowSheetBuilder

builder = ZRMFlowSheetBuilder(
    configuration = "axial",
    zone_template = zone_template,
    segments_area = [11.66e-04, 10.34e-04],
    void_in_volumes = [1.24e-06, 1.69e-06],
    void_out_volumes = [1.24e-06, 1.69e-06],
    name = "two_zonal_blocks"
)

flow_sheet_2 = builder.build_flow_sheet()
```

`build_flow_sheet()` returns a {class}`~CADETProcess.processModel.FlowSheet` containing the ZRM layout. The builder also exposes `inlet_port` and `outlet_port`, which can be used to connect external units. The `inlet_port` is used for the upstream connection, and the `outlet_port` is used for the downstream connection.

## External units

Now, the {class}`~CADETProcess.processModel.Inlet` and {class}`~CADETProcess.processModel.Outlet` of the system are configured.

```{code-cell} ipython3
from CADETProcess.processModel import Inlet, Outlet

inlet = Inlet(component_system, "inlet")

flow_rate = 1e-6    # m^3/s
inlet.c = [1]       # mM
inlet.flow_rate = flow_rate

outlet = Outlet(component_system, "outlet")
```

The residence-time delay caused by the system tubing can be considered. 

```{code-cell} ipython3
from CADETProcess.processModel import TubularReactor

delay_tube = TubularReactor(component_system, "delay_tube")
delay_tube.length = 4.978
delay_tube.diameter = 0.001
delay_tube.axial_dispersion = 0  # no dispersion

```

The external units are added to the flow sheet and connected to the ports exposed by the builder.

```{code-cell} ipython3
flow_sheet_2.add_unit(inlet)
flow_sheet_2.add_unit(delay_tube)
flow_sheet_2.add_unit(outlet)

flow_sheet_2.add_connection(inlet, delay_tube)
flow_sheet_2.add_connection(delay_tube, builder.inlet_port)
flow_sheet_2.add_connection(builder.outlet_port, outlet)
```

## Simulation

To simulate the system, a {class}`~CADETProcess.processModel.Process` object is created from the generated flow sheet and passed to the process simulator. Before running the simulation, the cycle time must be specified.

```{code-cell} ipython3
from CADETProcess.processModel import Process
from CADETProcess.simulator import Cadet

simulator = Cadet()

process_2 = Process(flow_sheet_2, "two_zonal_blocks")
process_2.cycle_time = 50

results_2 = simulator.simulate(process_2)
```

## Visualization: Two zonal blocks

The simulation results can be plotted directly from the solution object using the {meth}`~CADETProcess.solution.SolutionIO.plot` method.

```{code-cell} ipython3
:tags: [remove-stderr]

import numpy as np
from CADETProcess.reference import ReferenceIO

data = np.loadtxt('experimental_data/BSA_BTC_nonbinding.csv', delimiter=',')

time_experiment = data[:, 0]
c_experiment = data[:, 1]

from CADETProcess.reference import ReferenceIO
BTC = ReferenceIO(
    'lab_data', time_experiment, c_experiment
)


from CADETProcess.comparison import Comparator

comparator = Comparator('experimental_data/BSA_BTC_nonbinding.csv')
comparator.add_reference(BTC)
comparator.add_difference_metric('RMSE', BTC, 'outlet.outlet')
comparator.evaluate(results_2)


two_zones_plot = comparator.plot_comparison(results_2, x_axis_in_minutes=False, setup_figure_kwargs={"figsize": (6, 4)}) 
```

## Visualization: One zonal block

For comparison, a model with one zonal block can be generated by passing one MZ area and one inlet/outlet VZ volume.

```{code-cell} ipython3
:tags: [remove-stderr]

builder = ZRMFlowSheetBuilder(
    configuration="axial",
    zone_template=zone_template,
    segments_area=[22e-04],
    void_in_volumes=[2.93e-06],
    void_out_volumes=[2.93e-06],
    name = "one_zonal_block"

)


flow_sheet_1 = builder.build_flow_sheet()

flow_sheet_1.add_unit(inlet)
flow_sheet_1.add_unit(delay_tube)
flow_sheet_1.add_unit(outlet)

flow_sheet_1.add_connection(inlet, delay_tube)
flow_sheet_1.add_connection(delay_tube, builder.inlet_port)
flow_sheet_1.add_connection(builder.outlet_port, outlet)

process_1 = Process(flow_sheet_1, "one_zonal_block")
process_1.cycle_time = 50

results_1 = simulator.simulate(process_1)

%matplotlib
one_zones_plot = comparator.plot_comparison(results_1, x_axis_in_minutes=False, setup_figure_kwargs={"figsize": (6, 4)}) 

```


The two-zonal-block model shows better agreement with the experimental outlet profile than the single-zonal-block model. This indicates that the additional zonal block improves the representation of non-uniform flow and residence-time distribution within the membrane capsule.

