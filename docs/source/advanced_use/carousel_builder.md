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

(carousel_tutorial)=
# Carousel Builder
For many applications, the use of multiple columns can improve process performance when compared with conventional batch elution processes.
Next to the well known simulated moving bed (**SMB**) many other operating modes exist which extend the use of multiple columns, e.g. **Varicol**, or **PowerFeed** processes and gradient operations {cite}`SchmidtTraub2020`.

In all of the aforementioned processes, multiple chromatographic columns are mounted to a rotating column carousel and a central multiport switching valve distributes in- and outgoing streams to and from the columns.
After a given time, the column positions are moved to the next position in the carousel.
In this process, the columns pass through different zones which serve different purposes.

For example, in a classical SMB, four zones are present (see {ref}`Figure below <smb_system>`)
- Zone I: Elution of the strongly adsorbing component
- Zone II: Elution of the weakly adsorbing component
- Zone III: Adsorption of the strongly adsorbing component
- Zone IV : Adsorption of the weakly adsorbing component

Moreover, four in- and outlets are connected to the zones:
- Feed: Inlet containing the components to be separated
- Eluent: Inlet with elution buffer
- Extract: Outlet containing the strongly adsorbing component 
- Raffinate: Outlet containing the weakly adsorbing component

To facilitate the configuration of complex SMB, carousel, or other multi column systems systems, a `CarouselBuilder` was implemented in **CADET-Process** 
It allows a straight-forward configuration of the zones and returns a fully configured `Process` object including all internal connections, as well as switching events.

Here are some of the features:
- Any number of inlets and outlets or other peripheral units.
- Any number of zones
- Any number of columns per zone.
- Different column connectivity within the zones:
    * Serial
    * Parallel
- Different connectivity between zones:
    * Directly connected zones
    * Skip zones
    * Mix and split in- and outgoing streams
    * Allow for different flow direction in every zone.
- Any number of side streams

## SMB Process
To demonstrate the tool, consider a standard SMB process.

```{figure} ./figures/smb_flow_sheet.svg
:name: smb_system

Standard SMB system
```

Before configuring the zones, the binding and column models are configured.
The column is later used as a template for all columns in the system.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(2)

# Binding Model
from CADETProcess.processModel import Linear
binding_model = Linear(component_system)
binding_model.adsorption_rate = [6, 8]
binding_model.desorption_rate = [1, 1]

from CADETProcess.processModel import LumpedRateModelWithoutPores
column = LumpedRateModelWithoutPores(component_system, name='master_column')
column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7

column.binding_model = binding_model
```

Now, the inlets and outlets of the system are configured:

```{code-cell} ipython3
from CADETProcess.processModel import Source, Sink
feed = Source(component_system, name='feed')
feed.c = [10, 10]
feed.flow_rate = 2e-7
eluent = Source(component_system, name='eluent')
eluent.c = [0, 0]
eluent.flow_rate = 6e-7

raffinate = Sink(component_system, name='raffinate')
extract = Sink(component_system, name='extract')
```

To allow more complicated systems, **CADET-Process** provides two options for configuring zones, a `SerialZone` and a `ParallelZone`.
For both, the number of columns in the zone needs to be specified.
Since here all the zones only consist of one column, either can be used.

```{code-cell} ipython3
from CADETProcess.modelBuilder import SerialZone, ParallelZone

zone_I = SerialZone(component_system, 'zone_I', 1)
zone_II = SerialZone(component_system, 'zone_II', 1)
zone_III = SerialZone(component_system, 'zone_III', 1)
zone_IV = SerialZone(component_system, 'zone_IV', 1)
```

The `CarouselBuilder` can now be used like a regular `FlowSheet` where the zones are conceptually used like other `UnitOperations`.
After initializing the `CarouselBuilder`, the column template is assigned and all units and zones are added.

```{code-cell} ipython3
from CADETProcess.modelBuilder import CarouselBuilder
        
builder = CarouselBuilder(component_system, 'smb')
builder.column = column
builder.add_unit(feed)
builder.add_unit(eluent)

builder.add_unit(raffinate)
builder.add_unit(extract)

builder.add_unit(zone_I)
builder.add_unit(zone_II)
builder.add_unit(zone_III)
builder.add_unit(zone_IV)
```

Now, the connections are added to the builder.
To define split streams, the `output_state` is used which sets the ratio between outgoing streams of a unit operation in the flow sheet.

```{code-cell} ipython3

builder.add_connection(eluent, zone_I)

builder.add_connection(zone_I, extract)
builder.add_connection(zone_I, zone_II)
w_e = 0.15
builder.set_output_state(zone_I, [w_e, 1-w_e])

builder.add_connection(zone_II, zone_III)

builder.add_connection(feed, zone_III)

builder.add_connection(zone_III, raffinate)
builder.add_connection(zone_III, zone_IV)
w_r = 0.15
builder.set_output_state(zone_III, [w_r, 1-w_r])

builder.add_connection(zone_IV, zone_I)
```

Now, the switch time is assigned to the builder which determines after how much time a column is switched to the next position.
By calling the `build_process()` method, a regular `Process` object is constructed which can be simulated just as usual using **CADET**.
It contains the assembled flow sheet with all columns, as well as the events required for simulation.

```{code-cell} ipython3
builder.switch_time = 300
process = builder.build_process()
```

Since multi column systems often exhibit a transient startup behavior, it might be useful to simulate multiple cycles until cyclic stationarity is reached (see {ref}`stationarity_tutorial`).
Because this simulation is computationally expensive, only a few simulations are run for the documentation. 
Please run this simulation locally to see the full results.

```{code-cell} ipython3
from CADETProcess.simulator import Cadet

process_simulator = Cadet()
# process_simulator.evaluate_stationarity = True
process_simulator.n_cycles = 3

simulation_results = process_simulator.simulate(process)
```

The results can now be plotted. 
For example, this is how the concentration profiles of the raffinate and extract outlets are plotted:

```{code-cell} ipython3
_ = simulation_results.solution.raffinate.inlet.plot()
_ = simulation_results.solution.extract.inlet.plot()
```

It is important to note that for the purpose of simplifying the implementation, each `Zone` internally has an inlet and an outlet which are modelled using a `Cstr` with a very small volume.
The concentration of these in and outlets can also be plotted.
These units get a `_inlet` and `_outlet` suffix.
For example, this is the concentration of the inlet of zone III:

```{code-cell} ipython3
_ = simulation_results.solution.zone_III_inlet.outlet.plot()
```

## Carousel System
Here, another multi column system is considered.

```{figure} ./figures/carousel_flow_sheet.svg
:name: carousel_zones

Carousel system with 4 zones.
```
There exist four zones in this system:
- Wash: 3 columns in series
- Feed: 3 columns in parallel
- Dilute: 2 columns in series; reverse flow
- Elute: 2 Columns in series

First, the inlets and outlets of the system are configured:

```{code-cell} ipython3
from CADETProcess.processModel import Source, Sink
i_wash = Source(component_system, name='i_wash')
i_wash.c = [0, 0]
i_wash.flow_rate = 60/60/1e6
i_feed = Source(component_system, name='i_feed')
i_feed.c = [10, 10]
i_feed.flow_rate = 30/60/1e6
i_elute = Source(component_system, name='i_elute')
i_elute.c = [0, 0]
i_elute.flow_rate = 60/60/1e6

o_waste = Sink(component_system, name='o_waste')
o_product = Sink(component_system, name='o_product')
```

Now the zones are set up and the reverse flow is set in the dilution zone.

```{code-cell} ipython3
from CADETProcess.modelBuilder import SerialZone, ParallelZone

z_wash = SerialZone(component_system, 'z_wash', 3)
z_feed = ParallelZone(component_system, 'z_feed', 3)
z_dilute = SerialZone(component_system, 'z_dilute', 2, flow_direction=-1)
z_elute = SerialZone(component_system, 'z_elute', 2)
```

As in the previous example, the units and zones are added and connected in the `CarouselBuilder`

```{code-cell} ipython3
from CADETProcess.modelBuilder import CarouselBuilder
        
builder = CarouselBuilder(component_system, 'multi_zone')
builder.column = column
builder.add_unit(i_wash)
builder.add_unit(i_feed)
builder.add_unit(i_elute)

builder.add_unit(o_waste)
builder.add_unit(o_product)

builder.add_unit(z_wash)
builder.add_unit(z_feed)
builder.add_unit(z_dilute)
builder.add_unit(z_elute)

builder.add_connection(i_wash, z_wash)
builder.add_connection(z_wash, z_dilute)
builder.add_connection(i_feed, z_feed)
builder.add_connection(z_feed, z_dilute)
builder.add_connection(z_dilute, o_waste)
builder.add_connection(z_dilute, z_elute)
builder.set_output_state(z_dilute, [0.5, 0.5])
builder.add_connection(i_elute, z_elute)
builder.add_connection(z_elute, o_product)
```

