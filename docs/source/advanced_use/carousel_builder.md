---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Carousel Builder

To facilitate the simulation of complex SMB or carousel systems, it is important to automate setting up the flow sheets in a convenient way.
Here are some of the requirements:
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

For this purpose, a `CarouselBuilder` was implemented.
It allows a straight-forward configuration of the zones and returns a fully configured `Process`.

For example, consider the following system:

> include figure here 

First, we configure the binding and column model.
The column is later used as a template for all columns in the system.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import LumpedRateModelWithPores
component_system = ComponentSystem(2)
binding_model = Langmuir()

column = LumpedRateModelWithPores(n_comp, 'master_column')

column.length = 0.84
column.diameter = 6e-2
column.bed_porosity = 0.6
column.particle_porosity = 0.6
column.particle_radius = 714e-6/2
column.axial_dispersion = 8.0e-6
column.film_diffusion = n_comp*[5.0e-6]

column.binding_model = binding_model
```

Now, the inlets and outlets of the system are configured:
```{code-cell} ipython3
from CADETProcess.processModel import Source, Sink, 

source_serial = self.create_source(Q=1, name='source_serial')
sink_serial = self.create_sink(name='sink_serial')
source_parallel = self.create_source(Q=2, name='source_parallel')
sink_parallel = self.create_sink(name='sink_parallel')
```

Then, we configure the different zones.
There are two different zone models; the `SerialZone` and the `ParallelZone`.
For both, the number of columns in the zone needs to be specified.
Optionally, the flow direction can also be changed per zone.

```{code-cell} ipython3
from CADETProcess.modelBuilder import SerialZone, ParallelZone

serial_zone = SerialZone(n_comp, 'serial', 2, flow_direction=1)
parallel_zone = ParallelZone(n_comp, 'parallel', 2, flow_direction=-1)
```

After initializing the `CarouselBuilder`, the column template is assigned.
The `CarouselBuilder` can now be used like a regular `FlowSheet` where the zones are conceptually used like other `UnitOperations`.

```{code-cell} ipython3
from CADETProcess.modelBuilder import CarouselBuilder, 
        
builder = CarouselBuilder(n_comp, 'multi_zone')
builder.column = column
builder.add_unit(source_serial)
builder.add_unit(source_parallel)

builder.add_unit(sink_serial)
builder.add_unit(sink_parallel)

builder.add_unit(serial_zone)
builder.add_unit(parallel_zone)

builder.add_connection(source_serial, serial_zone)
builder.add_connection(serial_zone, sink_serial)
builder.add_connection(serial_zone, parallel_zone)
builder.set_output_state(serial_zone, [0.5, 0.5])

builder.add_connection(source_parallel, parallel_zone)
builder.add_connection(parallel_zone, sink_parallel)
```

Now, we can set a switch time and build the process

```{code-cell} ipython3
builder.switch_time = 1000
process = builder.build_process()
```

The process is a regular `Process` object and can be simulated just as usual using **CADET**.
Since carousel systems often exhibit a transient startup behavior, it might be useful to simulate multiple cycles until cyclic stationarity is reached (see {ref}`cyclic_stationarity`).

```{code-cell} ipython3
from CADETProcess.simulation import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True
simulation_results = process_simulator.simulate(process)
```

The results can now be plotted. 
For example, this is the outlet
It is important to note that for the purpose of simplifying the implementation, each `Zone` internally has an inlet and an outlet which are modelled using a `Cstr` with a very small volume. These can also be plotted.

```{code-cell} ipython3
simulation_results.solution.z_feed_outlet.outlet.plot()
simulation_results.solution.z_feed_outlet.outlet.plot()
```

