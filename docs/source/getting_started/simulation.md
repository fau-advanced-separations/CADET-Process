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

(simulation_tutorial)=
# Process Simulation
Starting point of process development is the setup of the `ProcessModel` (see {ref}`Framework overview <framework_overview>`), i.e., the specific configuration of the chromatographic process.
This is realized using `UnitOperations` as building blocks.

A `UnitOperation` represents the physico-chemical behavior of an apparatus and holds the model parameters.
currently the following models are available in **CADET-Process**: 
- the `Cstr`, an ideally mixed stirred tank without any concentration gradients. It can be used to model system periphery such as valves or pumps.
- the `TubularReactor` which has no particles and can be used to model tubing or reactive columns. Like all column models that follow, axial dispersion can be considered.
- the `LumpedRateModelWithoutPores` which introduces a stationary phase but neglects particle pores.
- the `LumpedRateModelWithPores` which introduces pores and film diffusion which limits the transport into the particle.
- the `GeneralRateModel` which additionally accounts for pore diffusion and surface diffusion.

All `UnitOperations` can be associated with `BindingModels` that describe the interaction of components with surfaces or chromatographic stationary phases.
For this purpose, a variety of equilibrium relations, for example,the simple `Linear` adsorption isotherm, competitive forms of the `Langmuir` and the `BiLangmuir` models, as well as the competitive `StericMassAction` law can be selected.

Moreover, `ReactionModels` can be used to model chemical reactions.

`Sources` and `Sinks` represent the in- and outlets of the process.
System boundary conditions are described using piecewise third order polynomials which also allows to model (non)linear gradients.

The `ProcessModel` allows connecting multiple `UnitOperations` in a `FlowSheet` to describe more complex operating modes.
By this, configurations of several columns or reactor-separator networks can be defined easily.
Furthermore, tubing and other external volumes can be modelled using `UnitOperations`.
This is made possible by programming the `FlowSheet` as a directed graph that describes the mass transfer between the individual `UnitOperations`.
To allow connecting the units correspondingly, every `UnitOperation` can have any number of in- and outputs except for `Sources` that represent streams entering the system, and `Sinks` that represent those exiting.

For defining the specific connections, every `UnitOperation` has an output state that describes the distribution of the outgoing stream to the succeeding `UnitOperations`.
Note that it is straightforward to also include internal recycles in the `FlowSheet`, which is important for systems such as SSR (see example in Section \ref{subsec:stationarity}) or SMB.

`Events` are used to describe the dynamic operation of the process which is particularly relevant for chromatography.
In these processes, dynamic changes occur at the inlets during operational steps like injection, elution, wash, regeneration or the use of gradients.
But also distinct `FlowSheet` actions like turning on and off recycles, bypasses, or switching the column configuration have to be considered.
By defining `Events`, the state of any attribute of a `UnitOperation` can be changed to a certain value at a given time. 
In order to reduce the complexity of the setup, dependencies of `Events` (i.e., conditional events) can be specified that define the time of an `Event` using any linear combination of other `Event` times.
Especially for more advanced processes, this reduces the degrees of freedom and facilitates the overall handiness.
For this purpose, also `Durations` can be defined to describe the time between the execution of two `Events`.
Finally, the cycle time $\Delta t_{cycle}$ of the `ProcessModel` defines the overall duration of a process after which the `Events` are repeated.
If the time of an `Event` is larger than the cycle time, the modulo operation can be applied because of the periodic nature of chromatographic processes.
Hence, the `Event` is performed at $t_{event}~\bmod~\Delta t_{cycle}$. 

By default, the $SI$ unit system is applied for all parameters in **CADET-Process**.
However, alternative unit systems can be applied, including dimensionless model parameters.

## Demonstration
To introduce the basic concepts of **CADET-Process**, a simple binary batch elution separation is considered.

```{figure} ../examples/operating_modes/figures/batch_elution_flow_sheet.svg
:name: batch_elution_flow_sheet

Flow sheet for batch elution process.
```

### Component System
First, a `ComponentSystem` needs to be created.
The `ComponentSystem` ensures that all parts of the process are have the same number of components. 
Moreover, components can be named which automatically adds legends to the plot methods.
For advanced use, see {ref}`component_system_reference`.

In this case, it is simple to setup; only the number of components needs to be specified.

```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(2)
```

To also names to the components, pass a list of strings in the constructor:
```{code-cell} ipython3
component_system = ComponentSystem(['A', 'B'])
```

### Binding Model
In this example, the `Langmuir` model is imported and parametrized.
For an overview of all models in **CADET-Process**, see {ref}`binding_reference`. 
It's important to note that the adsorption model is defined independently from the unit operation.
This facilitates reusing the same configuration in different unit operations or processes.

```{code-cell} ipython3
from CADETProcess.processModel import Langmuir

binding_model = Langmuir(component_system, name='langmuir')
```

Before setting parameter values, let us have inspect which parameters are available in the `BindingModel`.
For this purpose, you can access the `parameters` dictionary and print its content.
Note that for many parameters, reasonable default values are provided.

```{code-cell} ipython3
print(binding_model.parameters)
```

Now, we modify the parameters (values are only for demonstration purposes).

```{code-cell} ipython3
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.capacity = [100, 100]
```

### Unit Operations
Now, the unit operation models are instantiated.
For an overview of all models in **CADET-Process**, see {ref}`unit_operation_reference`. 

In a typical batch elution process, there is a feed and an eluent (see {ref}`Figure <batch_elution_flow_sheet>`).
It is assumed that the feed and eluent concentrations are constant over time.
Later, dynamic events are used to modify the flow rate of each source unit to model loading and elution.

```{code-cell} ipython3
from CADETProcess.processModel import Source, LumpedRateModelWithoutPores, Sink

feed = Source(component_system, name='feed')
feed.c = [10, 10]

eluent = Source(component_system, name='eluent')
eluent.c = [0, 0]
```

Now, the column model is configured.
Here, a `LumpedRateModelWithoutPores` is used.
Again, we can can inspect all the available parameters.

```{code-cell} ipython3
column = LumpedRateModelWithoutPores(component_system, name='column')
print(column.parameters)
```

After assigning the geometric and transport parameters, the previously defined binding model is associated the with the column object.
```{code-cell} ipython3
column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7
column.binding_model = binding_model
```

Additionally, the initial conditions need to be configured.
To view all states, use the following command:
```{code-cell} ipython3
print(column.initial_state)
```
By default, all concentrations are assumed to be zero.

Also the column discretization parameters can be modified.
```{code-cell} ipython3
print(column.discretization.parameters)
```

Currently, **CADET** uses a finite volume WENO scheme with.
Usually, only the number of control volumes (`ncol`) needs to be modified.
It is set to $100$ by default which is a reasonable value in most cases.

Finally, an outlet is configured:
```{code-cell} ipython3
outlet = Sink(component_system, name='outlet')
```

### Flow sheet
To represent the flow between different unit operations, a `FlowSheet` object is initiated.
All units need to be added and then connected accordingly.
For more information, see also {ref}`flow_sheet_reference`.

```{code-cell} ipython3
from CADETProcess.processModel import FlowSheet

fs = FlowSheet(component_system)

fs.add_unit(feed)
fs.add_unit(eluent)
fs.add_unit(column)
fs.add_unit(outlet)

fs.add_connection(feed, column)
fs.add_connection(eluent, column)
fs.add_connection(column, outlet)
```

### Process
The `Process` class is used to define dynamic changes to of the units and connections.
After instantiation, it is important to also set the overall duration of the process. 
Since **CADET-Process** is also designed for cyclic processes (see {ref}`stationarity_tutorial`), the corresponding attribute is called `cycle_time`.

```{figure} ../examples/operating_modes/figures/batch_elution_events_simple.svg
:name: batch_elution_events_simple

Events in a batch elution process.
```

In the given example, at $t=0$, the flow rate of the `feed` is turned on, whereas the flow rate of the `eluent` is set to zero.
After the injection time $\Delta t_{inj}$, the feed flow rate is set to zero and the eluent is switched on.

```{code-cell} ipython3
from CADETProcess.processModel import Process

process = Process(fs, 'process')
process.cycle_time = 600
```

The `add_event` method requires the following arguments:
- `name`: Name of the event.
- `parameter_path`: Path of the parameter that is changed in dot notation.
- `state`: Value of the attribute that is changed at Event execution.
- `time`: Time at which the event is executed.

```{code-cell} ipython3
t_inj = 60
Q = 60/(60*1e6)
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q, 0)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0, t_inj)

process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0, 0)
process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q, t_inj)
```

For more advanced operating modes, it is is also possible to set dependencies between event as to reduce the degrees of freedom in the system.
For example, the `eluent_off` event always occurs when the feed is switched on.
Similarly, the `eluent_on` event occurs simultaneously with the `feed_off` event.
To add a dependency to the process, the `add_dependency` method requires the following arguments:
- `dependent_event` : Name of the event whose value will depend on other events.
- `independent_events` : List of other events on which event depends
For more information, see {ref}`event_dependencies`.

```{code-cell} ipython3
process.add_event_dependency('eluent_on', ['feed_off'])
process.add_event_dependency('eluent_off', ['feed_on'])
```

The events can visualized by calling the `plot_events` method.

```{code-cell} ipython3
_ = process.plot_events()
```

### Simulator
To simulate the process, a process simulator needs to be configured.
If no path is specified, **CADET-Process** will try to autodetect **CADET**.

```{code-cell} ipython3
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
```
If a specific version of **CADET** is to be used, add the install path to the constructor:

```
process_simulator = Cadet(install_path='/path/to/cadet/directory')
```

To check that everything works correctly, you can call the `check_cadet` method:
```{code-cell} ipython3
process_simulator.check_cadet()
```

Now, run the simulation:
```{code-cell} ipython3
simulation_results = process_simulator.simulate(process)
```

The `simulation_results` object contains the solution for the inlet and outlet of every unit operation also provide plot methods.
```{code-cell} ipython3
_ = simulation_results.solution.column.inlet.plot()
_ = simulation_results.solution.column.outlet.plot()
```

For more information how to configure the solver and how to get access to more solutions (e.g. bulk phase), see {ref}`simulation_reference`.

