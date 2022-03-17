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

(ssr)=
# Steady State Recycling


```{code-cell} ipython3
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import Source, LumpedRateModelWithoutPores, Cstr, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

process_name = flow_sheet_name = 'ssr_binary'

# Component System
component_system = ComponentSystem()
component_system.add_component('A')
component_system.add_component('B')

# Binding Model
binding_model = Langmuir(component_system, name='langmuir')
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.capacity = [100, 100]

# Unit Operations
feed = Source(component_system, name='feed')
feed.c = [10, 10]

eluent = Source(component_system, name='eluent')
eluent.c = [0, 0]

tank = Cstr(component_system, name='tank')
tank.V = 0.001
tank.c = [10, 10]

column = LumpedRateModelWithoutPores(component_system, name='column')
column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7

column.binding_model = binding_model

outlet = Sink(component_system, name='outlet')

# Flow sheet
flow_sheet = FlowSheet(component_system, name=flow_sheet_name)

flow_sheet.add_unit(feed, feed_source=True)
flow_sheet.add_unit(eluent, eluent_source=True)
flow_sheet.add_unit(tank)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, chromatogram_sink=True)

flow_sheet.add_connection(feed, tank)
flow_sheet.add_connection(tank, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, tank)
flow_sheet.add_connection(column, outlet)

# Process
process = Process(flow_sheet, name=process_name)

Q = 60/(60*1e6)

# Create Events and Durations
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
process.add_duration('feed_duration')

process.add_event('inject_on', 'flow_sheet.tank.flow_rate', Q)
process.add_event('inject_off', 'flow_sheet.tank.flow_rate', 0.0)

process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
process.add_event_dependency('eluent_on', ['inject_off'])
process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
process.add_event_dependency('eluent_off', ['inject_on'])

process.add_event('recycle_on', 'flow_sheet.output_states.column', 0)
process.add_event('recycle_off', 'flow_sheet.output_states.column', 1)

process.add_event_dependency('feed_on', ['inject_off'])
process.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1,1])
process.add_event_dependency(
    'inject_off', ['inject_on', 'feed_duration', 'recycle_off', 'recycle_on'],
    [1,1,1,-1]
)

# Set process times
process.cycle_time = 600
process.feed_duration.time = 60
process.recycle_on.time = 360
process.recycle_off.time = 420

if __name__ == '__main__':
    from CADETProcess.simulation import Cadet
    process_simulator = Cadet()
    process_simulator.evaluate_stationarity = True

    simulation_results = process_simulator.simulate(process)
    _ = simulation_results.solution.column.outlet.plot()
    _ = simulation_results.solution.outlet.inlet.plot()
```
