# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# (ssr_process)=
# # Steady-State Recycling Process
# In addition to the recycled fraction, fresh feed can also be injected in each cycle, resulting in the formation of a cyclic steady-state.
# This process, called closed-loop steady-state recycling (CL-SSR), can achieve higher productivity compared to CLR.
# However, due to additional dispersion in the system periphery, maintaining the separation of components generated during the passage of the column is difficult to realize.
# Hence, determining the optimal time at which to add new feed is therefore complex.
# To overcome this problem, a tank can be inserted in which the recycling fraction and new feed are mixed.
# The recycling fraction and new feed are then injected together in a process called mixed-recycle steady-state recycling (MR-SSR).
# A schematic flow diagram of the MR-SSR process is shown below.
#
# ```{figure} ./figures/mrssr_flow_sheet.svg
# :name: mrssr_flow_sheet
#
# Flow sheet for mixed-recycle steady-state recycling process.
# ```
#
# For this demonstration, consider a two-component system with a Langmuir isotherm.
#
# ## Component System

# %%
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(['A', 'B'])

# %% [markdown]
# ## Binding Model

# %%
from CADETProcess.processModel import Langmuir

binding_model = Langmuir(component_system, name='langmuir')
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.capacity = [100, 100]

# %% [markdown]
# ## Unit Operations

# %%
from CADETProcess.processModel import (
    Inlet, Cstr, LumpedRateModelWithoutPores, Outlet
)
feed = Inlet(component_system, name='feed')
feed.c = [10, 10]

eluent = Inlet(component_system, name='eluent')
eluent.c = [0, 0]

tank = Cstr(component_system, name='tank')
tank.V = 0.001
tank.c = [10, 10]

column = LumpedRateModelWithoutPores(component_system, name='column')
column.binding_model = binding_model

column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7

column.solution_recorder.write_solution_bulk = True

outlet = Outlet(component_system, name='outlet')

# %% [markdown]
# ## Flow Sheet

# %%
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed, feed_inlet=True)
flow_sheet.add_unit(eluent, eluent_inlet=True)
flow_sheet.add_unit(tank)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(feed, tank)
flow_sheet.add_connection(tank, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, tank)
flow_sheet.add_connection(column, outlet)

# %% [markdown]
# ## Process
# To realize the recycling, the {attr}`~CADETProcess.processModel.FlowSheet.output_state` of the column needs to be modified.
# To reduce the number of event times that need to be specified, event dependencies are specified which enforce that always either feed or eluent are being pumped through the column.
#
# ```{figure} ./figures/mrssr_events.svg
# :name: mrssr_events
#
# Events for mixed-recycle steady-state recycling process with event dependencies.
# ```

# %%
from CADETProcess.processModel import Process
process = Process(flow_sheet, 'mrssr')

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
process.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])
process.add_event_dependency(
    'inject_off', ['inject_on', 'feed_duration', 'recycle_off', 'recycle_on'],
    [1, 1, 1, -1]
)

# %% [markdown]
# Now, the cycle time is set to $10~min$ and the `feed_duration` to $1~min$ and the recycling times are specified.

# %%
process.cycle_time = 600
process.feed_duration.time = 60
process.recycle_on.time = 360
process.recycle_off.time = 420

# %% [markdown]
# ## Simulate Process
# Since the process shows a startup behavior before reaching steady state, multiple cycles need to be simulated.
# For this purpose, a {class}`~CADETProcess.stationarity.StationarityEvaluator` is used (see {ref}`stationarity_guide`).

# %%
if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()
    process_simulator.evaluate_stationarity = True

    simulation_results = process_simulator.simulate(process)
    simulation_results.solution.column.outlet.plot()
    simulation_results.solution_cycles.column.outlet[-1].plot()

# %% [markdown]
# ## Optimize Fractionation Times
# To evaluate the process performance, fractionation times can determined to maximize yield and ensure purity constraints.

# %%
if __name__ == '__main__':
    from CADETProcess.fractionation import FractionationOptimizer
    fractionation_optimizer = FractionationOptimizer()

    fractionator = fractionation_optimizer.optimize_fractionation(
        simulation_results, purity_required=[0.95, 0.95]
    )

    print(fractionator.performance)
    _ = fractionator.plot_fraction_signal()
