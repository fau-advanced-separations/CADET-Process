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
# (batch_elution_example)=
# # Batch Elution Chromatography
# tags: batch elution, langmuir
#
# A basic chromatographic batch-elution setup is comprised of feed and eluent reservoirs, a pump that can deliver the necessary flow rate against the pressure drop of the packed column, a valve to select whether feed or eluent are pumped into the column, the column itself, and one or more valves for the collection of fractions.
#
# In **CADET-Process**, this can be modelled by connecting two {class}`Inlets <CADETProcess.processModel.Inlet>`, a column model (e.g. {class}`~CADETProcess.processModel.LumpedRateModelWithoutPores`), and an {class}`~CADETProcess.processModel.Outlet`.
#
# ```{figure} ./figures/flow_sheet.svg
# Flow sheet for batch elution process.
# ```
#
# For this example, consider a two-component system with a Langmuir isotherm.
#
# ## Component System

# %%
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem()
component_system.add_component('A')
component_system.add_component('B')

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
    Inlet, LumpedRateModelWithoutPores, Outlet
)
feed = Inlet(component_system, name='feed')
feed.c = [10, 10]

eluent = Inlet(component_system, name='eluent')
eluent.c = [0, 0]

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
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(feed, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, outlet)

# %% [markdown]
# ## Process
# To model the injection valve, {class}`Events <CADETProcess.processModel.Event>` are introduced that modify the {attr}`~CADETProcess.procesModel.Inlet.flow_rate` attribute of the {class}`~CADETProcess.processModel.Inlet` unit operations.
#
# ```{figure} ./figures/events.svg
# Events of batch elution process.
# ```

# %%
from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch elution')

Q = 60/(60*1e6)
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)

process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)

# %% [markdown]
# ### Event Dependencies
# To reduce the number of event times that need to be specified, event dependencies are specified which enforce that always either feed or eluent are being pumped through the column.
#
# ```{figure} ./figures/event_dependencies.svg
# Events of batch elution process with event dependencies.
# ```

# %%
process.add_duration('feed_duration')

process.add_event_dependency('eluent_on', ['feed_off'])
process.add_event_dependency('eluent_off', ['feed_on'])
process.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])

# %% [markdown]
# ### Event Times
# Now, the cycle time is set to $10~min$ and the `feed_duration` to $1~min$.

# %%
process.cycle_time = 600
process.feed_duration.time = 60

# %% [markdown]
# ## Simulate Process

# %%
if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    simulation_results = process_simulator.simulate(process)

# %% [markdown]
# ## Optimize Fractionation Times
# After simulation, the {class}`~CADETProcess.simulationResults.SimulationResults` can be analyzed to determine optimal fractionation times using the {mod}`~CADETProcess.fractionation` module.

# %%
if __name__ == '__main__':
    from CADETProcess.fractionation import FractionationOptimizer
    fractionation_optimizer = FractionationOptimizer()

    fractionator = fractionation_optimizer.optimize_fractionation(
        simulation_results, purity_required=[0.95, 0.95]
    )

    print(fractionator.performance)
    _ = fractionator.plot_fraction_signal()

# %% [markdown]
# By selecting appropriate operating conditions, such as injected amount and flow rate, an efficient operating scenario can be achieved in which the stationary phase is utilized very efficiently.
# The highest product recovery is achieved by "baseline separation," where the component peaks of the same injection do not overlap when leaving the column.
# Moreover, by minimizing the time between two injections, productivity can be improved.
# By allowing for waste fractions collected between product fractions or between peaks of consecutive injections, productivity and eluent consumption may be further improved at the cost of lower recovery.
#
# These operating conditions can be adjusted using model based design.
# For this purpose, an {class}`~CADETProcess.optimization.OptimizationProblem` is set up where to maximize process performance.
# This can be achieved by combining multiple parameters into a single objective (see {ref}`batch_elution_optimization_single`) or by setting up a multi-objective problem (see {ref}`batch_elution_optimization_multi`).
#
#
# ```{toctree}
# :maxdepth: 1
# :hidden:
#
# optimization_single
# optimization_multi
# ```
