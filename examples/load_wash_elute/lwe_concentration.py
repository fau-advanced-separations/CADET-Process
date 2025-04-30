# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# (lwe_example_concentration)=
# # Concentration Gradients
#
# ```{figure} ./figures/flow_sheet_concentration.svg
# Flow sheet for load-wash-elute process using a single inlet.
# ```

# %%
import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import StericMassAction
from CADETProcess.processModel import Inlet, GeneralRateModel, Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Component System
component_system = ComponentSystem()
component_system.add_component('Salt')
component_system.add_component('A')
component_system.add_component('B')
component_system.add_component('C')

# Binding Model
binding_model = StericMassAction(component_system, name='SMA')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [0.0, 35.5, 1.59, 7.7]
binding_model.desorption_rate = [0.0, 1000, 1000, 1000]
binding_model.characteristic_charge = [0.0, 4.7, 5.29, 3.7]
binding_model.steric_factor = [0.0, 11.83, 10.6, 10]
binding_model.capacity = 1200.0

# Unit Operations
inlet = Inlet(component_system, name='inlet')
inlet.flow_rate = 6.683738370512285e-8

column = GeneralRateModel(component_system, name='column')
column.binding_model = binding_model

column.length = 0.014
column.diameter = 0.02
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.75
column.axial_dispersion = 5.75e-8
column.film_diffusion = column.n_comp * [6.9e-6]
column.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
column.surface_diffusion = column.n_bound_states * [0.0]

column.c = [50, 0, 0, 0]
column.cp = [50, 0, 0, 0]
column.q = [binding_model.capacity, 0, 0, 0]

outlet = Outlet(component_system, name='outlet')

# Flow Sheet
flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

# %% [markdown]
# ```{figure} ./figures/events_concentration.svg
# Events of load-wash-elute process using a single inlet and modifying its concentration.
# ```

# %%
# Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 2000.0

load_duration = 9
t_gradient_start = 90.0
gradient_duration = process.cycle_time - t_gradient_start

c_load = np.array([50.0, 1.0, 1.0, 1.0])
c_wash = np.array([50.0, 0.0, 0.0, 0.0])
c_elute = np.array([500.0, 0.0, 0.0, 0.0])
gradient_slope = (c_elute - c_wash) / gradient_duration
c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))

process.add_event('load', 'flow_sheet.inlet.c', c_load)
process.add_event('wash', 'flow_sheet.inlet.c', c_wash, load_duration)
process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly, t_gradient_start)

# %%
if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    simulation_results = process_simulator.simulate(process)

    from CADETProcess.plotting import SecondaryAxis
    sec = SecondaryAxis()
    sec.components = ['Salt']
    sec.y_label = '$c_{salt}$'

    simulation_results.solution.column.outlet.plot(secondary_axis=sec)
