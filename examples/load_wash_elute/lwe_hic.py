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

import matplotlib
matplotlib.use("TkAgg")

from CADETProcess.processModel import ComponentSystem, HICConstantWaterActivity
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
binding_model = HICConstantWaterActivity(component_system, name='HIC_SWA')
# binding_model = HICWaterOnHydrophobicSurfaces(component_system, name='HIC_WHS')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [0.0, 0.7, 1, 200]
binding_model.desorption_rate = [0.0, 1000, 1000, 1000]
binding_model.hic_characteristic = [0.0, 10, 13, 4]
binding_model.capacity = [0.0, 10000000, 10000000, 10000000]
binding_model.beta_0 = 10. ** -0.5
binding_model.beta_1 = 10. ** -3.65
binding_model.bound_states = [0, 1, 1, 1]

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

salt_gradient_start_concentration = 3000
salt_gradient_end_concentration = 50

column.c = [salt_gradient_start_concentration, 0, 0, 0]
column.cp = [salt_gradient_start_concentration, 0, 0, 0]
column.q = [0, 0, 0]

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

c_load = np.array([salt_gradient_start_concentration, 1.0, 1.0, 1.0])
c_wash = np.array([salt_gradient_start_concentration, 0.0, 0.0, 0.0])
c_elute = np.array([salt_gradient_end_concentration, 0.0, 0.0, 0.0])
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
