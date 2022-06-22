#!/usr/bin/env python3

"""
===========================================
Simulate Load/Wash/Elute with salt gradient
===========================================

"""
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import StericMassAction
from CADETProcess.processModel import Source, GeneralRateModel, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Component System
component_system = ComponentSystem()
component_system.add_component('Salt')
component_system.add_component('Protein')

# Binding Model
binding_model = StericMassAction(component_system, name='SMA')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [0.0, 0.3]
binding_model.desorption_rate = [0.0, 1.5]
binding_model.characteristic_charge = [0.0, 7.0]
binding_model.steric_factor = [0.0, 50.0]
binding_model.capacity = 225.0

# Unit Operations
load = Source(component_system, name='load')
load.c = [180.0, 0.1]

wash = Source(component_system, name='wash')
wash.c = [70.0, 0.0]

elute = Source(component_system, name='elute')
elute.c = [500.0, 0.0]

column = GeneralRateModel(component_system, name='column')
column.binding_model = binding_model

column.length = 0.25
column.diameter = 0.0115
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33
column.axial_dispersion = 2.0e-7
column.film_diffusion = [2.0e-5, 2.0e-7]
column.pore_diffusion = [7e-5, 1e-9]
column.surface_diffusion = [0.0, 0.0]

column.c = [180, 0]
column.q = [binding_model.capacity, 0]

outlet = Sink(component_system, name='outlet')

# flow sheet
flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(load, feed_source=True)
flow_sheet.add_unit(wash, eluent_source=True)
flow_sheet.add_unit(elute, eluent_source=True)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, chromatogram_sink=True)

flow_sheet.add_connection(load, column)
flow_sheet.add_connection(wash, column)
flow_sheet.add_connection(elute, column)
flow_sheet.add_connection(column, outlet)

# Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0

## Create Events and Durations
Q = 2.88e-8
load_duration = 7500.0
gradient_start = 9500.0
gradient_slope = Q/(process.cycle_time - gradient_start)

process.add_event('load_on', 'flow_sheet.load.flow_rate', Q)
process.add_event('load_off', 'flow_sheet.load.flow_rate', 0.0)
process.add_duration('load_duration', time=load_duration)
process.add_event_dependency('load_off', ['load_on', 'load_duration'], [1, 1])

process.add_event('wash_off', 'flow_sheet.wash.flow_rate', 0)
process.add_event(
    'elute_off', 'flow_sheet.elute.flow_rate', 0
)

process.add_event(
    'wash_on', 'flow_sheet.wash.flow_rate', Q, time=load_duration
)
process.add_event_dependency('wash_on', ['load_off'])

process.add_event(
    'wash_gradient', 'flow_sheet.wash.flow_rate',
    [Q, -gradient_slope], gradient_start
    )
process.add_event(
    'elute_gradient', 'flow_sheet.elute.flow_rate', [0, gradient_slope]
    )
process.add_event_dependency('elute_gradient', ['wash_gradient'])

if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    simulation_results = process_simulator.simulate(process)

    from CADETProcess.plotting import SecondaryAxis
    sec = SecondaryAxis()
    sec.component_indices = [0]
    sec.y_label = '$c_{salt}$'

    simulation_results.solution.column.outlet.plot(secondary_axis=sec)
