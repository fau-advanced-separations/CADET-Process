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
feed = Source(component_system, name='feed')
feed.c = [180.0, 0.1]

eluent = Source(component_system, name='eluent')
eluent.c = [70.0, 0.0]

eluent_salt = Source(component_system, name='eluent_salt')
eluent_salt.c = [500.0, 0.0]

column = GeneralRateModel(component_system, name='column')
column.length = 0.25
column.diameter = 0.0115
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33
column.axial_dispersion = 2.0e-7
column.film_diffusion = [2.0e-5, 2.0e-7]
column.pore_diffusion = [7e-5, 1e-9]
column.surface_diffusion = [0.0, 0.0]

column.binding_model = binding_model

column.c = [180, 0]
column.q = [binding_model.capacity, 0]

outlet = Sink(component_system, name='outlet')

# flow sheet
fs = FlowSheet(component_system)

fs.add_unit(feed, feed_source=True)
fs.add_unit(eluent, eluent_source=True)
fs.add_unit(eluent_salt, eluent_source=True)
fs.add_unit(column)
fs.add_unit(outlet, chromatogram_sink=True)

fs.add_connection(feed, column)
fs.add_connection(eluent, column)
fs.add_connection(eluent_salt, column)
fs.add_connection(column, outlet)

# Process
lwe = Process(fs, 'lwe')
lwe.cycle_time = 15000.0

## Create Events and Durations
Q = 2.88e-8
feed_duration = 7500.0
gradient_start = 9500.0
gradient_slope = Q/(lwe.cycle_time - gradient_start)

lwe.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
lwe.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
lwe.add_duration('feed_duration', time=feed_duration)
lwe.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])

lwe.add_event('eluent_initialization', 'flow_sheet.eluent.flow_rate', 0)
lwe.add_event(
    'eluent_salt_initialization', 'flow_sheet.eluent_salt.flow_rate', 0
)

lwe.add_event(
    'wash', 'flow_sheet.eluent.flow_rate', Q, time=feed_duration
)
lwe.add_event_dependency('wash', ['feed_off'])

lwe.add_event(
    'neg_grad_start', 'flow_sheet.eluent.flow_rate',
    [Q, -gradient_slope], gradient_start
    )
lwe.add_event(
    'pos_grad_start', 'flow_sheet.eluent_salt.flow_rate', [0, gradient_slope]
    )
lwe.add_event_dependency('pos_grad_start', ['neg_grad_start'])

if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    lwe_sim_results = process_simulator.simulate(lwe)

    from CADETProcess.plotting import SecondaryAxis
    sec = SecondaryAxis()
    sec.component_indices = [0]
    sec.y_label = '$c_{salt}$'

    lwe_sim_results.solution.column.outlet.plot(secondary_axis=sec)
