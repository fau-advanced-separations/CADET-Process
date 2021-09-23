#!/usr/bin/env python3

"""
===========================================
Simulate Load/Wash/Elute with salt gradient 
===========================================

tags:
    steric mass action law
    lwe
    single column
    gradient
    
"""
from CADETProcess.processModel import StericMassAction
from CADETProcess.processModel import Source, GeneralRateModel, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

process_name = flow_sheet_name = 'lwe'

# Binding Model
binding_model = StericMassAction(n_comp=2, name='SMA')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [0.0, 0.3]
binding_model.desorption_rate = [0.0, 1.5]
binding_model.characteristic_charge = [0.0, 7.0]
binding_model.steric_factor = [0.0, 50.0]
binding_model.capacity = 225.0

# Unit Operations
feed = Source(n_comp=2, name='feed')
feed.c = [180.0, 0.1]

eluent = Source(n_comp=2, name='eluent')
eluent.c = [70.0, 0.0]

eluent_salt = Source(n_comp=2, name='eluent_salt')
eluent_salt.c = [500.0, 0.0]

column = GeneralRateModel(n_comp=2, name='column')
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

outlet = Sink(n_comp=2, name='outlet')

# flow sheet
fs = FlowSheet(n_comp=2, name=flow_sheet_name)

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
lwe = Process(fs, name=process_name)
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


## Optional
# sma_refc0 = lwe_model.root.input.model.unit_000.sec_002.lin_coeff[0] * (t_cycle - grad_start)
# lwe_model.root.input.model.unit_001.adsorption.sma_refc0 = sma_refc0
# lwe_model.root.input.model.unit_001.adsorption.sma_refq = lambda_

if __name__ == '__main__':
    from CADETProcess.simulation import Cadet
    process_simulator = Cadet(
        install_path='/usr/local',
    )
    lwe_sim_results = process_simulator.simulate(lwe)
    lwe_sim_results.solution.outlet[0].plot()
