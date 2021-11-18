#!/usr/bin/env python3

"""
===============================================
Simulate Batch Chromatography of Binary Mixture
===============================================
"""

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import Source, LumpedRateModelWithoutPores, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Component System
component_system = ComponentSystem()
component_system.add_component('A')
component_system.add_component('B')

# Binding Model
binding_model = Langmuir(component_system, name='langmuir')
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.saturation_capacity = [100, 100]

# Unit Operations
feed = Source(component_system, name='feed')
feed.c = [10, 10]

eluent = Source(component_system, name='eluent')
eluent.c = [0, 0]

column = LumpedRateModelWithoutPores(component_system, name='column')
column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7
column.binding_model = binding_model

outlet = Sink(component_system, name='outlet')

# flow sheet
fs = FlowSheet(component_system)

fs.add_unit(feed, feed_source=True)
fs.add_unit(eluent, eluent_source=True)
fs.add_unit(column)
fs.add_unit(outlet, chromatogram_sink=True)

fs.add_connection(feed, column)
fs.add_connection(eluent, column)
fs.add_connection(column, outlet)

# Process
batch_binary = Process(fs, 'batch_binary')

## Create Events and Durations
Q = 60/(60*1e6)
batch_binary.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
batch_binary.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
batch_binary.add_duration('feed_duration')

batch_binary.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
batch_binary.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)

batch_binary.add_event_dependency('eluent_on', ['feed_off'])
batch_binary.add_event_dependency('eluent_off', ['feed_on'])
batch_binary.add_event_dependency('feed_off', ['feed_on', 'feed_duration'],[1,1])

## Set process times
batch_binary.cycle_time = 176.45
batch_binary.feed_duration.time = 51.95

x = [374.4299005082142, 118.65641374452873]
batch_binary.cycle_time = x[0]
batch_binary.feed_duration.time = x[1]


if __name__ == '__main__':
    from CADETProcess.simulation import Cadet
    process_simulator = Cadet()

    process_simulator.evaluate_stationarity = False
    process_simulator.n_cycles = 4
    
    batch_binary_sim_results = process_simulator.simulate(batch_binary)

    from CADETProcess.fractionation import FractionationOptimizer
    purity_required = [0.95, 0.95]
    frac_opt = FractionationOptimizer(purity_required)
    
    batch_binary_frac = frac_opt.optimize_fractionation(
        batch_binary_sim_results.chromatograms,
        batch_binary.process_meta,
    )
    
    batch_binary_frac.plot_fraction_signal()
    performance = batch_binary_frac.performance
    print(performance)