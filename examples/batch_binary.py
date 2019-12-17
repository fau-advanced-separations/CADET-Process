#!/usr/bin/env python3

"""
===============================================
Simulate Batch Chromatography of Binary Mixture
===============================================
"""

from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import Source, Column, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

process_name = flow_sheet_name = 'batch_binary'

# Binding Model
binding_model = Langmuir(n_comp=2, name='langmuir')
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.saturation_capacity = [100, 100]

# Unit Operations
feed = Source(n_comp=2, name='feed')
feed.c = [10, 10]

eluent = Source(n_comp=2, name='eluent')
eluent.c = [0, 0]

column = Column(n_comp=2, name='column')
column.length = 0.6
column.diameter = 0.024
column.axial_dispersion = 4.7e-7
column.total_porosity = 0.7

column.binding_model = binding_model

outlet = Sink(n_comp=2, name='outlet')

# flow sheet
fs = FlowSheet(n_comp=2, name=flow_sheet_name)

fs.add_unit(feed, feed_source=True)
fs.add_unit(eluent, eluent_source=True)
fs.add_unit(column)
fs.add_unit(outlet, chromatogram_sink=True)

fs.add_connection(feed, column)
fs.add_connection(eluent, column)
fs.add_connection(column, outlet)

# Process
batch_binary = Process(fs, name=process_name)

Q = 60/(60*1e6)

# Create Events and Durations
batch_binary.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
batch_binary.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
batch_binary.add_duration('feed_duration', 'feed_on', 'feed_off')

batch_binary.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
batch_binary.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)

batch_binary.add_event_dependency('eluent_on', ['feed_off'])
batch_binary.add_event_dependency('eluent_off', ['feed_on'])
batch_binary.add_event_dependency('feed_off', ['feed_on', 'feed_duration'],[1,1])

# Set process times
batch_binary.cycle_time = 174.7
batch_binary.feed_duration.time = 51.2


if __name__ == '__main__':
    from CADETProcess.simulation import Cadet
    process_simulator = Cadet()
    process_simulator.evaluate_stationarity = False
    process_simulator.n_cycles = 4

    batch_binary_sim_results = process_simulator.simulate(batch_binary)

    from CADETProcess.fractionation import optimize_fractionation
    batch_binary_performance = optimize_fractionation(
            batch_binary_sim_results.chromatograms,
            batch_binary.process_meta,
            0.95)