#!/usr/bin/env python3

"""
=================================
Simulate Dextran Pulse Experiment
=================================

"""
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Source, LumpedRateModelWithPores, Sink
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

component_system = ComponentSystem(['Dextran'])

inlet = Source(component_system, 'inlet')
inlet.flow_rate = 2.88e-8

column = LumpedRateModelWithPores(component_system, 'column')
column.length = 0.25
column.cross_section_area = 1e-4
column.bed_porosity = 0.4
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33
column.axial_dispersion = 1.0e-7
column.film_diffusion = [0.0]

outlet = Sink(component_system, 'outlet')

flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

process = Process(flow_sheet, 'dextran')
process.cycle_time = 600

process.add_event('inject_on', 'flow_sheet.inlet.c', [1.0], 0)
process.add_event('inject_off', 'flow_sheet.inlet.c', [0.0], 50.0)

if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    simulator = Cadet()

    simulation_results = simulator.simulate(process)

    simulation_results.solution.column.outlet.plot()
