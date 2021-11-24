from addict import Dict
import numpy as np

from CADETProcess.processModel import Source, Sink
from CADETProcess.processModel import Cstr, LumpedRateModelWithoutPores
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process
from CADETProcess.simulation import Cadet

def simulate_solid_equilibria(binding_model, buffer, unit_model='cstr', flush=None):
    process_name = flow_sheet_name = 'initial_conditions'
    component_system = binding_model.component_system

    # Unit Operations
    buffer_source = Source(component_system, name='buffer')
    buffer_source.c = buffer

    if flush is None:
        flush = buffer
    flush_source = Source(component_system, 'flush')
    flush_source.c = flush

    if unit_model == 'cstr':
        unit = Cstr(component_system, 'cstr')
        unit.porosity = 0.5
        unit.V = 1e-6

        Q = 1e-6
        cycle_time = 1000*unit.volume/Q
        unit.flow_rate = Q
    elif unit_model == 'column':
        unit = LumpedRateModelWithoutPores(component_system, name='column')
        unit.length = 0.1
        unit.diameter = 0.01
        unit.axial_dispersion = 1e-6
        unit.total_porosity = 0.7

        Q = 60/(60*1e6)
        cycle_time = 10*unit.volume/Q

    try:
        q = binding_model.n_comp * binding_model.n_states * [0]
        q[0] = binding_model.capacity
        unit.q = q
    except AttributeError:
        pass

    unit.binding_model = binding_model

    outlet = Sink(component_system, name='outlet')

    # flow sheet
    fs = FlowSheet(component_system, name=flow_sheet_name)

    fs.add_unit(buffer_source)
    fs.add_unit(flush_source)
    fs.add_unit(unit)
    fs.add_unit(outlet, chromatogram_sink=True)

    fs.add_connection(buffer_source, unit)
    fs.add_connection(flush_source, unit)
    fs.add_connection(unit, outlet)

    # Process
    proc = Process(fs, name=process_name)

    proc.cycle_time = cycle_time

    ## Create Events and Durations
    proc.add_event('buffer_on', 'flow_sheet.buffer.flow_rate', Q)
    proc.add_event('buffer_off', 'flow_sheet.buffer.flow_rate', 0, 0.9*cycle_time)

    proc.add_event('eluent_off', 'flow_sheet.flush.flow_rate', 0.0, 0.0)
    proc.add_event('eluent_on', 'flow_sheet.flush.flow_rate', Q, 0.9*cycle_time)

    # Simulator
    process_simulator = Cadet()
    process_simulator.unit_return_parameters.write_solution_bulk = True
    process_simulator.unit_return_parameters.write_solution_solid = True

    proc_results = process_simulator.simulate(proc)

    if unit_model == 'cstr':
        init_q = proc_results.solution[unit.name].solid.solution[-1,:]
    elif unit_model == 'column':
        init_q = proc_results.solution[unit.name].solid.solution[-1,0,:]

    return init_q.tolist()
