from typing import Optional

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import (
    BindingBaseClass,
    Cstr,
    FlowSheet,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    Process,
)
from CADETProcess.simulator import Cadet


def simulate_solid_equilibria(
    binding_model: BindingBaseClass,
    buffer: list,
    unit_model: Optional[str] = "cstr",
    flush: Optional[list] = None,
    cadet_install_path: Optional[str] = None,
) -> list:
    """
    Simulate initial conditions for solid phase for given buffer.

    Parameters
    ----------
    binding_model : BindingBase
        Binding model describing relation between bulk and solif phase.
    buffer : list
        Buffer concentration in mM.
    unit_model : {'cstr', 'column'}, optional
        Unit model to be used in simulation. The default is 'cstr'.
    flush : list, optional
        Additional buffer for flushing column after loading.
        The default is None.
    cadet_install_path : str, optional
        Path to CADET installation. If None, the default CADET path is used.

    Raises
    ------
    CADETProcessError
        DESCRIPTION.

    Returns
    -------
    list
        Initial conditions for solid phase.
    """
    process_name = flow_sheet_name = "initial_conditions"
    component_system = binding_model.component_system

    # Unit Operations
    buffer_source = Inlet(component_system, name="buffer")
    buffer_source.c = buffer

    if flush is None:
        flush = buffer
    flush_source = Inlet(component_system, "flush")
    flush_source.c = flush

    if unit_model == "cstr":
        unit = Cstr(component_system, "cstr")
        unit.init_liquid_volume = 5e-7
        unit.const_solid_volume = 5e-7

        Q = 1e-6
        cycle_time = np.round(1000 * unit.volume / Q)
        unit.flow_rate = Q
    elif unit_model == "column":
        unit = LumpedRateModelWithoutPores(component_system, name="column")
        unit.length = 0.1
        unit.diameter = 0.01
        unit.axial_dispersion = 1e-6
        unit.total_porosity = 0.7

        Q = 60 / (60 * 1e6)
        cycle_time = np.round(10 * unit.volume / Q)
    else:
        raise CADETProcessError("Unknown unit model")

    try:
        q = binding_model.n_comp * binding_model.n_states * [0]
        capacity = binding_model.capacity
        if isinstance(capacity, list):
            capacity = capacity[0]

        q[0] = capacity
        unit.q = q
        c = binding_model.n_comp * [0]
        c[0] = buffer[0]
        unit.c = c
    except AttributeError:
        pass

    unit.binding_model = binding_model

    unit.solution_recorder.write_solution_bulk = True
    unit.solution_recorder.write_solution_solid = True

    outlet = Outlet(component_system, name="outlet")

    # flow sheet
    fs = FlowSheet(component_system, name=flow_sheet_name)

    fs.add_unit(buffer_source)
    fs.add_unit(flush_source)
    fs.add_unit(unit)
    fs.add_unit(outlet, product_outlet=True)

    fs.add_connection(buffer_source, unit)
    fs.add_connection(flush_source, unit)
    fs.add_connection(unit, outlet)

    # Process
    proc = Process(fs, name=process_name)
    proc.cycle_time = cycle_time

    # Create Events and Durations
    proc.add_event("buffer_on", "flow_sheet.buffer.flow_rate", Q)
    proc.add_event("buffer_off", "flow_sheet.buffer.flow_rate", 0, 0.9 * cycle_time)

    proc.add_event("eluent_off", "flow_sheet.flush.flow_rate", 0.0, 0.0)
    proc.add_event("eluent_on", "flow_sheet.flush.flow_rate", Q, 0.9 * cycle_time)

    # Simulator
    process_simulator = Cadet(cadet_install_path)
    proc_results = process_simulator.simulate(proc)

    if unit_model == "cstr":
        init_q = proc_results.solution[unit.name].solid.solution[-1, :]
    elif unit_model == "column":
        init_q = proc_results.solution[unit.name].solid.solution[-1, 0, :]

    init_q = np.round(init_q, 14)
    return init_q.tolist()
