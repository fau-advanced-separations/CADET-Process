"""
Run every binding model once in a CSTR to check that CADET-Core accepts its configuration.

The adapter tests compare the parameter maps with the binding classes, which share the
same source; only the engine can tell whether a model name or field is what it reads.
Parameter values are placeholders, not physically meaningful; the test only asserts that
the simulation completes.
"""

import pytest
from CADETProcess.dataStructure.parameter import Integer
from CADETProcess.processModel import (
    ComponentSystem,
    Cstr,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
    binding,
)
from CADETProcess.simulator import Cadet

try:
    simulator = Cadet()
except FileNotFoundError:
    simulator = None

BINDING_MODELS = [name for name in binding.__all__ if name != "BindingBaseClass"]

# Component layouts the engine requires beyond two components with one bound state each.
# Non-binding leading components are salt or pH modifiers; the generalized ion exchange
# model declares its non-binding modifier itself.
N_COMP = {"GeneralizedIonExchange": 3}
BOUND_STATES = {
    "HICUnified": [0, 0, 1],
    "HICConstantWaterActivity": [0, 1],
    "HICWaterOnHydrophobicSurfaces": [0, 1],
    "MultiComponentColloidal": [0, 1],
    "AffinityComplexTitration": [0, 1],
    # Salt binds in one state; a single protein state leaves no rate between states.
    "SimplifiedMultistateStericMassAction": [1, 2],
}

# Models that placeholder values of 1 make invalid or too stiff to integrate get the
# protein values of CADET-Core's binding model tests instead; salt entries are zero.
COLLOIDAL_PARAMETERS = {
    "phase_ratio": 4.9e7,
    "kappa_exponential": 1.8,
    "kappa_factor": 1e7,
    "kappa_constant": 1.2,
    "coordination_number": 4,
    "logkeq_ph_exponent": [0, 1.3],
    "logkeq_power_exponent": [0, 1.4],
    "logkeq_power_factor": [0, 1.7],
    "logkeq_exponent_factor": [0, 2.1],
    "logkeq_exponent_multiplier": [0, 2.5],
    "bpp_ph_exponent": [0, 2.5],
    "bpp_power_exponent": [0, 1.1],
    "bpp_power_factor": [0, 2.9],
    "bpp_exponent_factor": [0, 1.3],
    "bpp_exponent_multiplier": [0, 3.0],
    "protein_radius": [0, 1.1e-8],
    "kinetic_rate_constant": [0, 0.9],
}
SIMPLIFIED_MSSMA_PARAMETERS = {
    "adsorption_rate": [0, 3.55, 1.59],
    "desorption_rate": [0, 10, 10],
    "characteristic_charge_first": [0, 1.5],
    "characteristic_charge_last": [0, 2.0],
    "quadratic_modifiers_charge": [0, 0.1],
    "steric_factor_first": [0, 10.6],
    "steric_factor_last": [0, 11.83],
    "quadratic_modifiers_steric": [0, -0.1],
    "exchange_from_weak_stronger": [0, 0.8],
    "linear_exchange_ws": [0, 0.2],
    "quadratic_exchange_ws": [0, -0.1],
    "exchange_from_stronger_weak": [0, 0.9],
    "linear_exchange_sw": [0, 0.1],
    "quadratic_exchange_sw": [0, -0.2],
    "capacity": 100,
}
REFERENCE_PARAMETERS = {
    "MultiComponentColloidal": COLLOIDAL_PARAMETERS,
    "SimplifiedMultistateStericMassAction": SIMPLIFIED_MSSMA_PARAMETERS,
}


def make_binding_model(name: str) -> tuple[ComponentSystem, binding.BindingBaseClass]:
    """Create a binding model with every required parameter set to a placeholder."""
    bound_states = BOUND_STATES.get(name)
    n_comp = len(bound_states) if bound_states else N_COMP.get(name, 2)
    component_system = ComponentSystem(n_comp)
    binding_model = getattr(binding, name)(component_system, name="binding")

    if bound_states:
        binding_model.bound_states = bound_states

    if name in REFERENCE_PARAMETERS:
        binding_model.parameters = REFERENCE_PARAMETERS[name]

    for parameter in binding_model.missing_parameters:
        descriptor = getattr(type(binding_model), parameter)
        setattr(binding_model, parameter, 1 if isinstance(descriptor, Integer) else 1.0)

    return component_system, binding_model


def make_cstr_process(
    component_system: ComponentSystem,
    binding_model: binding.BindingBaseClass,
) -> Process:
    """Create a process feeding a CSTR that carries the binding model."""
    feed = [1] * component_system.n_comp

    inlet = Inlet(component_system, name="inlet")
    inlet.flow_rate = 1e-6
    inlet.c = feed

    # Start filled with the feed, as from an equilibrated buffer; at zero salt, fractional
    # charges make salt-dependent rates non-differentiable. Consistent initialization
    # sets the bound state.
    cstr = Cstr(component_system, name="cstr")
    cstr.c = feed
    cstr.init_liquid_volume = 1e-3
    cstr.const_solid_volume = 1e-4
    cstr.binding_model = binding_model

    outlet = Outlet(component_system, name="outlet")

    flow_sheet = FlowSheet(component_system, name="flow_sheet")
    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(cstr)
    flow_sheet.add_unit(outlet)
    flow_sheet.add_connection(inlet, cstr)
    flow_sheet.add_connection(cstr, outlet)

    process = Process(flow_sheet, name="process")
    process.cycle_time = 10

    return process


@pytest.mark.skipif(simulator is None, reason="CADET-Core is not installed.")
@pytest.mark.parametrize("name", BINDING_MODELS)
def test_binding_model_runs_in_engine(name):
    process = make_cstr_process(*make_binding_model(name))

    simulator.simulate(process)
