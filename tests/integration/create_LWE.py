import numpy as np
from CADETProcess.processModel import (
    MCT,
    ComponentSystem,
    Cstr,
    FlowSheet,
    GeneralRateModel,
    Inlet,
    LumpedRateModelWithoutPores,
    LumpedRateModelWithPores,
    Outlet,
    Process,
    StericMassAction,
    TubularReactor,
)


def create_lwe(unit_type: str = "GeneralRateModel", **kwargs) -> Process:
    """
    Create a process with the specified unit type and configuration.

    Parameters
    ----------
    unit_type : str, optional
        The type of unit operation, by default 'GeneralRateModel'.
    **kwargs : dict
        Additional parameters for configuring the unit operation.

    Returns
    -------
    Process
        The configured process.
    """
    n_comp: int = kwargs.get("n_comp", 4)
    component_system = ComponentSystem(n_comp)

    if unit_type == "Cstr":
        unit = configure_cstr(component_system, **kwargs)
    elif unit_type == "GeneralRateModel":
        unit = configure_general_rate_model(component_system, **kwargs)
    elif unit_type == "TubularReactor":
        unit = configure_tubular_reactor(component_system, **kwargs)
    elif unit_type == "LumpedRateModelWithoutPores":
        unit = configure_lumped_rate_model_without_pores(component_system, **kwargs)
    elif unit_type == "LumpedRateModelWithPores":
        unit = configure_lumped_rate_model_with_pores(component_system, **kwargs)
    elif unit_type == "MCT":
        unit = configure_multichannel_transport_model(component_system, **kwargs)
    else:
        raise ValueError(f"Unknown unit operation type {unit_type}")

    flow_sheet = setup_flow_sheet(unit, component_system)

    process = Process(flow_sheet, "process")
    process.cycle_time = 120 * 60

    c1_lwe = [[50.0], [0.0], [[100.0, 0.2]]]
    cx_lwe = [[1.0], [0.0], [0.0]]

    process.add_event(
        "load", "flow_sheet.inlet.c", c1_lwe[0] + cx_lwe[0] * (n_comp - 1), 0
    )
    process.add_event(
        "wash", "flow_sheet.inlet.c", c1_lwe[1] + cx_lwe[1] * (n_comp - 1), 10
    )
    process.add_event(
        "elute", "flow_sheet.inlet.c", c1_lwe[2] + cx_lwe[2] * (n_comp - 1), 90
    )

    return process


def setup_flow_sheet(unit, component_system: ComponentSystem) -> FlowSheet:
    """
    Set up the flow sheet for the process.

    Parameters
    ----------
    unit : UnitOperation
        The unit operation to be added to the flow sheet.
    component_system : ComponentSystem
        The component system of the process.

    Returns
    -------
    FlowSheet
        The configured flow sheet.
    """
    flow_sheet = FlowSheet(component_system)
    inlet = Inlet(component_system, name="inlet")
    inlet.flow_rate = 1.2e-3
    outlet = Outlet(component_system, name="outlet")

    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(unit)
    flow_sheet.add_unit(outlet)

    if unit.has_ports:
        flow_sheet.add_connection(inlet, unit, destination_port="channel_0")
        flow_sheet.add_connection(unit, outlet, origin_port="channel_0")
    else:
        flow_sheet.add_connection(inlet, unit)
        flow_sheet.add_connection(unit, outlet)

    return flow_sheet


def configure_cstr(component_system: ComponentSystem, **kwargs) -> Cstr:
    """
    Configure a continuous stirred-tank reactor (CSTR).

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the CSTR.

    Returns
    -------
    Cstr
        The configured CSTR.
    """
    cstr = Cstr(component_system, name="Cstr")

    total_volume = 1e-3
    total_porosity = 0.37 + (1.0 - 0.37) * 0.75
    cstr.init_liquid_volume = total_porosity * total_volume
    cstr.const_solid_volume = (1 - total_porosity) * total_volume

    configure_solution_recorder(cstr, **kwargs)
    configure_steric_mass_action(cstr, component_system, **kwargs)

    return cstr


def configure_general_rate_model(
    component_system: ComponentSystem, **kwargs
) -> GeneralRateModel:
    """
    Configure a general rate model.

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the general rate model.

    Returns
    -------
    GeneralRateModel
        The configured general rate model.
    """
    grm = GeneralRateModel(component_system, name="GeneralRateModel")

    grm.length = 0.014
    grm.diameter = 0.01 * 2
    grm.bed_porosity = 0.37
    grm.axial_dispersion = 5.75e-8
    grm.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]

    configure_solution_recorder(grm, **kwargs)
    configure_discretization(grm, **kwargs)
    configure_particles(grm)
    configure_steric_mass_action(grm, component_system, **kwargs)
    configure_film_diffusion(grm, component_system.n_comp)
    configure_flow_direction(grm, **kwargs)

    return grm


def configure_tubular_reactor(
    component_system: ComponentSystem, **kwargs
) -> TubularReactor:
    """
    Configure a tubular reactor.

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the tubular reactor.

    Returns
    -------
    TubularReactor
        The configured tubular reactor.
    """
    tr = TubularReactor(component_system, name="TubularReactor")

    tr.length = 0.014
    tr.diameter = 0.01 * 2
    tr.axial_dispersion = 5.75e-8

    configure_solution_recorder(tr, **kwargs)
    configure_discretization(tr, **kwargs)
    configure_flow_direction(tr, **kwargs)

    return tr


def configure_lumped_rate_model_without_pores(
    component_system: ComponentSystem, **kwargs
) -> LumpedRateModelWithoutPores:
    """
    Configure a lumped rate model without pores.

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the lumped rate model.

    Returns
    -------
    LumpedRateModelWithoutPores
        The configured lumped rate model.
    """
    lrm = LumpedRateModelWithoutPores(
        component_system, name="LumpedRateModelWithoutPores"
    )

    lrm.length = 0.014
    lrm.diameter = 0.01 * 2
    lrm.total_porosity = 0.37 + (1.0 - 0.37) * 0.75
    lrm.axial_dispersion = 5.75e-8

    configure_solution_recorder(lrm, **kwargs)
    configure_discretization(lrm, **kwargs)
    configure_steric_mass_action(lrm, component_system, **kwargs)
    configure_flow_direction(lrm, **kwargs)

    return lrm


def configure_lumped_rate_model_with_pores(
    component_system: ComponentSystem, **kwargs
) -> LumpedRateModelWithPores:
    """
    Configure a lumped rate model with pores.

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the lumped rate model.

    Returns
    -------
    LumpedRateModelWithPores
        The configured lumped rate model.
    """
    lrmp = LumpedRateModelWithPores(component_system, name="LumpedRateModelWithPores")

    lrmp.length = 0.014
    lrmp.diameter = 0.01 * 2
    lrmp.bed_porosity = 0.37
    lrmp.axial_dispersion = 5.75e-8

    configure_solution_recorder(lrmp, **kwargs)
    configure_discretization(lrmp, **kwargs)
    configure_particles(lrmp)
    configure_steric_mass_action(lrmp, component_system, **kwargs)
    configure_film_diffusion(lrmp, component_system.n_comp)
    configure_flow_direction(lrmp, **kwargs)

    return lrmp


def configure_multichannel_transport_model(
    component_system: ComponentSystem, **kwargs
) -> MCT:
    """
    Configure a multichannel transport model.

    Parameters
    ----------
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the multichannel transport model.

    Returns
    -------
    MCT
        The configured multichannel transport model.
    """
    mct = MCT(component_system, nchannel=3, name="MCT")

    mct.length = 0.014
    mct.channel_cross_section_areas = 3 * [2 * np.pi * (0.01**2)]
    mct.axial_dispersion = 5.75e-8

    n_comp: int = component_system.n_comp

    mct.exchange_matrix = np.array(
        [
            [n_comp * [0.0], n_comp * [0.001], n_comp * [0.0]],
            [n_comp * [0.002], n_comp * [0.0], n_comp * [0.003]],
            [n_comp * [0.0], n_comp * [0.0], n_comp * [0.0]],
        ]
    )

    configure_solution_recorder(mct, **kwargs)
    configure_discretization(mct, **kwargs)
    configure_flow_direction(mct, **kwargs)

    return mct


def configure_discretization(unit_operation, **kwargs) -> None:
    """
    Configure the discretization settings for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    **kwargs : dict
        Additional parameters for configuring the discretization.
    """
    n_col: int = kwargs.get("n_col", 100)
    n_par: int = kwargs.get("n_par", 2)
    ad_jacobian: bool = kwargs.get("ad_jacobian", False)

    if "npar" in unit_operation.discretization.parameters:
        unit_operation.discretization.npar = n_par
        unit_operation.discretization.par_disc_type = "EQUIDISTANT_PAR"

    unit_operation.discretization.ncol = n_col
    unit_operation.discretization.use_analytic_jacobian = not ad_jacobian


def configure_particles(unit_operation) -> None:
    """
    Configure the particle settings for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    """
    par_radius = 4.5e-5
    par_porosity = 0.75

    unit_operation.particle_radius = par_radius
    unit_operation.particle_porosity = par_porosity
    unit_operation.discretization.par_geom = "SPHERE"


def configure_steric_mass_action(unit_operation, component_system, **kwargs) -> None:
    """
    Configure the steric mass action binding model for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    component_system : ComponentSystem
        The component system of the process.
    **kwargs : dict
        Additional parameters for configuring the steric mass action binding model.
    """
    is_kinetic = kwargs.get("is_kinetic", True)

    kA = 35.5
    kD = 1000.0
    nu = 4.7
    sigma = 11.83
    sma_lambda = 1.2e3

    binding_model = StericMassAction(component_system)

    binding_model.is_kinetic = is_kinetic
    binding_model.n_binding_sites = 1
    binding_model.adsorption_rate = kA
    binding_model.desorption_rate = kD
    binding_model.characteristic_charge = nu
    binding_model.steric_factor = sigma
    binding_model.capacity = sma_lambda

    unit_operation.binding_model = binding_model


def configure_film_diffusion(unit_operation, n_comp) -> None:
    """
    Configure the film diffusion settings for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    n_comp : int
        The number of components in the process.
    """
    unit_operation.film_diffusion = [6.9e-6] * n_comp


def configure_flow_direction(unit_operation, **kwargs) -> None:
    """
    Configure the flow direction for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    **kwargs : dict
        Additional parameters for configuring the flow direction.
    """
    reverse_flow = kwargs.get("reverse_flow", False)
    unit_operation.flow_direction = -1 if reverse_flow else 1


def configure_solution_recorder(unit_operation, **kwargs) -> None:
    """
    Configure the solution recorder for a unit operation.

    Parameters
    ----------
    unit_operation : UnitOperation
        The unit operation to be configured.
    **kwargs : dict
        Additional parameters for configuring the solution recorder.
    """
    for write_solution, value in unit_operation.solution_recorder.parameters.items():
        if value is False:
            unit_operation.solution_recorder.parameters[write_solution] = True
