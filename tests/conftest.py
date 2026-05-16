import numpy as np
import pytest
from CADETProcess.processModel import (
    MCT,
    ComponentSystem,
    Cstr,
    GeneralRateModel,
    Inlet,
    LumpedRateModelWithoutPores,
    LumpedRateModelWithPores,
    TubularReactor,
)

length = 0.6
diameter = 0.024
bed_porosity = 0.3
particle_porosity = 0.6
particle_radius = 1e-4
total_porosity = bed_porosity + (1 - bed_porosity) * particle_porosity
axial_dispersion = 4.7e-7
film_diffusion = [0, 1e-6]
pore_diffusion = [0, 1e-11]

nchannel = 3
channel_cross_section_areas = [0.1, 0.1, 0.1]
exchange_matrix = np.array([
    [[0.0], [0.01], [0.0]],
    [[0.02], [0.0], [0.03]],
    [[0.0], [0.0], [0.0]],
])

cross_section_area = np.pi / 4 * diameter**2
init_liquid_volume = cross_section_area * length * total_porosity
const_solid_volume = cross_section_area * length * (1 - total_porosity)


@pytest.fixture
def component_system():
    return ComponentSystem(2)


@pytest.fixture
def inlet(component_system):
    return Inlet(component_system, name="test_inlet")


@pytest.fixture
def cstr(component_system):
    unit = Cstr(component_system, name="test_cstr")
    unit.const_solid_volume = const_solid_volume
    unit.init_liquid_volume = init_liquid_volume
    unit.flow_rate = 1
    return unit


@pytest.fixture
def tubular_reactor(component_system):
    unit = TubularReactor(component_system, name="test_tubular_reactor")
    unit.length = length
    unit.diameter = diameter
    unit.axial_dispersion = axial_dispersion
    return unit


@pytest.fixture
def lrm(component_system):
    unit = LumpedRateModelWithoutPores(component_system, name="test_lrm")
    unit.length = length
    unit.diameter = diameter
    unit.axial_dispersion = axial_dispersion
    unit.total_porosity = total_porosity
    return unit


@pytest.fixture
def lrmp(component_system):
    unit = LumpedRateModelWithPores(component_system, name="test_lrmp")
    unit.length = length
    unit.diameter = diameter
    unit.axial_dispersion = axial_dispersion
    unit.bed_porosity = bed_porosity
    unit.particle_radius = particle_radius
    unit.particle_porosity = particle_porosity
    unit.film_diffusion = film_diffusion
    return unit


@pytest.fixture
def mct():
    unit = MCT(ComponentSystem(1), nchannel=nchannel, name="test_mct")
    unit.length = length
    unit.channel_cross_section_areas = channel_cross_section_areas
    unit.axial_dispersion = axial_dispersion
    unit.exchange_matrix = exchange_matrix
    return unit


@pytest.fixture
def grm(component_system):
    unit = GeneralRateModel(component_system, name="test_grm")
    unit.length = length
    unit.diameter = diameter
    unit.axial_dispersion = axial_dispersion
    unit.bed_porosity = bed_porosity
    unit.particle_radius = particle_radius
    unit.particle_porosity = particle_porosity
    unit.film_diffusion = film_diffusion
    unit.pore_diffusion = pore_diffusion
    return unit
