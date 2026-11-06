import numpy as np
import pytest
from CADETProcess import CADETProcessError

length = 0.6
diameter = 0.024

cross_section_area = np.pi / 4 * diameter**2
volume_liquid = cross_section_area * length
volume = cross_section_area * length

bed_porosity = 0.3
particle_porosity = 0.6
particle_radius = 1e-4
total_porosity = bed_porosity + (1 - bed_porosity) * particle_porosity
const_solid_volume = volume * (1 - total_porosity)
init_liquid_volume = volume * total_porosity

axial_dispersion = 4.7e-7
film_diffusion_0 = 0
film_diffusion_1 = 1e-6
pore_diffusion_0 = 0
pore_diffusion_1 = 1e-11

nchannel = 3
channel_cross_section_areas = [0.1, 0.1, 0.1]
exchange_matrix = np.array([
    [[0.0], [0.01], [0.0]],
    [[0.02], [0.0], [0.03]],
    [[0.0], [0.0], [0.0]]
])
flow_direction = 1


@pytest.mark.parametrize(
    "unit_operation, expected_geometry",
    [
        (
            "cstr",
            {
                "volume_liquid": total_porosity * volume,
                "volume_solid": (1 - total_porosity) * volume,
            },
        ),
        (
            "lrm",
            {
                "cross_section_area": cross_section_area,
                "total_porosity": total_porosity,
                "volume": volume,
                "volume_interstitial": total_porosity * volume,
                "volume_liquid": total_porosity * volume,
                "volume_solid": (1 - total_porosity) * volume,
            },
        ),
        (
            "lrmp",
            {
                "cross_section_area": cross_section_area,
                "total_porosity": total_porosity,
                "volume": volume,
                "volume_interstitial": bed_porosity * volume,
                "volume_liquid": total_porosity * volume,
                "volume_solid": (1 - total_porosity) * volume,
            },
        ),
        (
            "mct",
            {
                "volume": length * sum(channel_cross_section_areas),
            },
        ),
        (
            "grm2d",
            {
                "cross_section_area": cross_section_area,
                "total_porosity": total_porosity,
                "volume": volume,
                "volume_interstitial": bed_porosity * volume,
                "volume_liquid": total_porosity * volume,
                "volume_solid": (1 - total_porosity) * volume,
            },
        ),
    ],
)
def test_geometry(unit_operation, expected_geometry, request):
    unit = request.getfixturevalue(unit_operation)

    if "total_porosity" in expected_geometry:
        assert unit.total_porosity == expected_geometry["total_porosity"]

    if "volume" in expected_geometry:
        assert unit.volume == expected_geometry["volume"]

    if "volume_interstitial" in expected_geometry:
        assert np.isclose(
            unit.volume_interstitial, expected_geometry["volume_interstitial"]
        )

    if "volume_liquid" in expected_geometry:
        assert np.isclose(unit.volume_liquid, expected_geometry["volume_liquid"])

    if "volume_solid" in expected_geometry:
        assert np.isclose(unit.volume_solid, expected_geometry["volume_solid"])

    if "cross_section_area" in expected_geometry:
        assert unit.cross_section_area == expected_geometry["cross_section_area"]

        unit.cross_section_area = cross_section_area / 2
        assert np.isclose(unit.diameter, diameter / (2**0.5))


radius = diameter / 2


@pytest.mark.parametrize(
    "radial_disc_type, radial_compartments, expected_bed_porosity",
    [
        # Cell areas scale with 1, 3, 5, 7, 9 (out of 25)
        ("EQUIDISTANT", None, 0.48),
        # Equal cell areas
        ("EQUIVOLUME", None, 0.4),
        # Cell areas 0.25, 0.11, 0.13, 0.15, 0.36 of the cross section
        ("USER_DEFINED", [0, 0.5, 0.6, 0.7, 0.8, 1.0], 0.426),
    ],
)
def test_grm2d_radially_varying_porosity(
    grm2d, radial_disc_type, radial_compartments, expected_bed_porosity
):
    grm2d.discretization.radial_disc_type = radial_disc_type
    if radial_compartments is not None:
        grm2d.discretization.radial_compartments = [
            r * radius for r in radial_compartments
        ]
    grm2d.bed_porosity = [0.2, 0.3, 0.4, 0.5, 0.6]

    expected_total_porosity = (
        expected_bed_porosity + (1 - expected_bed_porosity) * particle_porosity
    )

    assert np.isclose(grm2d.total_porosity, expected_total_porosity)
    assert np.isclose(
        grm2d.cross_section_area_interstitial,
        expected_bed_porosity * cross_section_area,
    )
    assert np.isclose(grm2d.volume_interstitial, expected_bed_porosity * volume)
    assert np.isclose(grm2d.volume_solid, (1 - expected_total_porosity) * volume)


@pytest.mark.parametrize("porosity", [0.3, [0.2, 0.3, 0.4, 0.5, 0.6]])
def test_grm2d_rejects_diameter_from_interstitial_velocity(grm2d, porosity):
    grm2d.bed_porosity = porosity
    diameter_before = grm2d.diameter
    with pytest.raises(CADETProcessError, match="for a 2D column"):
        grm2d.set_diameter_from_interstitial_velocity(Q=1e-6, u0=1e-3)
    assert grm2d.diameter == diameter_before


@pytest.mark.parametrize(
    "input_c, expected_c",
    [
        (1, np.array([[1, 0, 0, 0], [1, 0, 0, 0]])),
        ([1, 1], np.array([[1, 0, 0, 0], [1, 0, 0, 0]])),
        ([1, 2], np.array([[1, 0, 0, 0], [2, 0, 0, 0]])),
        ([[1, 0], [2, 0]], np.array([[1, 0, 0, 0], [2, 0, 0, 0]])),
        ([[1, 2], [3, 4]], np.array([[1, 2, 0, 0], [3, 4, 0, 0]])),
    ],
)
def test_polynomial_inlet_concentration(inlet, input_c, expected_c):
    inlet.c = input_c
    np.testing.assert_equal(inlet.c, expected_c)


@pytest.mark.parametrize(
    "unit_operation, input_flow_rate, expected_flow_rate",
    [
        ("inlet", 1, np.array([1, 0, 0, 0])),
        ("inlet", [1, 0], np.array([1, 0, 0, 0])),
        ("inlet", [1, 1], np.array([1, 1, 0, 0])),
        ("cstr", 1, np.array([1, 0, 0, 0])),
        ("cstr", [1, 0], np.array([1, 0, 0, 0])),
        ("cstr", [1, 1], np.array([1, 1, 0, 0])),
    ],
)
def test_polynomial_flow_rate(
    unit_operation, input_flow_rate, expected_flow_rate, request
):
    unit = request.getfixturevalue(unit_operation)
    unit.flow_rate = input_flow_rate
    np.testing.assert_equal(unit.flow_rate, expected_flow_rate)


@pytest.mark.parametrize(
    "unit_operation, expected_parameters",
    [
        (
            "cstr",
            {
                "flow_rate": np.array([1, 0, 0, 0]),
                "init_liquid_volume": init_liquid_volume,
                "flow_rate_filter": 0,
                "c": [0, 0],
                "q": None,
                "const_solid_volume": const_solid_volume,
            },
        ),
        (
            "tubular_reactor",
            {
                "length": length,
                "diameter": diameter,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "flow_direction": flow_direction,
                "c": [0, 0],
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
        (
            "lrm",
            {
                "length": length,
                "diameter": diameter,
                "total_porosity": total_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "flow_direction": flow_direction,
                "c": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
        (
            "lrmp",
            {
                "length": length,
                "diameter": diameter,
                "bed_porosity": bed_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "pore_accessibility": [1, 1],
                "film_diffusion": [film_diffusion_0, film_diffusion_1],
                "particle_radius": particle_radius,
                "particle_porosity": particle_porosity,
                "flow_direction": flow_direction,
                "c": [0, 0],
                "cp": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "par_geom": "SPHERE",
                    "use_analytic_jacobian": True,
                    "gs_type": True,
                    "max_krylov": 0,
                    "max_restarts": 10,
                    "schur_safety": 1e-08,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
        (
            "grm",
            {
                "length": length,
                "diameter": diameter,
                "bed_porosity": bed_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "pore_accessibility": [1, 1],
                "film_diffusion": [film_diffusion_0, film_diffusion_1],
                "particle_radius": particle_radius,
                "particle_porosity": particle_porosity,
                "pore_diffusion": [pore_diffusion_0, pore_diffusion_1],
                "surface_diffusion": None,
                "surface_diffusion_dependence": "NONE",
                "flow_direction": flow_direction,
                "c": [0, 0],
                "cp": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "par_geom": "SPHERE",
                    "npar": 5,
                    "par_disc_type": "EQUIDISTANT_PAR",
                    "par_boundary_order": 2,
                    "fix_zero_surface_diffusion": False,
                    "use_analytic_jacobian": True,
                    "gs_type": True,
                    "max_krylov": 0,
                    "max_restarts": 10,
                    "schur_safety": 1e-08,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
        (
            "grm2d",
            {
                "length": length,
                "diameter": diameter,
                "bed_porosity": [bed_porosity] * 5,
                "axial_dispersion": [axial_dispersion, axial_dispersion] * 5,
                "col_dispersion_radial": [1e-10] * 5 * 2,
                "pore_accessibility": [1, 1],
                "film_diffusion": [film_diffusion_0, film_diffusion_1],
                "particle_radius": particle_radius,
                "particle_porosity": particle_porosity,
                "pore_diffusion": [pore_diffusion_0, pore_diffusion_1],
                "surface_diffusion": None,
                "flow_direction": flow_direction,
                "c": [0, 0] * 5,
                "cp": [0, 0] * 5,
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "nrad": 5,
                    "radial_disc_type": "EQUIDISTANT",
                    "par_geom": "SPHERE",
                    "npar": 5,
                    "par_disc_type": "EQUIDISTANT_PAR",
                    "par_boundary_order": 2,
                    "fix_zero_surface_diffusion": False,
                    "use_analytic_jacobian": True,
                    "gs_type": True,
                    "max_krylov": 0,
                    "max_restarts": 10,
                    "schur_safety": 1e-08,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
        (
            "mct",
            {
                "nchannel": nchannel,
                "length": length,
                "channel_cross_section_areas": channel_cross_section_areas,
                "axial_dispersion": nchannel * [axial_dispersion],
                "exchange_matrix": exchange_matrix,
                "flow_direction": 1,
                "c": [[0, 0, 0]],
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": ["ATRN_ERR", "LEVMAR"],
                    },
                },
            },
        ),
    ],
)
def test_parameters(unit_operation, expected_parameters, request):
    unit = request.getfixturevalue(unit_operation)
    np.testing.assert_equal(expected_parameters, unit.parameters)


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_velocity",
    [
        ("tubular_reactor", 2, 2),
        ("lrmp", 2, 4),
        ("tubular_reactor", 0, 0),
        ("lrmp", 0, 0),
    ],
)
def test_interstitial_velocity(unit_operation, flow_rate, expected_velocity, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = 3 if unit_operation == "tubular_reactor" else [3, 2]

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    assert np.isclose(
        unit.calculate_interstitial_velocity(flow_rate), expected_velocity
    )


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_velocity",
    [
        ("tubular_reactor", 2, 2),
        ("lrmp", 2, 2),
        ("tubular_reactor", 0, 0),
        ("lrmp", 0, 0),
    ],
)
def test_superficial_velocity(unit_operation, flow_rate, expected_velocity, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    assert np.isclose(
        unit.calculate_superficial_velocity(flow_rate), expected_velocity
    )


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_ntp",
    [
        ("tubular_reactor", 2, [1 / 3, 1 / 3]),
        ("tubular_reactor", 0, [0, 0]),
    ],
)
def test_ntp(unit_operation, flow_rate, expected_ntp, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = 3

    np.testing.assert_almost_equal(unit.NTP(flow_rate), expected_ntp)


@pytest.mark.parametrize(
    "unit_operation, flow_rate, axial_dispersion, expected_bodenstein",
    [
        ("tubular_reactor", 2, 3, [2 / 3, 2 / 3]),
        ("lrmp", 2, [3, 2], [4 / 3, 2]),
        ("lrmp", 2, [1, 2], [4, 2]),
    ],
)
def test_bodenstein_number(
    unit_operation, flow_rate, axial_dispersion, expected_bodenstein, request
):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = axial_dispersion

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    np.testing.assert_almost_equal(
        unit.calculate_bodenstein_number(flow_rate), expected_bodenstein
    )


# Factors each surface diffusion dependence law uses; the colloidal affinity law reuses
# the exponential and power factors.
SURFACE_DIFFUSION_FACTORS = {
    "LIQUID_SALT_EXPONENTIAL": [
        "surface_diffusion_exponent_factor",
        "surface_diffusion_exponent_multiplier",
    ],
    "LIQUID_SALT_POWER": [
        "surface_diffusion_power_factor",
        "surface_diffusion_power_exponent",
    ],
    "LIQUID_SALT_COLLOIDAL_AFFINITY": [
        "surface_diffusion_exponent_factor",
        "surface_diffusion_exponent_multiplier",
        "surface_diffusion_power_factor",
        "surface_diffusion_power_exponent",
        "surface_diffusion_logkeq_factor",
        "surface_diffusion_logkeq_exponent",
        "surface_diffusion_logkeq_constant",
    ],
}
ALL_SURFACE_DIFFUSION_FACTORS = SURFACE_DIFFUSION_FACTORS[
    "LIQUID_SALT_COLLOIDAL_AFFINITY"
]


@pytest.fixture
def grm_with_binding(grm):
    from CADETProcess.processModel import Langmuir

    grm.binding_model = Langmuir(grm.component_system)
    return grm


def test_surface_diffusion_dependence_requires_binding_model(grm):
    grm.surface_diffusion_dependence = "NONE"

    with pytest.raises(CADETProcessError, match="without binding model"):
        grm.surface_diffusion_dependence = "LIQUID_SALT_POWER"


@pytest.mark.parametrize("dependence", SURFACE_DIFFUSION_FACTORS)
def test_surface_diffusion_dependence_requires_its_factors(
    grm_with_binding, dependence
):
    factors = SURFACE_DIFFUSION_FACTORS[dependence]
    # A factor of another law does not count towards the selected one.
    unused = [f for f in ALL_SURFACE_DIFFUSION_FACTORS if f not in factors]
    for factor in unused:
        setattr(grm_with_binding, factor, [1.0, 1.0])

    grm_with_binding.surface_diffusion_dependence = dependence
    assert set(factors) <= set(grm_with_binding.missing_parameters)

    for factor in factors:
        setattr(grm_with_binding, factor, [-0.5, 2.0])
    assert not set(ALL_SURFACE_DIFFUSION_FACTORS) & set(
        grm_with_binding.missing_parameters
    )


def test_surface_diffusion_factors_not_required_without_dependence(grm_with_binding):
    assert not set(ALL_SURFACE_DIFFUSION_FACTORS) & set(
        grm_with_binding.missing_parameters
    )


@pytest.mark.parametrize("dependence", SURFACE_DIFFUSION_FACTORS)
def test_surface_diffusion_dependence_round_trips_through_parameters(
    grm_with_binding, dependence
):
    grm_with_binding.surface_diffusion_dependence = dependence
    for value, factor in enumerate(ALL_SURFACE_DIFFUSION_FACTORS):
        setattr(grm_with_binding, factor, [value, -value])

    copy = type(grm_with_binding)(grm_with_binding.component_system, name="copy")
    copy.binding_model = grm_with_binding.binding_model
    copy.parameters = grm_with_binding.parameters

    assert copy.surface_diffusion_dependence == dependence
    for value, factor in enumerate(ALL_SURFACE_DIFFUSION_FACTORS):
        assert getattr(copy, factor) == [value, -value]


@pytest.mark.parametrize("dependence", ["NONE", *SURFACE_DIFFUSION_FACTORS])
def test_surface_diffusion_export_contains_only_used_factors(
    grm_with_binding, dependence
):
    from CADETProcess.simulator import Cadet

    try:
        simulator = Cadet()
    except FileNotFoundError:
        pytest.skip("CADET-Core is not installed.")

    if dependence != "NONE":
        grm_with_binding.surface_diffusion_dependence = dependence
    for factor in ALL_SURFACE_DIFFUSION_FACTORS:
        setattr(grm_with_binding, factor, [1.0, 1.0])

    unit_config = simulator.get_unit_config(grm_with_binding)

    exported = {
        key for key in unit_config if key.startswith("PAR_SURFDIFFUSION_")
    } - {"PAR_SURFDIFFUSION_MULTIPLEX"}
    expected_keys = {
        "LIQUID_SALT_EXPONENTIAL": {"EXPFACTOR", "EXPARGMULT"},
        "LIQUID_SALT_POWER": {"POWFACTOR", "POWEXP"},
        "LIQUID_SALT_COLLOIDAL_AFFINITY": {
            "EXPFACTOR",
            "EXPARGMULT",
            "POWFACTOR",
            "POWEXP",
            "LOGKEQFACTOR",
            "LOGKEQEXP",
            "LOGKEQCONST",
        },
    }.get(dependence, set())
    assert exported == {"PAR_SURFDIFFUSION_DEP"} | {
        f"PAR_SURFDIFFUSION_{key}" for key in expected_keys
    }
    assert unit_config["PAR_SURFDIFFUSION_DEP"] == dependence


@pytest.mark.parametrize("unit_operation", ["lrmp", "grm"])
def test_setting_parameters_leaves_input_unchanged(unit_operation, request):
    import copy

    unit = request.getfixturevalue(unit_operation)
    parameters = copy.deepcopy(dict(unit.parameters))
    expected = copy.deepcopy(parameters)

    target = type(unit)(unit.component_system, name="target")
    target.parameters = parameters

    np.testing.assert_equal(parameters, expected)


if __name__ == "__main__":
    pytest.main([__file__])
