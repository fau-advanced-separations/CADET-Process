import numpy as np
import pytest

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
                        "subsolvers": "LEVMAR",
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
                        "subsolvers": "LEVMAR",
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
                        "subsolvers": "LEVMAR",
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
                        "subsolvers": "LEVMAR",
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
                        "subsolvers": "LEVMAR",
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
        ("tubular_reactor", 0, ZeroDivisionError),
        ("lrmp", 0, ZeroDivisionError),
    ],
)
def test_interstitial_velocity(unit_operation, flow_rate, expected_velocity, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = 3 if unit_operation == "tubular_reactor" else [3, 2]

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    if expected_velocity is ZeroDivisionError:
        with pytest.raises(ZeroDivisionError):
            unit.calculate_interstitial_velocity(flow_rate)
    else:
        assert np.isclose(
            unit.calculate_interstitial_velocity(flow_rate), expected_velocity
        )


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_velocity",
    [
        ("tubular_reactor", 2, 2),
        ("lrmp", 2, 2),
        ("tubular_reactor", 0, ZeroDivisionError),
        ("lrmp", 0, ZeroDivisionError),
    ],
)
def test_superficial_velocity(unit_operation, flow_rate, expected_velocity, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    if expected_velocity is ZeroDivisionError:
        with pytest.raises(ZeroDivisionError):
            unit.calculate_superficial_velocity(flow_rate)
    else:
        assert np.isclose(
            unit.calculate_superficial_velocity(flow_rate), expected_velocity
        )


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_ntp",
    [
        ("tubular_reactor", 2, [1 / 3, 1 / 3]),
        ("tubular_reactor", 0, ZeroDivisionError),
    ],
)
def test_ntp(unit_operation, flow_rate, expected_ntp, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = 3

    if expected_ntp is ZeroDivisionError:
        with pytest.raises(ZeroDivisionError):
            unit.NTP(flow_rate)
    else:
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


if __name__ == "__main__":
    pytest.main([__file__])
