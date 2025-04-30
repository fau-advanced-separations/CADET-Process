import shutil
import unittest
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.testing as npt
import pytest
from CADETProcess import CADETProcessError, SimulationResults
from CADETProcess.processModel import Process
from CADETProcess.processModel.discretization import NoDiscretization
from CADETProcess.simulator import Cadet

from tests.create_LWE import create_lwe


def detect_cadet(install_path: Optional[Path] = None):
    try:
        simulator = Cadet(install_path)
        found_cadet = True
        install_path = simulator.install_path
    except FileNotFoundError:
        found_cadet = False

    return found_cadet, install_path


install_path = None
found_cadet, install_path = detect_cadet()


class Test_Adapter(unittest.TestCase):
    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_check_cadet(self):
        simulator = Cadet(install_path)

        self.assertTrue(simulator.check_cadet())

        file_name = simulator.get_tempfile_name()
        cwd = simulator.temp_dir
        sim = simulator.get_new_cadet_instance()
        sim.create_lwe(cwd / file_name)

        # Check for CADET-Python >= v1.1, which introduced the .run_simulation interface.
        # If it's not present, assume CADET-Python <= 1.0.4 and use the old .run_load() interface
        # This check can be removed at some point in the future.
        if hasattr(sim, "run_simulation"):
            return_information = sim.run_simulation()
        else:
            return_information = sim.run_load()

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_create_lwe(self):
        simulator = Cadet(install_path)

        file_name = simulator.get_tempfile_name()
        cwd = simulator.temp_dir
        sim = simulator.get_new_cadet_instance()
        sim.create_lwe(cwd / file_name)

        # Check for CADET-Python >= v1.1, which introduced the .run_simulation interface.
        # If it's not present, assume CADET-Python <= 1.0.4 and use the old .run_load() interface
        # This check can be removed at some point in the future.
        if hasattr(sim, "run_simulation"):
            return_information = sim.run_simulation()
        else:
            return_information = sim.run_load()

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_version(self):
        simulator = Cadet(install_path)
        version_pattern = r"\d\.\d\.\d"
        self.assertRegex(
            simulator.version, version_pattern, "Version format should be X.X.X"
        )

    def tearDown(self):
        shutil.rmtree("./tmp", ignore_errors=True)


unit_types = [
    "Cstr",
    "GeneralRateModel",
    "TubularReactor",
    "LumpedRateModelWithoutPores",
    "LumpedRateModelWithPores",
    "MCT",
]

parameter_combinations = [
    {},  # Default parameters
    {"n_par": 1},
    {"n_col": 1},
]

# Parameters to skip for specific unit types
exclude_rules = {
    "Cstr": [{"n_col": 1}, {"n_par": 1}],
    "TubularReactor": [{"n_par": 1}],
    "LumpedRateModelWithoutPores": [{"n_par": 1}],
    "LumpedRateModelWithPores": [{"n_par": 1}],
    "MCT": [{"n_par": 1}],
}


# Helper function to format parameters
def format_params(params):
    if not params:
        return "default"
    return "-".join(f"{k}={v}" for k, v in params.items())


process_test_cases = [
    pytest.param(
        (unit_type, params),
        id=f"{unit_type}-{format_params(params)}",
    )
    for unit_type in unit_types
    for params in parameter_combinations
    if not (unit_type in exclude_rules and params in exclude_rules[unit_type])
]

use_dll_options = [True, False]

simulation_test_cases = [
    pytest.param(
        (unit_type, params, use_dll),
        id=f"{unit_type}-{format_params(params)}-dll={use_dll}",
    )
    for unit_type in unit_types
    for params in parameter_combinations
    for use_dll in use_dll_options
    if not (unit_type in exclude_rules and params in exclude_rules[unit_type])
]


def run_simulation(
    process: Process, install_path: Optional[str] = None, use_dll: bool = False
) -> SimulationResults:
    """
    Run the CADET simulation for the given process and handle potential issues.

    Parameters
    ----------
    process : Process
        The process to simulate.
    install_path : str, optional
        The path to the CADET installation.

    Returns
    -------
    SimulationResults
        The results of the simulation.

    Raises
    ------
    CADETProcessError
        If the simulation fails with an error.
    """
    try:
        process_simulator = Cadet(install_path)
        process_simulator.use_dll = use_dll
        simulation_results = process_simulator.simulate(process)

        if not simulation_results.exit_flag == 0:
            raise CADETProcessError(
                f"LWE simulation failed with {simulation_results.exit_message}."
            )

        return simulation_results

    except Exception as e:
        raise CADETProcessError(f"CADET simulation failed: {e}.") from e


@pytest.fixture()
def process(request: pytest.FixtureRequest):
    """
    Fixture to set up the process for each unit type without running the simulation.
    """
    unit_type, kwargs = request.param
    process = create_lwe(unit_type, **kwargs)
    return process


@pytest.fixture
def simulation_results(request: pytest.FixtureRequest):
    """
    Fixture to set up the simulation for each unit type with different `use_dll` options.
    """
    unit_type, kwargs, use_dll = request.param  # Extract `use_dll`
    process = create_lwe(unit_type, **kwargs)  # Process remains unchanged
    simulation_results = run_simulation(process, install_path, use_dll=use_dll)
    return simulation_results


@pytest.mark.parametrize("process", process_test_cases, indirect=True)
@pytest.mark.slow
class TestProcessWithLWE:
    def return_process_config(self, process: Process) -> dict:
        """
        Returns the process configuration.

        Parameters
        ----------
        process : Process
            The process object.

        Returns
        -------
        dict
            The configuration of the process.
        """
        process_simulator = Cadet(install_path)
        process_config = process_simulator.get_process_config(process).input
        return process_config

    def test_model_config(self, process: Process):
        """
        Test the model configuration for various unit types in the process.

        Parameters
        ----------
        process : Process
            The process object.
        """
        process_config = self.return_process_config(process)

        n_comp = process.component_system.n_comp
        unit = process.flow_sheet.units[1]

        model_config = process_config.model
        input_config = model_config.unit_000
        output_config = model_config.unit_002
        unit_config = model_config.unit_001

        # ASSERT INPUT CONFIGURATION
        c1_lwe = [[50.0], [0.0], [[100.0, 0.2]]]
        cx_lwe = [[1.0], [0.0], [0.0]]

        expected_input_config = {
            "UNIT_TYPE": "INLET",
            "NCOMP": n_comp,
            "INLET_TYPE": "PIECEWISE_CUBIC_POLY",
            "discretization": {"nbound": n_comp * [0]},
            "sec_000": {
                "const_coeff": np.array(c1_lwe[0] + cx_lwe[0] * (n_comp - 1)),
                "lin_coeff": np.array([0.0] * n_comp),
                "quad_coeff": np.array([0.0] * n_comp),
                "cube_coeff": np.array([0.0] * n_comp),
            },
            "sec_001": {
                "const_coeff": np.array(c1_lwe[1] + cx_lwe[1] * (n_comp - 1)),
                "lin_coeff": np.array([0.0] * n_comp),
                "quad_coeff": np.array([0.0] * n_comp),
                "cube_coeff": np.array([0.0] * n_comp),
            },
            "sec_002": {
                "const_coeff": np.array([c1_lwe[2][0][0]] + cx_lwe[2] * (n_comp - 1)),
                "lin_coeff": np.array([c1_lwe[2][0][1]] + cx_lwe[2] * (n_comp - 1)),
                "quad_coeff": np.array([0.0] * n_comp),
                "cube_coeff": np.array([0.0] * n_comp),
            },
        }

        npt.assert_equal(input_config, expected_input_config)

        # ASSERT OUTPUT CONFIGURATION
        expected_output_config = {
            "UNIT_TYPE": "OUTLET",
            "NCOMP": n_comp,
            "discretization": {"nbound": n_comp * [0]},
        }

        npt.assert_equal(output_config, expected_output_config)

        # ASSERT MODEL CONFIGURATION
        assert unit_config.NCOMP == n_comp
        assert model_config.nunits == 3

        if unit.name == "Cstr":
            self.check_cstr(unit, unit_config)
        elif unit.name == "GeneralRateModel":
            self.check_general_rate_model(unit, unit_config)
        elif unit.name == "TubularReactor":
            self.check_tubular_reactor(unit, unit_config)
        elif unit.name == "LumpedRateModelWithoutPores":
            self.check_lumped_rate_model_without_pores(unit, unit_config)
        elif unit.name == "LumpedRateModelWithPores":
            self.check_lumped_rate_model_with_pores(unit, unit_config)
        elif unit.name == "MCT":
            self.check_mct(unit, unit_config)

    def check_cstr(self, unit, unit_config):
        """
        Check the configuration for a CSTR unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        assert unit_config.UNIT_TYPE == "CSTR"
        assert unit_config.INIT_Q == n_comp * [0]
        assert unit_config.INIT_C == n_comp * [0]
        assert unit_config.INIT_LIQUID_VOLUME == 0.0008425
        assert unit_config.CONST_SOLID_VOLUME == 0.00015749999999999998
        assert unit_config.FLOWRATE_FILTER == 0.0
        assert unit_config.nbound == [1, 1, 1, 1]

        self.check_adsorption_config(unit, unit_config)

    def check_general_rate_model(self, unit, unit_config):
        """
        Check the configuration for a General Rate Model unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        assert unit_config.UNIT_TYPE == "GENERAL_RATE_MODEL"
        assert unit_config.INIT_Q == n_comp * [0]
        assert unit_config.INIT_C == n_comp * [0]
        assert unit_config.INIT_CP == n_comp * [0]
        assert unit_config.VELOCITY == unit.flow_direction
        assert unit_config.COL_DISPERSION == n_comp * [5.75e-08]
        assert unit_config.CROSS_SECTION_AREA == np.pi * 0.01**2
        assert unit_config.COL_LENGTH == 0.014
        assert unit_config.COL_POROSITY == 0.37
        assert unit_config.FILM_DIFFUSION == [6.9e-6] * n_comp

        self.check_particle_config(unit_config)
        self.check_adsorption_config(unit, unit_config)
        self.check_discretization(unit, unit_config)

    def check_tubular_reactor(self, unit, unit_config):
        """
        Check the configuration for a Tubular Reactor unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        assert unit_config.UNIT_TYPE == "LUMPED_RATE_MODEL_WITHOUT_PORES"
        assert unit_config.INIT_C == n_comp * [0]
        assert unit_config.VELOCITY == unit.flow_direction
        assert unit_config.COL_DISPERSION == n_comp * [5.75e-08]
        assert unit_config.CROSS_SECTION_AREA == np.pi * 0.01**2
        assert unit_config.COL_LENGTH == 0.014
        assert unit_config.TOTAL_POROSITY == 1

        self.check_discretization(unit, unit_config)

    def check_lumped_rate_model_without_pores(self, unit, unit_config):
        """
        Check the configuration for a Lumped Rate Model Without Pores unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        assert unit_config.UNIT_TYPE == "LUMPED_RATE_MODEL_WITHOUT_PORES"
        assert unit_config.INIT_C == n_comp * [0]
        assert unit_config.VELOCITY == unit.flow_direction
        assert unit_config.COL_DISPERSION == n_comp * [5.75e-08]
        assert unit_config.CROSS_SECTION_AREA == np.pi * 0.01**2
        assert unit_config.COL_LENGTH == 0.014
        assert unit_config.TOTAL_POROSITY == 0.8425

        self.check_adsorption_config(unit, unit_config)
        self.check_discretization(unit, unit_config)

    def check_lumped_rate_model_with_pores(self, unit, unit_config):
        """
        Check the configuration for a Lumped Rate Model With Pores unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        assert unit_config.UNIT_TYPE == "LUMPED_RATE_MODEL_WITH_PORES"
        assert unit_config.INIT_C == n_comp * [0]
        assert unit_config.INIT_Q == n_comp * [0]
        assert unit_config.INIT_CP == n_comp * [0]
        assert unit_config.VELOCITY == unit.flow_direction
        assert unit_config.COL_DISPERSION == n_comp * [5.75e-08]
        assert unit_config.CROSS_SECTION_AREA == np.pi * 0.01**2
        assert unit_config.COL_LENGTH == 0.014
        assert unit_config.COL_POROSITY == 0.37
        assert unit_config.FILM_DIFFUSION == [6.9e-6] * n_comp

        self.check_adsorption_config(unit, unit_config)
        self.check_discretization(unit, unit_config)
        self.check_particle_config(unit_config)

    def check_mct(self, unit, unit_config):
        """
        Check the configuration for a Multi-Channel Transport unit.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp
        n_channel = unit.nchannel

        assert unit_config.UNIT_TYPE == "MULTI_CHANNEL_TRANSPORT"
        npt.assert_equal(unit_config.INIT_C, n_comp * [(n_channel * [0])])
        assert unit_config.VELOCITY == unit.flow_direction
        npt.assert_equal(
            unit_config.EXCHANGE_MATRIX,
            np.array(
                [
                    [n_comp * [0.0], n_comp * [0.001], n_comp * [0.0]],
                    [n_comp * [0.002], n_comp * [0.0], n_comp * [0.003]],
                    [n_comp * [0.0], n_comp * [0.0], n_comp * [0.0]],
                ]
            ),
        )
        assert unit_config.COL_DISPERSION == n_comp * n_channel * [5.75e-08]
        assert unit_config.NCHANNEL == 3
        assert unit_config.CHANNEL_CROSS_SECTION_AREAS == 3 * [2 * np.pi * (0.01**2)]

        self.check_discretization(unit, unit_config)

    def check_adsorption_config(self, unit, unit_config):
        """
        Check the adsorption configuration.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        n_comp = unit.component_system.n_comp

        expected_adsorption_config = {
            "ADSORPTION_MODEL": "STERIC_MASS_ACTION",
            "IS_KINETIC": True,
            "SMA_KA": n_comp * [35.5],
            "SMA_KD": n_comp * [1000.0],
            "SMA_LAMBDA": 1200.0,
            "SMA_NU": n_comp * [4.7],
            "SMA_SIGMA": n_comp * [11.83],
            "SMA_REFC0": 1.0,
            "SMA_REFQ": 1.0,
        }

        assert unit_config.adsorption_model == "STERIC_MASS_ACTION"
        npt.assert_equal(unit_config.adsorption, expected_adsorption_config)

    def check_particle_config(self, unit_config):
        """
        Check the particle configuration.

        Parameters
        ----------
        unit_config : dict
            The configuration of the unit.
        """
        assert unit_config.PAR_POROSITY == 0.75
        assert unit_config.PAR_RADIUS == 4.5e-05
        assert unit_config.discretization.par_geom == "SPHERE"

    def check_discretization(self, unit, unit_config):
        """
        Check the discretization configuration.

        Parameters
        ----------
        unit : Unit
            The unit object.
        unit_config : dict
            The configuration of the unit.
        """
        assert unit_config.discretization.ncol == unit.discretization.ncol
        assert (
            unit_config.discretization.use_analytic_jacobian
            == unit.discretization.use_analytic_jacobian
        )
        assert unit_config.discretization.reconstruction == "WENO"

        npt.assert_equal(
            unit_config.discretization.weno,
            {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
        )

        npt.assert_equal(
            unit_config.discretization.consistency_solver,
            {
                "solver_name": "LEVMAR",
                "init_damping": 0.01,
                "min_damping": 0.0001,
                "max_iterations": 50,
                "subsolvers": "LEVMAR",
            },
        )

        if "spatial_method" in unit.discretization.parameters:
            assert (
                unit_config.discretization.spatial_method
                == unit.discretization.spatial_method
            )

        if "npar" in unit.discretization.parameters:
            assert unit_config.discretization.npar == unit.discretization.npar
            assert unit_config.discretization.par_disc_type == "EQUIDISTANT_PAR"

    def test_solver_config(self, process: Process):
        """
        Test the solver configuration for the process.

        Parameters
        ----------
        process : Process
            The process object.
        """
        process_config = self.return_process_config(process)
        solver_config = process_config.solver

        expected_solver_config = {
            "nthreads": 1,
            "consistent_init_mode": 1,
            "consistent_init_mode_sens": 1,
            "user_solution_times": np.arange(0.0, 120 * 60 + 1),
            "sections": {
                "nsec": 3,
                "section_times": [0.0, 10.0, 90.0, 7200.0],
                "section_continuity": [0, 0],
            },
            "time_integrator": {
                "abstol": 1e-08,
                "algtol": 1e-12,
                "reltol": 1e-06,
                "reltol_sens": 1e-12,
                "init_step_size": 1e-06,
                "max_steps": 1000000,
                "max_step_size": 0.0,
                "errortest_sens": False,
                "max_newton_iter": 1000000,
                "max_errtest_fail": 1000000,
                "max_convtest_fail": 1000000,
                "max_newton_iter_sens": 1000000,
            },
        }

        npt.assert_equal(solver_config, expected_solver_config)

    def test_return_config(self, process: Process):
        """
        Test the return configuration for the process.

        Parameters
        ----------
        process : Process
            The process object.
        """
        process_config = self.return_process_config(process)
        return_config = process_config["return"]

        # Assert that all values in return_config.unit_001 (model unit operation) are True
        for key, value in return_config["unit_001"].items():
            assert value, f"The value for key '{key}' is not True. Found: {value}"

    def test_sensitivity_config(self, process: Process):
        """
        Test the sensitivity configuration for the process.

        Parameters
        ----------
        process : Process
            The process object.
        """
        process_config = self.return_process_config(process)
        sensitivity_config = process_config.sensitivity

        expected_sensitivity_config = {"sens_method": "ad1", "nsens": 0}
        npt.assert_equal(sensitivity_config, expected_sensitivity_config)


@pytest.mark.parametrize("simulation_results", simulation_test_cases, indirect=True)
@pytest.mark.slow
class TestResultsWithLWE:
    def test_trigger_simulation(self, simulation_results):
        """
        Test to trigger the simulation.
        """
        simulation_results = simulation_results
        assert simulation_results is not None

    def test_compare_solution_shape(self, simulation_results):
        """
        Compare the dimensions of the solution object against the expected solution shape.
        """
        simulation_results = simulation_results
        process = simulation_results.process
        unit = process.flow_sheet.units[1]

        # for units without ports
        if not unit.has_ports:
            # assert solution inlet has shape (t, n_comp)
            assert simulation_results.solution[unit.name].inlet.solution_shape == (
                int(process.cycle_time + 1),
                process.component_system.n_comp,
            )
            # assert solution outlet has shape (t, n_comp)
            assert simulation_results.solution[unit.name].outlet.solution_shape == (
                int(process.cycle_time + 1),
                process.component_system.n_comp,
            )
            # assert solution bulk has shape (t, n_col, n_comp)
            if not isinstance(unit.discretization, NoDiscretization):
                assert simulation_results.solution[unit.name].bulk.solution_shape == (
                    int(process.cycle_time + 1),
                    unit.discretization.ncol,
                    process.component_system.n_comp,
                )

        # for units with ports
        else:
            # assert solution inlet is given for each port
            assert len(simulation_results.solution[unit.name].inlet) == unit.n_ports
            # assert solution for channel 0 has shape (t, n_comp)
            assert simulation_results.solution[
                unit.name
            ].inlet.channel_0.solution_shape == (
                int(process.cycle_time + 1),
                process.component_system.n_comp,
            )
            # assert solution bulk has shape (t, n_col, n_ports, n_comp)
            assert simulation_results.solution[unit.name].bulk.solution_shape == (
                int(process.cycle_time + 1),
                unit.discretization.ncol,
                unit.n_ports,
                process.component_system.n_comp,
            )

        # for units with particles
        if unit.supports_binding and not isinstance(
            unit.discretization, NoDiscretization
        ):
            # for units with solid phase and particle discretization
            if "npar" in unit.discretization.parameters:
                # assert solution solid has shape (t, n_col, n_par, n_comp)
                assert simulation_results.solution[unit.name].solid.solution_shape == (
                    int(process.cycle_time + 1),
                    unit.discretization.ncol,
                    unit.discretization.npar,
                    process.component_system.n_comp,
                )
            # for units with solid phase and without particle discretization
            else:
                # assert solution solid has shape (t, ncol, n_comp)
                assert simulation_results.solution[unit.name].solid.solution_shape == (
                    int(process.cycle_time + 1),
                    unit.discretization.ncol,
                    process.component_system.n_comp,
                )

        # for units with particle mobile phase and particle discretization
        if (
            unit.supports_particle_reaction
            and unit.name != "LumpedRateModelWithoutPores"
        ):
            # assert soluction particle has shape (t, n_col, n_par, n_comp)
            if "npar" in unit.discretization.parameters:
                assert simulation_results.solution[
                    unit.name
                ].particle.solution_shape == (
                    int(process.cycle_time + 1),
                    unit.discretization.ncol,
                    unit.discretization.npar,
                    process.component_system.n_comp,
                )
            # for units with particle mobiles phase and without particle discretization
            else:
                # assert solution particle has shape (t, n_col, n_comp)
                assert simulation_results.solution[
                    unit.name
                ].particle.solution_shape == (
                    int(process.cycle_time + 1),
                    unit.discretization.ncol,
                    process.component_system.n_comp,
                )


if __name__ == "__main__":
    pytest.main([__file__])
