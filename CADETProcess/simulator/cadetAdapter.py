from collections import defaultdict
from functools import wraps
import os
import platform
from pathlib import Path
import shutil
import subprocess
from subprocess import TimeoutExpired
import time
import tempfile
import warnings

from addict import Dict
import numpy as np
from cadet import Cadet as CadetAPI

from CADETProcess import CADETProcessError
from CADETProcess import settings
from CADETProcess.dataStructure import (
    Bool, Switch, UnsignedFloat, UnsignedInteger,
)

from .simulator import SimulatorBase
from CADETProcess import SimulationResults
from CADETProcess.solution import (
    SolutionIO, SolutionBulk, SolutionParticle, SolutionSolid, SolutionVolume
)
from CADETProcess.processModel import NoBinding, BindingBaseClass
from CADETProcess.processModel import NoReaction, ReactionBaseClass
from CADETProcess.processModel import NoDiscretization, DGMixin
from CADETProcess.processModel import (
    UnitBaseClass, Inlet, Cstr, TubularReactor, LumpedRateModelWithoutPores
)
from CADETProcess.processModel import Process


__all__ = [
    'Cadet',
    'ModelSolverParameters',
    'UnitParameters',
    'AdsorptionParameters',
    'ReactionParameters',
    'SolverParameters',
    'SolverTimeIntegratorParameters',
    'ReturnParameters',
    'SensitivityParameters',
]


class Cadet(SimulatorBase):
    """CADET class for running a simulation for given process objects.

    Attributes
    ----------
    install_path: str
        Path to the root of the CADET installation
    time_out : UnsignedFloat
        Maximum duration for simulations
    model_solver_parameters : ModelSolverParameters
        Container for solver parameters
    unit_discretization_parameters : UnitDiscretizationParameters
        Container for unit discretization parameters
    discretization_weno_parameters : DiscretizationWenoParameters
        Container for weno discretization parameters in units
    adsorption_consistency_solver_parameters : ConsistencySolverParameters
        Container for consistency solver parameters
    solver_parameters : SolverParameters
        Container for general solver settings
    time_integrator_parameters : SolverTimeIntegratorParameters
        Container for time integrator parameters
    return_parameters : ReturnParameters
        Container for return information of the system

    ..todo::
        Implement method for loading CADET file that have not been generated
        with CADETProcess and create Process

    See Also
    --------
    ReturnParameters
    ModelSolverParameters
    SolverParameters
    SolverTimeIntegratorParameters
    CadetAPI

    """

    timeout = UnsignedFloat()
    use_dll = Bool(default=False)
    _force_constant_flow_rate = False

    def __init__(self, install_path=None, temp_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cadet_root = None
        self.cadet_cli_path = None
        self.cadet_create_lwe_path = None
        self.cadet_dll_path = None

        if install_path is None:
            self.autodetect_cadet()
        else:
            self.install_path = install_path

        self.model_solver_parameters = ModelSolverParameters()
        self.solver_parameters = SolverParameters()
        self.time_integrator_parameters = SolverTimeIntegratorParameters()

        self.return_parameters = ReturnParameters()
        self.sensitivity_parameters = SensitivityParameters()

        if temp_dir is None:
            temp_dir = settings.temp_dir / 'simulation_files'
        self.temp_dir = temp_dir

    @property
    def temp_dir(self):
        if not self._temp_dir.exists():
            self._temp_dir.mkdir(exist_ok=True, parents=True)
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, temp_dir):
        self._temp_dir = temp_dir

    def locks_process(func):
        """Lock process to enable caching."""
        @wraps(func)
        def wrapper(self, process, *args, **kwargs):
            locked_process = False

            if not process.lock:
                process.lock = True
                locked_process = True

            results = func(self, process, *args, **kwargs)

            if locked_process:
                process.lock = False

            return results

        return wrapper

    def autodetect_cadet(self):
        """
        Autodetect installation CADET based on operating system and API usage.

        Returns
        -------
        cadet_root : Path
            Installation path of the CADET program.
        """
        executable = 'cadet-cli'
        if platform.system() == 'Windows':
            executable += '.exe'

        # Searching for the executable in system path
        path = shutil.which(executable)

        if path is None:
            raise FileNotFoundError(
                "Could not autodetect CADET installation. Please provide path."
            )
        else:
            self.logger.info(f"Found CADET executable at {path}")

        cli_path = Path(path)

        cadet_root = None
        if cli_path is not None:
            cadet_root = cli_path.parent.parent
            self.install_path = cadet_root

        return cadet_root

    @property
    def cadet_path(self):
        if self.use_dll and self.found_dll:
            return self.cadet_cll_path
        return self.cadet_cli_path

    @property
    def found_dll(self):
        flag = False
        if self.cadet_dll_path is not None:
            flag = True
        return flag

    @property
    def install_path(self):
        """str: Path to the installation of CADET.

        This can either be the root directory of the installation or the path to the
        executable file 'cadet-cli'. If a file path is provided, the root directory will
        be inferred.

        Raises
        ------
        FileNotFoundError
            If CADET cannot be found at the specified path.

        Warnings
        --------
        If the specified install_path is not the root of the CADET installation, it will
        be inferred from the file path.

        See Also
        --------
        check_cadet
        """
        return self._install_path

    @install_path.setter
    def install_path(self, install_path):
        """
        Set the installation path of CADET.

        Parameters
        ----------
        install_path : str or Path
            Path to the root of the CADET installation.
            It should either be the root directory of the installation or the path
            to the executable file 'cadet-cli'.
            If a file path is provided, the root directory will be inferred.
        """
        if install_path is None:
            self._install_path = None
            self.cadet_cli_path = None
            self.cadet_dll_path = None
            self.cadet_create_lwe_path = None

            return

        install_path = Path(install_path)

        if install_path.is_file():
            cadet_root = install_path.parent.parent
            warnings.warn(
                "The specified install_path is not the root of the CADET installation. "
                "It has been inferred from the file path."
            )
        else:
            cadet_root = install_path

        self._install_path = cadet_root

        cli_executable = 'cadet-cli'
        lwe_executable = 'createLWE'

        if platform.system() == 'Windows':
            cli_executable += '.exe'
            lwe_executable += '.exe'

        cadet_cli_path = cadet_root / 'bin' / cli_executable
        if cadet_cli_path.is_file():
            self.cadet_cli_path = cadet_cli_path
        else:
            raise FileNotFoundError(
                "CADET could not be found. Please check the path"
            )

        cadet_create_lwe_path = cadet_root / 'bin' / lwe_executable
        if cadet_create_lwe_path.is_file():
            self.cadet_create_lwe_path = cadet_create_lwe_path.as_posix()

        if platform.system() == 'Windows':
            dll_path = cadet_root / 'bin' / 'cadet.dll'
            dll_debug_path = cadet_root / 'bin' / 'cadet_d.dll'
        else:
            dll_path = cadet_root / 'lib' / 'lib_cadet.so'
            dll_debug_path = cadet_root / 'lib' / 'lib_cadet_d.so'

        # Look for debug dll if dll is not found.
        if not dll_path.is_file() and dll_debug_path.is_file():
            dll_path = dll_debug_path

        # Look for debug dll if dll is not found.
        if dll_path.is_file():
            self.cadet_dll_path = dll_path.as_posix()

        if platform.system() != 'Windows':
            try:
                cadet_lib_path = cadet_root / 'lib'
                if cadet_lib_path.as_posix() not in os.environ['LD_LIBRARY_PATH']:
                    os.environ['LD_LIBRARY_PATH'] += \
                        os.pathsep + cadet_lib_path.as_posix()
            except KeyError:
                os.environ['LD_LIBRARY_PATH'] = cadet_lib_path.as_posix()

    def check_cadet(self):
        """
        Check if CADET installation can run a basic LWE example.

        Returns
        -------
        bool
            True if the test simulation completed successfully, False otherwise.

        Raises
        ------
        CADETProcessError
            If the simulation fails, an exception is raised with the error message.

        Notes
        -----
        This method tests the CADET installation by executing a basic LWE (Load, Wash,
        Elute) example. It creates a CADET model using the LWE data stored in an HDF5
        file and runs the simulation. After the simulation, it checks the return code to
        determine if the test was successful.
        """
        lwe_hdf5_path = Path(self.temp_dir) / 'LWE.h5'

        cadet_model = self.create_lwe(lwe_hdf5_path)

        data = cadet_model.run()
        os.remove(lwe_hdf5_path)

        if data.returncode == 0:
            flag = True
            print("Test simulation completed successfully")
        else:
            flag = False
            raise CADETProcessError(f"Simulation failed with {data}")

        return flag

    def get_tempfile_name(self):
        f = next(tempfile._get_candidate_names())
        return self.temp_dir / f'{f}.h5'

    def create_lwe(self, file_path=None):
        """Create basic LWE example.

        Parameters
        ----------
        file_path : Path, optional
            Path to store HDF5 file. If None, temporary file will be created and
            deleted after simulation.

        Returns
        -------

        """
        if file_path is None:
            file_name = self.get_tempfile_name().as_posix()
            cwd = self.temp_dir.as_posix()
        else:
            file_path = Path(file_path).absolute()
            file_name = file_path.name
            cwd = file_path.parent.as_posix()

        ret = subprocess.run(
            [self.cadet_create_lwe_path, '-o', file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd
        )
        if ret.returncode != 0:
            if ret.stdout:
                print('Output', ret.stdout.decode('utf-8'))
            if ret.stderr:
                print('Errors', ret.stderr.decode('utf-8'))
            raise CADETProcessError(
                "Failure: Creation of test simulation ran into problems"
            )

        cadet_model = self.get_new_cadet_instance()

        cadet_model.filename = file_path.as_posix()

        cadet_model.load()

        if file_path is None:
            os.remove(file_path)

        return cadet_model

    @locks_process
    def run(self, process, cadet=None, file_path=None):
        """Interface to the solver run function.

        The configuration is extracted from the process object and then saved
        as a temporary .h5 file. After termination, the same file is processed
        and the results are returned.

        Cadet Return information:
        - 0: pass (everything allright)
        - 1: Standard Error
        - 2: IO Error
        - 3: Solver Error

        Parameters
        ----------
        process : Process
            process to be simulated

        Returns
        -------
        results : SimulationResults
            Simulation results including process and solver configuration.

        Raises
        ------
        TypeError
            If process is not instance of Process

        See Also
        --------
        get_process_config
        get_simulation_results

        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if cadet is None:
            cadet = self.get_new_cadet_instance()

        cadet.root = self.get_process_config(process)

        if cadet.is_file:
            if file_path is None:
                cadet.filename = self.get_tempfile_name()
            else:
                cadet.filename = file_path
            cadet.save()

        try:
            start = time.time()
            return_information = cadet.run_load(timeout=self.timeout)
            elapsed = time.time() - start
        except TimeoutExpired:
            raise CADETProcessError('Simulator timed out') from None
        finally:
            if file_path is None:
                os.remove(cadet.filename)

        if return_information.returncode != 0:
            self.logger.error(
                f'Simulation of {process.name} '
                f'with parameters {process.config} failed.'
            )
            raise CADETProcessError(
                f'CADET Error: Simulation failed with {return_information.stderr}'
            ) from None

        try:
            results = self.get_simulation_results(
                process, cadet, elapsed, return_information
            )
        except TypeError:
            raise CADETProcessError(
                'Unexpected error reading SimulationResults.'
            )

        return results

    def get_new_cadet_instance(self):
        cadet = CadetAPI()
        # Because the initialization in __init__ isn't guaranteed to be called in multiprocessing
        #  situations, ensure that the cadet_path has actually been set.
        if not hasattr(cadet, "cadet_path"):
            cadet.cadet_path = self.cadet_path
        return cadet

    def save_to_h5(self, process, file_path):
        cadet = self.get_new_cadet_instance()
        cadet.root = self.get_process_config(process)
        cadet.filename = file_path
        cadet.save()

    def run_h5(self, file_path):
        cadet = self.get_new_cadet_instance()
        cadet.filename = file_path
        cadet.load()
        cadet.run_load(timeout=self.timeout)

        return cadet

    def load_from_h5(self, file_path):
        cadet = self.get_new_cadet_instance()
        cadet.filename = file_path
        cadet.load()

        return cadet

    @locks_process
    def get_process_config(self, process):
        """Create the CADET config.

        Returns
        -------
        config : Dict
            /

        Notes
        -----
            Sensitivities not implemented yet.

        See Also
        --------
        get_input_model
        get_input_solver
        get_input_return
        get_input_sensitivity

        """
        config = Dict()
        config.input.model = self.get_input_model(process)
        config.input.solver = self.get_input_solver(process)
        config.input['return'] = self.get_input_return(process)
        config.input.sensitivity = self.get_input_sensitivity(process)

        return config

    def load_simulation_results(self, process, file_path):
        cadet = self.load_from_h5(file_path)
        results = self.get_simulation_results(process, cadet)

        return results

    @locks_process
    def get_simulation_results(
            self,
            process,
            cadet,
            time_elapsed=None,
            return_information=None):
        """Read simulation results from CADET configuration.

        Parameters
        ----------
        process :  Process
            Process that was simulated.
        cadet : CadetAPI
            Cadet object with simulation results.
        time_elapsed : float
            Time of simulation.
        return_information: str
            CADET-cli return information.

        Returns
        -------
        results : SimulationResults
            Simulation results including process and solver configuration.

        ..todo::
            Implement method to read .h5 files that have no process associated.

        """
        if time_elapsed is None:
            time_elapsed = cadet.root.meta.time_sim
        time = self.get_solution_time(process)
        if return_information is None:
            exit_flag = None
            exit_message = None
        else:
            exit_flag = return_information.returncode
            exit_message = return_information.stderr.decode()

        try:
            solution = Dict()
            for unit in process.flow_sheet.units:
                solution[unit.name] = defaultdict(list)
                unit_index = self.get_unit_index(process, unit)
                unit_solution = cadet.root.output.solution[unit_index]
                unit_coordinates = \
                    cadet.root.output.coordinates[unit_index].copy()
                particle_coordinates = \
                    unit_coordinates.pop('particle_coordinates_000', None)

                flow_in = process.flow_rate_timelines[unit.name].total_in
                flow_out = process.flow_rate_timelines[unit.name].total_out

                for cycle in range(self.n_cycles):
                    start = cycle * len(time)
                    end = (cycle + 1) * len(time)

                    if 'solution_inlet' in unit_solution.keys():
                        sol_inlet = unit_solution.solution_inlet[start:end, :]
                        solution[unit.name]['inlet'].append(
                            SolutionIO(
                                unit.name,
                                unit.component_system, time, sol_inlet,
                                flow_in
                            )
                        )

                    if 'solution_outlet' in unit_solution.keys():
                        sol_outlet = unit_solution.solution_outlet[start:end, :]
                        solution[unit.name]['outlet'].append(
                            SolutionIO(
                                unit.name,
                                unit.component_system, time, sol_outlet,
                                flow_out
                            )
                        )

                    if 'solution_bulk' in unit_solution.keys():
                        sol_bulk = unit_solution.solution_bulk[start:end, :]
                        solution[unit.name]['bulk'].append(
                            SolutionBulk(
                                unit.name,
                                unit.component_system, time, sol_bulk,
                                **unit_coordinates
                            )
                        )

                    if 'solution_particle' in unit_solution.keys():
                        sol_particle = unit_solution.solution_particle[start:end, :]
                        solution[unit.name]['particle'].append(
                            SolutionParticle(
                                unit.name,
                                unit.component_system, time, sol_particle,
                                **unit_coordinates,
                                particle_coordinates=particle_coordinates
                            )
                        )

                    if 'solution_solid' in unit_solution.keys():
                        sol_solid = unit_solution.solution_solid[start:end, :]
                        solution[unit.name]['solid'].append(
                            SolutionSolid(
                                unit.name,
                                unit.component_system,
                                unit.binding_model.bound_states,
                                time, sol_solid,
                                **unit_coordinates,
                                particle_coordinates=particle_coordinates
                            )
                        )

                    if 'solution_volume' in unit_solution.keys():
                        sol_volume = unit_solution.solution_volume[start:end, :]
                        solution[unit.name]['volume'].append(
                            SolutionVolume(
                                unit.name,
                                unit.component_system,
                                time,
                                sol_volume
                            )
                        )

            solution = Dict(solution)

            sensitivity = Dict()
            for i, sens in enumerate(process.parameter_sensitivities):
                sens_index = f'param_{i:03d}'
                for unit in process.flow_sheet.units:
                    sensitivity[sens.name][unit.name] = defaultdict(list)
                    unit_index = self.get_unit_index(process, unit)
                    unit_sensitivity = cadet.root.output.sensitivity[sens_index][unit_index]
                    unit_coordinates = \
                        cadet.root.output.coordinates[unit_index].copy()
                    particle_coordinates = \
                        unit_coordinates.pop('particle_coordinates_000', None)

                    flow_in = process.flow_rate_timelines[unit.name].total_in
                    flow_out = process.flow_rate_timelines[unit.name].total_out

                    for cycle in range(self.n_cycles):
                        start = cycle * len(time)
                        end = (cycle + 1) * len(time)

                        if 'sens_inlet' in unit_sensitivity.keys():
                            sens_inlet = unit_sensitivity.sens_inlet[start:end, :]
                            sensitivity[sens.name][unit.name]['inlet'].append(
                                SolutionIO(
                                    unit.name,
                                    unit.component_system, time, sens_inlet,
                                    flow_in
                                )
                            )

                        if 'sens_outlet' in unit_sensitivity.keys():
                            sens_outlet = unit_sensitivity.sens_outlet[start:end, :]
                            sensitivity[sens.name][unit.name]['outlet'].append(
                                SolutionIO(
                                    unit.name,
                                    unit.component_system, time, sens_outlet,
                                    flow_out
                                )
                            )

                        if 'sens_bulk' in unit_sensitivity.keys():
                            sens_bulk = unit_sensitivity.sens_bulk[start:end, :]
                            sensitivity[sens.name][unit.name]['bulk'].append(
                                SolutionBulk(
                                    unit.name,
                                    unit.component_system, time, sens_bulk,
                                    **unit_coordinates
                                )
                            )

                        if 'sens_particle' in unit_sensitivity.keys():
                            sens_particle = unit_sensitivity.sens_particle[start:end, :]
                            sensitivity[sens.name][unit.name]['particle'].append(
                                SolutionParticle(
                                    unit.name,
                                    unit.component_system, time, sens_particle,
                                    **unit_coordinates,
                                    particle_coordinates=particle_coordinates
                                )
                            )

                        if 'sens_solid' in unit_sensitivity.keys():
                            sens_solid = unit_sensitivity.sens_solid[start:end, :]
                            sensitivity[sens.name][unit.name]['solid'].append(
                                SolutionSolid(
                                    unit.name,
                                    unit.component_system,
                                    unit.binding_model.bound_states,
                                    time, sens_solid,
                                    **unit_coordinates,
                                    particle_coordinates=particle_coordinates
                                )
                            )

                        if 'sens_volume' in unit_sensitivity.keys():
                            sens_volume = unit_sensitivity.sens_volume[start:end, :]
                            sensitivity[sens.name][unit.name]['volume'].append(
                                SolutionVolume(
                                    unit.name,
                                    unit.component_system,
                                    time,
                                    sens_volume
                                )
                            )

            sensitivity = Dict(sensitivity)

            system_state = {
                'state': cadet.root.output.last_state_y,
                'state_derivative': cadet.root.output.last_state_ydot
            }

            chromatograms = [
                solution[chrom.name].outlet[-1]
                for chrom in process.flow_sheet.product_outlets
            ]

        except KeyError:
            raise CADETProcessError('Results don\'t match Process')

        results = SimulationResults(
            solver_name=str(self),
            solver_parameters=dict(),
            exit_flag=exit_flag,
            exit_message=exit_message,
            time_elapsed=time_elapsed,
            process=process,
            solution_cycles=solution,
            sensitivity_cycles=sensitivity,
            system_state=system_state,
            chromatograms=chromatograms
        )

        return results

    def get_input_model(self, process):
        """Config branch /input/model/

        Notes
        -----
        !!! External functions not implemented yet

        See Also
        --------
        model_connections
        model_solver
        model_units
        input_model_parameters

        """
        input_model = Dict()

        input_model.connections = self.get_model_connections(process)
        # input_model.external = self.model_external # !!! not working yet
        input_model.solver = self.model_solver_parameters.parameters
        input_model.update(self.get_model_units(process))

        if process.system_state is not None:
            input_model['INIT_STATE_Y'] = process.system_state
        if process.system_state_derivative is not None:
            input_model['INIT_STATE_YDOT'] = process.system_state_derivative

        return input_model

    def get_model_connections(self, process):
        """Config branch /input/model/connections"""
        model_connections = Dict()
        if self._force_constant_flow_rate:
            model_connections['CONNECTIONS_INCLUDE_DYNAMIC_FLOW'] = 0
        else:
            model_connections['CONNECTIONS_INCLUDE_DYNAMIC_FLOW'] = 1

        index = 0

        section_states = process.flow_rate_section_states

        for cycle in range(0, self.n_cycles):
            for flow_rates_state in section_states.values():

                switch_index = f'switch_{index:03d}'
                model_connections[switch_index].section = index

                connections = self.cadet_connections(
                    flow_rates_state, process.flow_sheet
                )
                model_connections[switch_index].connections = connections
                index += 1

        model_connections.nswitches = index

        return model_connections

    def cadet_connections(self, flow_rates, flow_sheet):
        """list: Connections matrix for flow_rates state.

        Parameters
        ----------
        flow_rates : dict
            UnitOperations with outgoing flow rates.
        flow_sheet : FlowSheet
            Object which hosts units (for getting unit index).

        Returns
        -------
        ls : list
            Connections matrix for DESCRIPTION.

        """
        table = Dict()
        enum = 0

        for origin, unit_flow_rates in flow_rates.items():
            origin = flow_sheet[origin]
            origin_index = flow_sheet.get_unit_index(origin)
            for dest, flow_rate in unit_flow_rates.destinations.items():
                destination = flow_sheet[dest]
                destination_index = flow_sheet.get_unit_index(destination)
                if np.any(flow_rate):
                    table[enum] = []
                    table[enum].append(int(origin_index))
                    table[enum].append(int(destination_index))
                    table[enum].append(-1)
                    table[enum].append(-1)
                    Q = flow_rate.tolist()
                    if self._force_constant_flow_rate:
                        table[enum] += [Q[0]]
                    else:
                        table[enum] += Q
                    enum += 1

        ls = []
        for connection in table.values():
            ls += connection

        return ls

    def get_unit_index(self, process, unit):
        """Helper function for getting unit index in CADET format unit_xxx.

        Parameters
        ----------
        process : Process
            process to be simulated
        unit : UnitOperation
            Indexed object

        Returns
        -------
        unit_index : str
            Return the unit index in CADET format unit_XXX

        """
        index = process.flow_sheet.get_unit_index(unit)
        return f'unit_{index:03d}'

    def get_model_units(self, process):
        """Config branches for all units /input/model/unit_000 ... unit_xxx.

        See Also
        --------
        get_unit_config
        get_unit_index

        """
        model_units = Dict()

        model_units.nunits = len(process.flow_sheet.units)

        for unit in process.flow_sheet.units:
            unit_index = self.get_unit_index(process, unit)
            model_units[unit_index] = self.get_unit_config(unit)

        self.set_section_dependent_parameters(model_units, process)

        return model_units

    def get_unit_config(self, unit):
        """Config branch /input/model/unit_xxx for individual unit.

        The unit operation parameters are converted to CADET format

        Notes
        -----
        In CADET, the parameter unit_config['discretization'].NBOUND should be
        moved to binding config or unit config

        See Also
        --------
        get_adsorption_config

        """
        unit_parameters = UnitParameters(unit)

        unit_config = Dict(unit_parameters.parameters)

        if not isinstance(unit.binding_model, NoBinding):
            if unit.binding_model.n_binding_sites > 1:
                n_bound = \
                    [unit.binding_model.n_binding_sites] * unit.binding_model.n_comp
            else:
                n_bound = unit.binding_model.bound_states

            unit_config['adsorption'] = \
                self.get_adsorption_config(unit.binding_model)
            unit_config['adsorption_model'] = \
                unit_config['adsorption']['ADSORPTION_MODEL']
        else:
            n_bound = unit.n_comp*[0]

        if not isinstance(unit.discretization, NoDiscretization):
            unit_config['discretization'] = unit.discretization.parameters
            if isinstance(unit.discretization, DGMixin):
                unit_config['UNIT_TYPE'] += '_DG'

        if isinstance(unit, Cstr) \
                and not isinstance(unit.binding_model, NoBinding):
            unit_config['nbound'] = n_bound
        else:
            unit_config['discretization']['nbound'] = n_bound

        # Bulk Reaction
        if not isinstance(unit.bulk_reaction_model, NoReaction):
            parameters = self.get_reaction_config(unit.bulk_reaction_model)

            unit_config['reaction_model'] = parameters['REACTION_MODEL']
            # Converting bulk reaction to particle reaction interface (used by LRM)
            if isinstance(unit, TubularReactor):
                for key, value in parameters.items():
                    key = key.replace('bulk', 'liquid')
                    unit_config['reaction'][key] = value
            else:
                unit_config['reaction_bulk'] = parameters

        # Particle Reaction
        if not isinstance(unit.particle_reaction_model, NoReaction):
            parameters = self.get_reaction_config(unit.particle_reaction_model)
            if isinstance(unit, LumpedRateModelWithoutPores):
                unit_config['reaction_model'] = parameters['REACTION_MODEL']
                unit_config['reaction'] = parameters
            else:
                unit_config['reaction_model_particles'] = parameters['REACTION_MODEL']
                unit_config['reaction_particle'].update(parameters)

        if isinstance(unit, Inlet):
            unit_config['sec_000']['const_coeff'] = unit.c[:, 0]
            unit_config['sec_000']['lin_coeff'] = unit.c[:, 1]
            unit_config['sec_000']['quad_coeff'] = unit.c[:, 2]
            unit_config['sec_000']['cube_coeff'] = unit.c[:, 3]

        return unit_config

    def set_section_dependent_parameters(self, model_units, process):
        """Add time dependent model parameters to units."""
        section_states = process.section_states.values()

        section_index = 0
        for cycle in range(0, self.n_cycles):
            for param_states in section_states:
                for param, state in param_states.items():
                    param = param.split('.')
                    unit_name = param[1]
                    param_name = param[-1]
                    try:
                        unit = process.flow_sheet[unit_name]
                    except KeyError:
                        if unit_name == 'output_states':
                            continue
                        else:
                            raise CADETProcessError(
                                'Unexpected section dependent parameter'
                            )
                    if param_name == 'flow_rate':
                        continue
                    unit_index = process.flow_sheet.get_unit_index(unit)
                    if isinstance(unit, Inlet) and param_name == 'c':
                        self.add_inlet_section(
                            model_units, section_index, unit_index, state
                        )
                    else:
                        unit_model = unit.model
                        self.add_parameter_section(
                            model_units, section_index, unit_index,
                            unit_model, param_name, state
                        )

                section_index += 1

    def add_inlet_section(self, model_units, sec_index, unit_index, coeffs):
        unit_index = f'unit_{unit_index:03d}'
        section_index = f'sec_{sec_index:03d}'

        model_units[unit_index][section_index]['const_coeff'] = coeffs[:, 0]
        model_units[unit_index][section_index]['lin_coeff'] = coeffs[:, 1]
        model_units[unit_index][section_index]['quad_coeff'] = coeffs[:, 2]
        model_units[unit_index][section_index]['cube_coeff'] = coeffs[:, 3]

    def add_parameter_section(
            self, model_units, sec_index, unit_index, unit_model,
            parameter, state):
        """Add section value to parameter branch."""
        unit_index = f'unit_{unit_index:03d}'
        parameter_name = \
            inv_unit_parameters_map[unit_model]['parameters'][parameter]

        if sec_index == 0:
            model_units[unit_index][parameter_name] = []
        model_units[unit_index][parameter_name] += list(state.ravel())

    def get_adsorption_config(self, binding):
        """Config branch /input/model/unit_xxx/adsorption for individual unit.

        Binding model parameters are extracted and converted to CADET format.

        Parameters
        ----------
        binding : BindingBaseClass
            Binding model

        See Also
        --------
        get_unit_config
        """
        adsorption_config = AdsorptionParameters(binding).parameters

        return adsorption_config

    def get_reaction_config(self, reaction):
        """Config branch /input/model/unit_xxx/reaction for individual unit.

        Reaction model parameters are extracted and converted to CADET format.

        Parameters
        ----------
        reaction : ReactionBaseClass
            Reaction model

        See Also
        --------
        get_unit_config

        """
        reaction_config = ReactionParameters(reaction).parameters

        return reaction_config

    def get_input_solver(self, process):
        """Config branch /input/solver/

        See Also
        --------
        solver_sections
        solver_time_integrator

        """
        input_solver = Dict()

        input_solver.update(self.solver_parameters.parameters)
        input_solver.user_solution_times = \
            self.get_solution_time_complete(process)
        input_solver.sections = self.get_solver_sections(process)
        input_solver.time_integrator = self.time_integrator_parameters.parameters

        return input_solver

    def get_solver_sections(self, process):
        """Config branch /input/solver/sections"""
        solver_sections = Dict()

        solver_sections.nsec = self.n_cycles * process.n_sections

        solver_sections.section_times = \
            self.get_section_times_complete(process)

        solver_sections.section_continuity = [0] * (solver_sections.nsec - 1)

        return solver_sections

    def get_input_return(self, process):
        """Config branch /input/return"""
        return_parameters = self.return_parameters.parameters
        unit_return_parameters = self.get_unit_return_parameters(process)
        return {**return_parameters, **unit_return_parameters}

    def get_unit_return_parameters(self, process):
        """Config branches for all units /input/return/unit_000 ... unit_xxx"""
        unit_return_parameters = Dict()
        for unit in process.flow_sheet.units:
            unit_index = self.get_unit_index(process, unit)
            unit_return_parameters[unit_index] = \
                unit.solution_recorder.parameters

        return unit_return_parameters

    def get_input_sensitivity(self, process):
        """Config branch /input/sensitivity"""
        sensitivity_parameters = self.sensitivity_parameters.parameters
        parameter_sensitivities = self.get_parameter_sensitivities(process)
        return {**sensitivity_parameters, **parameter_sensitivities}

    def get_parameter_sensitivities(self, process):
        """Config branches for all parameter sensitivities /input/sensitivity/param_000 ... param_xxx"""
        parameter_sensitivities = Dict()
        parameter_sensitivities.nsens = process.n_sensitivities
        for i, sens in enumerate(process.parameter_sensitivities):
            sens_index = f'param_{i:03d}'
            parameter_sensitivities[sens_index] = \
                self.get_sensitivity_config(process, sens)

        return parameter_sensitivities

    def get_sensitivity_config(self, process, sens):
        config = Dict()

        unit_indices = []
        parameters = []
        components = []

        for param, unit, associated_model, comp, coeff in zip(
                sens.parameters, sens.units, sens.associated_models, sens.components,
                sens.polynomial_coefficients):
            unit_index = process.flow_sheet.get_unit_index(unit)
            unit_indices.append(unit_index)

            if associated_model is None:
                model = unit.model
                if model == 'Inlet' and param == 'c':
                    if coeff == 0:
                        coeff = 'CONST_COEFF'
                    elif coeff == 1:
                        coeff = 'CONST_COEFF'
                    elif coeff == 2:
                        coeff = 'QUAD_COEFF'
                    elif coeff == 3:
                        coeff = 'CUBE_COEFF'
                    parameter = coeff
                else:
                    parameter = inv_unit_parameters_map[model]['parameters'][param]
            else:
                model = associated_model.model
                if isinstance(associated_model, BindingBaseClass):
                    parameter = inv_adsorption_parameters_map[model]['parameters'][param]
                if isinstance(associated_model, ReactionBaseClass):
                    parameter = inv_reaction_parameters_map[model]['parameters'][param]
            parameters.append(parameter)

            component_system = unit.component_system
            comp = -1 if comp is None else component_system.indices[comp]
            components.append(comp)

        config.sens_unit = unit_indices
        config.sens_name = parameters
        config.sens_comp = components

        config.sens_partype = -1    # !!! Check when multiple particle types enabled.
        if not all([index is None for index in sens.bound_state_indices]):
            config.sens_reaction = [
                -1 if index is None else index for index in sens.bound_state_indices
            ]
        else:
            config.sens_reaction = -1

        if not all([index is None for index in sens.bound_state_indices]):
            config.sens_boundphase = [
                -1 if index is None else index for index in sens.bound_state_indices
            ]
        else:
            config.sens_boundphase = -1

        if not all([index is None for index in sens.section_indices]):
            config.sens_section = [
                -1 if index is None else index for index in sens.section_indices
            ]
        else:
            config.sens_section = -1

        if not all([index is None for index in sens.abstols]):
            config.sens_abstol = sens.abstols

        config.factors = sens.factors

        return config

    def __str__(self):
        return 'CADET'


from CADETProcess.dataStructure import Structure, ParameterWrapper
class ModelSolverParameters(Structure):
    """Converter for model solver parameters from CADETProcess to CADET.

    Attributes
    ----------
    gs_type : {1, 0}, optional
        Valid modes:
        - 0: Classical Gram-Schmidet orthogonalization.
        - 1: Modified Gram-Schmidt.
        The default is 1.
    max_krylov : int, optional
        Size of the Krylov subspace in the iterative linear GMRES solver.
        The default is 0.
    max_restarts : int, optional
        Maximum number of restarts in the GMRES algorithm. If lack of memory is not an
        issue, better use a larger Krylov space than restarts.
        The default is 10.
    schur_safety : float, optional
        Schur safety factor.
        Influences the tradeoff between linear iterations and nonlinear error control
        The default is 1e-8.
    linear_solution_mode : int
        Valid modes:
        - 0: Automatically chose mode based on heuristic.
        - 1: Solve system of models in parallel
        - 2: Solve system of models sequentially (only possible for systems without cyclic connections)
        The default is 0.

    See Also
    --------
    Structure

    """

    gs_type = Switch(default=1, valid=[0, 1])
    max_krylov = UnsignedInteger(default=0)
    max_restarts = UnsignedInteger(default=10)
    schur_safety = UnsignedFloat(default=1e-8)
    linear_solution_mode = UnsignedInteger(default=0, ub=2)

    _parameters = [
        'gs_type',
        'max_krylov',
        'max_restarts',
        'schur_safety',
        'linear_solution_mode',
    ]


unit_parameters_map = {
    'GeneralRateModel': {
        'name': 'GENERAL_RATE_MODEL',
        'parameters': {
            'NCOMP': 'n_comp',
            'INIT_C': 'c',
            'INIT_Q': 'q',
            'INIT_CP': 'cp',
            'COL_DISPERSION': 'axial_dispersion',
            'COL_LENGTH': 'length',
            'COL_POROSITY': 'bed_porosity',
            'FILM_DIFFUSION': 'film_diffusion',
            'PAR_POROSITY': 'particle_porosity',
            'PAR_RADIUS': 'particle_radius',
            'PORE_ACCESSIBILITY': 'pore_accessibility',
            'PAR_DIFFUSION': 'pore_diffusion',
            'PAR_SURFDIFFUSION': 'surface_diffusion',
            'CROSS_SECTION_AREA': 'cross_section_area',
            'VELOCITY': 'flow_direction',
        },
        'fixed': {
            'PAR_SURFDIFFUSION_MULTIPLEX': 0,
        },
    },
    'LumpedRateModelWithPores': {
        'name': 'LUMPED_RATE_MODEL_WITH_PORES',
        'parameters': {
            'NCOMP': 'n_comp',
            'INIT_C': 'c',
            'INIT_CP': 'cp',
            'INIT_Q': 'q',
            'COL_DISPERSION': 'axial_dispersion',
            'COL_LENGTH': 'length',
            'COL_POROSITY': 'bed_porosity',
            'FILM_DIFFUSION': 'film_diffusion',
            'PAR_POROSITY': 'particle_porosity',
            'PAR_RADIUS': 'particle_radius',
            'PORE_ACCESSIBILITY': 'pore_accessibility',
            'CROSS_SECTION_AREA': 'cross_section_area',
            'VELOCITY': 'flow_direction',
        },
    },
    'LumpedRateModelWithoutPores': {
        'name': 'LUMPED_RATE_MODEL_WITHOUT_PORES',
        'parameters': {
            'NCOMP': 'n_comp',
            'INIT_C': 'c',
            'INIT_Q': 'q',
            'COL_DISPERSION': 'axial_dispersion',
            'COL_LENGTH': 'length',
            'TOTAL_POROSITY': 'total_porosity',
            'CROSS_SECTION_AREA': 'cross_section_area',
            'VELOCITY': 'flow_direction',
        },
    },
    'TubularReactor': {
        'name': 'LUMPED_RATE_MODEL_WITHOUT_PORES',
        'parameters': {
            'NCOMP': 'n_comp',
            'INIT_C': 'c',
            'COL_DISPERSION': 'axial_dispersion',
            'COL_LENGTH': 'length',
            'CROSS_SECTION_AREA': 'cross_section_area',
            'VELOCITY': 'flow_direction',
            },
        'fixed': {
            'TOTAL_POROSITY': 1,
        },
    },
    'Cstr': {
        'name': 'CSTR',
        'parameters': {
            'NCOMP': 'n_comp',
            'INIT_VOLUME': 'V',
            'INIT_C': 'c',
            'INIT_Q': 'q',
            'POROSITY': 'porosity',
            'FLOWRATE_FILTER': 'flow_rate_filter',
        },
    },
    'Inlet': {
        'name': 'INLET',
        'parameters': {
            'NCOMP': 'n_comp',
        },
        'fixed': {
            'INLET_TYPE': 'PIECEWISE_CUBIC_POLY',
        },
    },
    'Outlet': {
        'name': 'OUTLET',
        'parameters': {
            'NCOMP': 'n_comp',
        },
    },
    'MixerSplitter': {
        'name': 'CSTR',
        'parameters': {
            'NCOMP': 'n_comp',
        },
        'fixed': {
            'INIT_VOLUME': 1e-9,
            'INIT_C': [0]
        },
    },
}

inv_unit_parameters_map = {
    unit: {
        'name': values['name'],
        'parameters': {
            v: k for k, v in values['parameters'].items()
        }
    } for unit, values in unit_parameters_map.items()
}


class UnitParameters(ParameterWrapper):
    """Converter for UnitOperation parameters from CADETProcess to CADET.

    See Also
    --------
    ParameterWrapper
    AdsorptionParameters
    ReactionParameters

    """

    _baseClass = UnitBaseClass

    _unit_parameters = unit_parameters_map

    _model_parameters = _unit_parameters
    _model_type = 'UNIT_TYPE'


adsorption_parameters_map = {
    'NoBinding': {
        'name': 'NONE',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
        },
    },
    'Linear': {
        'name': 'LINEAR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'LIN_KA': 'adsorption_rate',
            'LIN_KD': 'desorption_rate'
        },
    },
    'Langmuir': {
        'name': 'MULTI_COMPONENT_LANGMUIR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCL_KA': 'adsorption_rate',
            'MCL_KD': 'desorption_rate',
            'MCL_QMAX': 'capacity'
        },
    },
    'LangmuirLDF': {
        'name': 'MULTI_COMPONENT_LANGMUIR_LDF',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCLLDF_KEQ': 'equilibrium_constant',
            'MCLLDF_KKIN': 'driving_force_coefficient',
            'MCLLDF_QMAX': 'capacity'
        },
    },
    'BiLangmuir': {
        'name': 'MULTI_COMPONENT_BILANGMUIR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCBL_KA': 'adsorption_rate',
            'MCBL_KD': 'desorption_rate',
            'MCBL_QMAX': 'capacity'
        },
    },
    'BiLangmuirLDF': {
        'name': 'MULTI_COMPONENT_BILANGMUIR_LDF',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCBLLDF_KEQ': 'equilibrium_constant',
            'MCBLLDF_KKIN': 'driving_force_coefficient',
            'MCBLLDF_QMAX': 'capacity'
        },
    },
    'FreundlichLDF': {
        'name': 'FREUNDLICH_LDF',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'FLDF_KKIN': 'driving_force_coefficient',
            'FLDF_KF': 'freundlich_coefficient',
            'FLDF_N': 'exponent'
        },
    },
    'StericMassAction': {
        'name': 'STERIC_MASS_ACTION',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'SMA_KA': 'adsorption_rate',
            'SMA_KD': 'desorption_rate',
            'SMA_LAMBDA': 'capacity',
            'SMA_NU': 'characteristic_charge',
            'SMA_SIGMA': 'steric_factor',
            'SMA_REFC0': 'reference_liquid_phase_conc',
            'SMA_REFQ': 'reference_solid_phase_conc'
        },
    },
    'AntiLangmuir': {
        'name': 'MULTI_COMPONENT_ANTILANGMUIR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCAL_KA': 'adsorption_rate',
            'MCAL_KD': 'desorption_rate',
            'MCAL_QMAX': 'capacity',
            'MCAL_ANTILANGMUIR': 'antilangmuir'
        },
    },
    'Spreading': {
        'name': 'MULTI_COMPONENT_SPREADING',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MCSPR_KA': 'adsorption_rate',
            'MCSPR_KD': 'desorption_rate',
            'MCSPR_QMAX': 'capacity',
            'MCSPR_K12': 'exchange_from_1_2',
            'MCSPR_K21': 'exchange_from_2_1',
        },
    },
    'MobilePhaseModulator': {
        'name': 'MOBILE_PHASE_MODULATOR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MPM_KA': 'adsorption_rate',
            'MPM_KD': 'desorption_rate',
            'MPM_QMAX': 'capacity',
            'MPM_BETA': 'ion_exchange_characteristic',
            'MPM_GAMMA': 'hydrophobicity'
        },
    },
    'ExtendedMobilePhaseModulator': {
        'name': 'EXTENDED_MOBILE_PHASE_MODULATOR',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'EMPM_KA': 'adsorption_rate',
            'EMPM_KD': 'desorption_rate',
            'EMPM_QMAX': 'capacity',
            'EMPM_BETA': 'ion_exchange_characteristic',
            'EMPM_GAMMA': 'hydrophobicity',
            'EMPM_COMP_MODE': 'component_mode',
        },
    },
    'SelfAssociation': {
        'name': 'SELF_ASSOCIATION',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'SAI_KA1': 'adsorption_rate',
            'SAI_KA2': 'adsorption_rate_dimerization',
            'SAI_KD': 'desorption_rate',
            'SAI_NU': 'characteristic_charge',
            'SAI_SIGMA': 'steric_factor',
            'SAI_LAMBDA': 'capacity',
            'SAI_REFC0': 'reference_liquid_phase_conc',
            'SAI_REFQ': 'reference_solid_phase_conc'
        },
    },
    'BiStericMassAction': {
        'name': 'BI_STERIC_MASS_ACTION',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'BISMA_KA': 'adsorption_rate',
            'BISMA_KD': 'desorption_rate',
            'BISMA_LAMBDA': 'capacity',
            'BISMA_NU': 'characteristic_charge',
            'BISMA_SIGMA': 'steric_factor',
            'BISMA_REFC0': 'reference_liquid_phase_conc',
            'BISMA_REFQ': 'reference_solid_phase_conc'
        },
    },
    'MultistateStericMassAction': {
        'name': 'MULTISTATE_STERIC_MASS_ACTION',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'MSSMA_KA': 'adsorption_rate',
            'MSSMA_KD': 'desorption_rate',
            'MSSMA_LAMBDA': 'capacity',
            'MSSMA_NU': 'characteristic_charge',
            'MSSMA_SIGMA': 'steric_factor',
            'MSSMA_RATES': 'conversion_rate',
            'MSSMA_REFC0': 'reference_liquid_phase_conc',
            'MSSMA_REFQ': 'reference_solid_phase_conc'
        },
    },
    'SimplifiedMultistateStericMassAction': {
        'name': 'MULTISTATE_STERIC_MASS_ACTION',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'SMSSMA_KA': 'adsorption_rate',
            'SMSSMA_KD': 'desorption_rate',
            'SMSSMA_NU_MIN': 'characteristic_charge_first',
            'SMSSMA_NU_MAX': 'characteristic_charge_last',
            'SMSSMA_NU_QUAD': 'quadratic_modifiers_charge',
            'SMSSMA_SIGMA_MIN': 'steric_factor_first',
            'SMSSMA_SIGMA_MAX': 'steric_factor_last',
            'SMSSMA_SIGMA_QUAD': 'quadratic_modifiers_steric',
            'SMSSMA_LAMBDA': 'capacity',
            'SMSSMA_KWS': 'exchange_from_weak_stronger',
            'SMSSMA_KWS_LIN': 'linear_exchange_ws',
            'SMSSMA_KWS_QUAD': 'quadratic_exchange_ws',
            'SMSSMA_KSW': 'exchange_from_stronger_weak',
            'SMSSMA_KSW_LIN': 'linear_exchange_sw',
            'SMSSMA_KSW_QUAD': 'quadratic_exchange_sw',
            'SMSSMA_REFC0': 'reference_liquid_phase_conc',
            'SMSSMA_REFQ': 'reference_solid_phase_conc'
        },
    },
    'Saska': {
        'name': 'SASKA',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'SASKA_H': 'henry_const',
            'SASKA_K': 'quadratic_factor',
        },
    },
    'GeneralizedIonExchange': {
        'name': 'GENERALIZED_ION_EXCHANGE',
        'parameters': {
            'IS_KINETIC': 'is_kinetic',
            'GIEX_KA': 'adsorption_rate',
            'GIEX_KA_LIN': 'adsorption_rate_linear',
            'GIEX_KA_QUAD': 'adsorption_rate_quadratic',
            'GIEX_KA_CUBE': 'adsorption_rate_cubic',
            'GIEX_KA_SALT': 'adsorption_rate_salt',
            'GIEX_KA_PROT': 'adsorption_rate_protein',
            'GIEX_KD': 'desorption_rate',
            'GIEX_KD_LIN': 'desorption_rate_linear',
            'GIEX_KD_QUAD': 'desorption_rate_quadratic',
            'GIEX_KD_CUBE': 'desorption_rate_cubic',
            'GIEX_KD_SALT': 'desorption_rate_salt',
            'GIEX_KD_PROT': 'desorption_rate_protein',
            'GIEX_NU_BREAKS': 'characteristic_charge_breaks',
            'GIEX_NU': 'characteristic_charge',
            'GIEX_NU_LIN': 'characteristic_charge_linear',
            'GIEX_NU_QUAD': 'characteristic_charge_quadratic',
            'GIEX_NU_CUBE': 'characteristic_charge_cubic',
            'GIEX_SIGMA': 'steric_factor',
            'GIEX_LAMBDA': 'capacity',
            'GIEX_REFC0': 'reference_liquid_phase_conc',
            'GIEX_REFQ': 'reference_solid_phase_conc',
        },
    }
}

inv_adsorption_parameters_map = {
    model: {
        'name': values['name'],
        'parameters': {
            v: k for k, v in values['parameters'].items()
        }
    } for model, values in adsorption_parameters_map.items()
}


class AdsorptionParameters(ParameterWrapper):
    """Converter for Binding model parameters from CADETProcess to CADET.

    See Also
    --------
    ParameterWrapper
    ReactionParameters
    UnitParameters
    """

    _baseClass = BindingBaseClass

    _adsorption_parameters = adsorption_parameters_map

    _model_parameters = _adsorption_parameters
    _model_type = 'ADSORPTION_MODEL'


reaction_parameters_map = {
    'NoReaction': {
        'name': 'NONE',
        'parameters': {},
    },
    'MassActionLaw': {
        'name': 'MASS_ACTION_LAW',
        'parameters': {
            'mal_stoichiometry_bulk': 'stoich',
            'mal_exponents_bulk_fwd': 'exponents_fwd',
            'mal_exponents_bulk_bwd': 'exponents_bwd',
            'mal_kfwd_bulk': 'k_fwd',
            'mal_kbwd_bulk': 'k_bwd',
            }
    },
    'MassActionLawParticle': {
        'name': 'MASS_ACTION_LAW',
        'parameters': {
            'mal_stoichiometry_liquid': 'stoich_liquid',
            'mal_exponents_liquid_fwd': 'exponents_fwd_liquid',
            'mal_exponents_liquid_bwd': 'exponents_bwd_liquid',
            'mal_kfwd_liquid': 'k_fwd_liquid',
            'mal_kbwd_liquid': 'k_bwd_liquid',

            'mal_stoichiometry_solid': 'stoich_solid',
            'mal_exponents_solid_fwd': 'exponents_fwd_solid',
            'mal_exponents_solid_bwd': 'exponents_bwd_solid',
            'mal_kfwd_solid': 'k_fwd_solid',
            'mal_kbwd_solid': 'k_bwd_solid',

            'mal_exponents_liquid_fwd_modsolid':
                'exponents_fwd_liquid_modsolid',
            'mal_exponents_liquid_bwd_modsolid':
                'exponents_bwd_liquid_modsolid',
            'mal_exponents_solid_fwd_modliquid':
                'exponents_fwd_solid_modliquid',
            'mal_exponents_solid_bwd_modliquid':
                'exponents_bwd_solid_modliquid',
        }
    }
}


inv_reaction_parameters_map = {
    model: {
        'name': values['name'],
        'parameters': {
            v: k for k, v in values['parameters'].items()
        }
    } for model, values in adsorption_parameters_map.items()
}


class ReactionParameters(ParameterWrapper):
    """Converter for Reaction model parameters from CADETProcess to CADET.

    See Also
    --------
    ParameterWrapper
    AdsorptionParameters
    UnitParameters
    """

    _baseClass = ReactionBaseClass

    _reaction_parameters = reaction_parameters_map

    _model_parameters = _reaction_parameters
    _model_type = 'REACTION_MODEL'


class SolverParameters(Structure):
    """Class for defining the solver parameters for CADET.

    Attributes
    ----------
    nthreads : int
        Number of used threads.
    consistent_init_mode : int, optional
        Consistent initialization mode.
        Valid values are:
        - 0: None
        - 1: Full
        - 2: Once, full
        - 3: Lean
        - 4: Once, lean
        - 5: Full once, then lean
        - 6: None once, then full
        - 7: None once, then lean
        The default is 1.
    consistent_init_mode_sens : int, optional
        Consistent initialization mode for parameter sensitivities.
        Valid values are:
        - 0: None
        - 1: Full
        - 2: Once, full
        - 3: Lean
        - 4: Once, lean
        - 5: Full once, then lean
        - 6: None once, then full
        - 7: None once, then lean
        The default is 1.

    See Also
    --------
    Parameters

    """

    nthreads = UnsignedInteger(default=1)
    consistent_init_mode = UnsignedInteger(default=1, ub=7)
    consistent_init_mode_sens = UnsignedInteger(default=1, ub=7)

    _parameters = [
        'nthreads', 'consistent_init_mode', 'consistent_init_mode_sens'
    ]


class SolverTimeIntegratorParameters(Structure):
    """Converter for time integartor parameters from CADETProcess to CADET.

    Attributes
    ----------
    abstol: float, optional
        Absolute tolerance in the solution of the original system.
        The default is 1e-8.
    algtol: float, optional
        Tolerance in the solution of the nonlinear consistency equations.
        The default is 1e-12.
    reltol: float, optional
        Relative tolerance in the solution of the original system.
        The default is 1e-6.
    reltol_sens: float, optional
        Relative tolerance in the solution of the sensitivity systems.
        The default is 1e-12.
    init_step_size: float, optional
        Initial time integrator step size.
        The default is 1e-6.
    max_steps: int, optional
        Maximum number of timesteps taken by IDAS
        The default is 1000000.
    max_step_size: float, optional
        Maximum size of timesteps taken by IDAS.
        The default is 0.0 (unlimited).
    errortest_sens: bool, optional
        If True: Use (forward) sensitivities in local error test
        The default is True.
    max_newton_iter: int, optional
        Maximum number of Newton iterations in time step.
        The default is 3.
    max_errtest_fail: int, optional
        Maximum number of local error test failures in time step
        The default is 7.
    max_convtest_fail: int, optional
        Maximum number of Newton convergence test failures
        The default is 10.
    max_newton_iter_sens: int, optional
        Maximum number of Newton iterations in forward sensitivity time step
        The default is 3.

    See Also
    --------
    Structure

    """

    abstol = UnsignedFloat(default=1e-8)
    algtol = UnsignedFloat(default=1e-12)
    reltol = UnsignedFloat(default=1e-6)
    reltol_sens = UnsignedFloat(default=1e-12)
    init_step_size = UnsignedFloat(default=1e-6)
    max_steps = UnsignedInteger(default=1000000)
    max_step_size = UnsignedFloat(default=0.0)
    errortest_sens = Bool(default=False)
    max_newton_iter = UnsignedInteger(default=1000000)
    max_errtest_fail = UnsignedInteger(default=1000000)
    max_convtest_fail = UnsignedInteger(default=1000000)
    max_newton_iter_sens = UnsignedInteger(default=1000000)

    _parameters = [
        'abstol', 'algtol', 'reltol', 'reltol_sens', 'init_step_size',
        'max_steps', 'max_step_size', 'errortest_sens', 'max_newton_iter',
        'max_errtest_fail', 'max_convtest_fail', 'max_newton_iter_sens'
    ]


class ReturnParameters(Structure):
    """Solution writer for system."""

    write_solution_times = Bool(default=True)
    write_solution_last = Bool(default=True)
    write_sens_last = Bool(default=True)
    split_components_data = Bool(default=False)
    split_ports_data = Bool(default=False)

    _parameters = [
        'write_solution_times', 'write_solution_last', 'write_sens_last',
        'split_components_data', 'split_ports_data'
    ]


class SensitivityParameters(Structure):
    """Sensitivity parameters."""

    sens_method = Switch(default='ad1', valid=['ad1'])

    _parameters = ['sens_method']
