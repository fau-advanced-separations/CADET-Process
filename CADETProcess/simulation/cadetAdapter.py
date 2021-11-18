import os
import platform
from pathlib import Path
import shutil
import subprocess
from subprocess import TimeoutExpired
import time
import tempfile

from addict import Dict
import numpy as np
from cadet import Cadet as CadetAPI

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Bool, Switch, UnsignedFloat, UnsignedInteger,
    DependentlySizedUnsignedList, List,
)
from CADETProcess.common import TimeSignal, Chromatogram

from .solver import SolverBase
from .solver import SimulationResults
from .solution import (
    SolutionIO, SolutionBulk, SolutionParticle, SolutionSolid, SolutionVolume
)
from CADETProcess.processModel import NoBinding, BindingBaseClass
from CADETProcess.processModel import NoReaction, ReactionBaseClass
from CADETProcess.processModel import NoDiscretization
from CADETProcess.processModel import (
    UnitBaseClass, Source, Cstr, LumpedRateModelWithoutPores
)
from CADETProcess.processModel import Process

class Cadet(SolverBase):
    """CADET class for running a simulation for given process objects.

    Attributes
    ----------
    install_path: str
        Path to the installation of CADET
    temp_dir : str
        Path to directory for temporary files
    time_out : UnsignedFloat
        Maximum duration for simulations
    model_solver_parameters : ModelSolverParametersGroup
        Container for solver parameters
    unit_discretization_parameters : UnitDiscretizationParametersGroup
        Container for unit discretization parameters
    discretization_weno_parameters : DiscretizationWenoParametersGroup
        Container for weno discretization parameters in units
    adsorption_consistency_solver_parameters : ConsistencySolverParametersGroup
        Container for consistency solver parameters
    solver_parameters : SolverParametersGroup
        Container for general solver settings
    time_integrator_parameters : SolverTimeIntegratorParametersGroup
        Container for time integrator parameters
    return_parameters : ReturnParametersGroup
        Container for return information of the system
    unit_return_parameters : UnitReturnParametersGroup
        Container for return information of units

    Note
    ----
    !!! UnitParametersGroup and AdsorptionParametersGroup should be implemented
    with global options that are then copied for each unit in get_unit_config
    !!! Implement method for loading CADET file that have not been generated
    with CADETProcess and create Process

    See also
    --------
    ReturnParametersGroup
    ModelSolverParametersGroup
    SolverParametersGroup
    SolverTimeIntegratorParametersGroup
    cadetInterface
    """
    timeout = UnsignedFloat()

    def __init__(self, install_path=None, temp_dir=None, *args, **kwargs):
        self.install_path = install_path
        self.temp_dir = temp_dir

        super().__init__(*args, **kwargs)

        self.model_solver_parameters = ModelSolverParametersGroup()
        self.solver_parameters = SolverParametersGroup()
        self.time_integrator_parameters = SolverTimeIntegratorParametersGroup()

        self.return_parameters = ReturnParametersGroup()
        self.unit_return_parameters = UnitReturnParametersGroup()

    @property
    def install_path(self):
        """str: Path to the installation of CADET

        Parameters
        ----------
        install_path : str or None
            Path to the installation of CADET.
            If None, the system installation will be used.

        Raises
        ------
        FileNotFoundError
            If CADET can not be found.

        See Also
        --------
        check_cadet()
        """
        return self._install_path

    @install_path.setter
    def install_path(self, install_path):
        if install_path is None:
            try:
                if platform.system() == 'Windows':
                    executable_path = Path(shutil.which("cadet-cli.exe"))
                else:
                    executable_path = Path(shutil.which("cadet-cli"))
            except TypeError:
                raise FileNotFoundError(
                    "CADET could not be found. Please set an install path"
                )
            install_path = executable_path.parent.parent

        install_path = Path(install_path)
        if platform.system() == 'Windows':
            cadet_bin_path = install_path / "bin" / "cadet-cli.exe"
        else:
            cadet_bin_path = install_path / "bin" / "cadet-cli"

        if cadet_bin_path.exists():
            self._install_path = install_path
            CadetAPI.cadet_path = cadet_bin_path
        else:
            raise FileNotFoundError(
                "CADET could not be found. Please check the path"
            )

        cadet_lib_path = install_path / "lib"
        try:
            if cadet_lib_path.as_posix() not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] = \
                    cadet_lib_path.as_posix() \
                    + os.pathsep \
                    + os.environ['LD_LIBRARY_PATH']
        except KeyError:
            os.environ['LD_LIBRARY_PATH'] = cadet_lib_path.as_posix()


    def check_cadet(self):
        """Wrapper around a basic CADET example for testing functionality"""
        if platform.system() == 'Windows':
            lwe_path = self.install_path / "bin" / "createLWE.exe"
        else:
            lwe_path = self.install_path / "bin" / "createLWE"
        ret = subprocess.run(
            [lwe_path.as_posix()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.temp_dir
        )
        if ret.returncode != 0:
            if ret.stdout:
                print('Output', ret.stdout.decode('utf-8'))
            if ret.stderr:
                print('Errors', ret.stderr.decode('utf-8'))
            raise CADETProcessError(
                "Failure: Creation of test simulation ran into problems"
            )

        lwe_hdf5_path = Path(self.temp_dir) / 'LWE.h5'

        sim = CadetAPI()
        sim.filename = lwe_hdf5_path.as_posix()
        data = sim.run()
        os.remove(sim.filename)

        if data.returncode == 0:
            print("Test simulation completed successfully")
        else:
            print(data)
            raise CADETProcessError(
                "Simulation failed"
            )

    @property
    def temp_dir(self):
        return tempfile.gettempdir()

    @temp_dir.setter
    def temp_dir(self, temp_dir):
        if temp_dir is not None:
            try:
                exists = Path(temp_dir).exists()
            except TypeError:
                raise CADETProcessError('Not a valid path')
            if not exists:
                raise CADETProcessError('Not a valid path')

        tempfile.tempdir = temp_dir

    def get_tempfile_name(self):
        f = next(tempfile._get_candidate_names())
        return os.path.join(self.temp_dir, f + '.h5')


    def run(self, process, file_path=None):
        """Interface to the solver run function

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

        See also
        --------
        get_process_config
        get_simulation_results
        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        cadet = CadetAPI()
        cadet.root = self.get_process_config(process)

        if file_path is None:
            cadet.filename = self.get_tempfile_name()
        else:
            cadet.filename = file_path

        cadet.save()

        try:
            start = time.time()
            return_information = cadet.run(timeout=self.timeout)
            elapsed = time.time() - start
        except TimeoutExpired:
             raise CADETProcessError('Simulator timed out')

        if return_information.returncode != 0:
            self.logger.error(
                'Simulation of {} with parameters {} failed.'.format(
                    process.name, process.config
                )
            )
            raise CADETProcessError(
                'CADET Error: {}'.format(return_information.stderr)
            )

        try:
            cadet.load()
            results = self.get_simulation_results(
                process, cadet, elapsed, return_information
            )
        except TypeError:
            raise CADETProcessError('Unexpected error reading SimulationResults.')

        # Remove files
        if file_path is None:
            os.remove(cadet.filename)

        return results

    def save_to_h5(self, process, file_path):
        cadet = CadetAPI()
        cadet.root = self.get_process_config(process)
        cadet.filename = file_path
        cadet.save()

    def load_from_h5(self, file_path):
        cadet = CadetAPI()
        cadet.filename = file_path
        cadet.load()

        return cadet

    def get_process_config(self, process):
        """Create the CADET config.

        Returns
        -------
        config : Dict
            /

        Note
        ----
        Sensitivities not implemented yet.

        See also
        --------
        input_model
        input_solver
        input_return
        """
        process.lock = True
        config = Dict()
        config.input.model = self.get_input_model(process)
        config.input.solver = self.get_input_solver(process)
        config.input['return'] = self.get_input_return(process)
        process.lock = False

        return config

    def get_simulation_results(
        self,
        process,
        cadet,
        time_elapsed,
        return_information,
        ):
        """Saves the simulated results for each unit into the dictionary
        concentration_record for the complete simulation and splitted into each
        cycle.

        For each unit in the flow_sheet of a process the index
        of the unit is get by calling the method get_unit_index. The process
        results of the simualtion are saved in concentration_record of the
        process for each unit for the key complete. For saving the process
        resulst for each cycle start and end variables are defined an saved
        under the key cycles in the concentration_record dictionary for each
        unit.

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

        Notes
        -----
        !!! Implement method to read .h5 files that have no process associated.
        """
        time = process.time

        solution = Dict()
        from collections import defaultdict
        try:
            for unit in process.flow_sheet.units:
                solution[unit.name] = defaultdict(list)
                unit_index = self.get_unit_index(process, unit)
                unit_solution = cadet.root.output.solution[unit_index]
                unit_coordinates = cadet.root.output.coordinates[unit_index].copy()
                particle_coordinates = \
                    unit_coordinates.pop('particle_coordinates_000', None)
                        
                for cycle in range(process._n_cycles):
                    start = cycle * (len(process.time) -1)
                    end = (cycle + 1) * (len(process.time) - 1) + 1
                
                    if 'solution_inlet' in unit_solution.keys():
                        sol_inlet = unit_solution.solution_inlet[start:end,:]
                        solution[unit.name]['inlet'].append(
                            SolutionIO(unit.component_system, time, sol_inlet)
                        )
                    
                    if 'solution_outlet' in unit_solution.keys():
                        sol_outlet = unit_solution.solution_outlet[start:end,:]
                        solution[unit.name]['outlet'].append(
                            SolutionIO(unit.component_system, time, sol_outlet)
                        )
    
                    if 'solution_bulk' in unit_solution.keys():
                        sol_bulk = unit_solution.solution_bulk[start:end,:]
                        solution[unit.name]['bulk'].append(
                            SolutionBulk(
                                unit.component_system, time, sol_bulk, 
                                **unit_coordinates
                            )
                        )
                        
                    if 'solution_particle' in unit_solution.keys():
                        sol_particle = unit_solution.solution_particle[start:end,:]
                        solution[unit.name]['particle'].append(
                            SolutionParticle(
                                unit.component_system, time, sol_particle,
                                **unit_coordinates, 
                                particle_coordinates=particle_coordinates
                            )
                        )
    
                    if 'solution_solid' in unit_solution.keys():
                        sol_solid = unit_solution.solution_solid[start:end,:]
                        solution[unit.name]['solid'].append(
                            SolutionSolid(
                                unit.component_system, unit.binding_model.n_states, 
                                time, sol_solid, 
                                **unit_coordinates,
                                particle_coordinates=particle_coordinates
                            )
                        )
                        
                    if 'solution_volume' in unit_solution.keys():
                        sol_volume = unit_solution.solution_volume[start:end,:]
                        solution[unit.name]['volume'].append(
                            SolutionVolume(unit.component_system, time, sol_volume)
                        )
                        
            solution = Dict(solution)
            
            system_state = {
                'state': cadet.root.output.last_state_y,
                'state_derivative': cadet.root.output.last_state_ydot
            }

            chromatograms = [
                Chromatogram(
                    process.time, solution[chrom.name].outlet[-1].solution,
                    process.flow_rate_timelines[chrom.name].total,
                    name=chrom.name
                )
                for chrom in process.flow_sheet.chromatogram_sinks
            ]

        except KeyError:
            raise CADETProcessError('Results don\'t match Process')

        results = SimulationResults(
            solver_name = str(self),
            solver_parameters = dict(),
            exit_flag = return_information.returncode,
            exit_message = return_information.stderr.decode(),
            time_elapsed = time_elapsed,
            process_name = process.name,
            process_config = process.config,
            process_meta = process.process_meta,
            solution_cycles = solution,
            system_state = system_state,
            chromatograms = chromatograms
        )

        return results

    def get_input_model(self, process):
        """Config branch /input/model/

        Note
        ----
        !!! External functions not implemented yet

        See also
        --------
        model_connections
        model_solver
        model_units
        input_model_parameters
        """
        input_model = Dict()

        input_model.connections = self.get_model_connections(process)
        # input_model.external = self.model_external # !!! not working yet
        input_model.solver = self.model_solver_parameters.to_dict()
        input_model.update(self.get_model_units(process))

        if process.system_state is not None:
            input_model['INIT_STATE_Y'] = process.system_state
        if process.system_state_derivative is not None:
            input_model['INIT_STATE_YDOT'] = process.system_state_derivative

        return input_model

    def get_model_connections(self, process):
        """Config branch /input/model/connections
        """
        model_connections = Dict()
        model_connections['CONNECTIONS_INCLUDE_DYNAMIC_FLOW'] = 1
        index = 0

        section_states = process.flow_rate_section_states

        for cycle in range(0, process._n_cycles):
            for flow_rates_state in section_states.values():

                switch_index = 'switch' + '_{0:03d}'.format(index)
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
                    table[enum] += flow_rate.tolist()
                    enum += 1

        ls = []
        for connection in table.values():
            ls += connection

        return ls

    def get_unit_index(self, process, unit):
        """Helper function for getting unit index in CADET format unit_xxx.

        Parameters
        -----------
        process : Process
            process to be simulated
        unit : UnitOperation
            Indexed object

        Returns
        -------
        unit_index : str
            Return the unit index in CADET format unitXXX
        """
        index = process.flow_sheet.get_unit_index(unit)
        return 'unit' + '_{0:03d}'.format(index)

    def get_model_units(self, process):
        """Config branches for all units /input/model/unit_000 ... unit_xxx.

        See also
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

        The parameters from the unit are extracted and converted to CADET format

        Note
        ----
        For now, only constant values for the concentration in sources are valid.

        In CADET, the parameter unit_config['discretization'].NBOUND should be
        moved to binding config or unit config

        See also
        --------
        get_adsorption_config
        """
        unit_parameters = UnitParametersGroup(unit)

        unit_config = Dict(unit_parameters.to_dict())

        if not isinstance(unit.binding_model, NoBinding):
            n_bound = [unit.binding_model.n_states] * unit.binding_model.n_comp
            unit_config['adsorption'] = \
                    self.get_adsorption_config(unit.binding_model)
            unit_config['adsorption_model'] = unit_config['adsorption']['ADSORPTION_MODEL']
        else:
            n_bound = unit.n_comp*[0]

        if not isinstance(unit.discretization, NoDiscretization):
            unit_config['discretization'] = unit.discretization.parameters

        if isinstance(unit, Cstr) and not isinstance(unit.binding_model, NoBinding):
            unit_config['nbound'] = n_bound
        else:
            unit_config['discretization']['nbound'] = n_bound

        if not isinstance(unit.bulk_reaction_model, NoReaction):
            parameters = self.get_reaction_config(unit.bulk_reaction_model)
            if isinstance(unit, LumpedRateModelWithoutPores):
                unit_config['reaction_model'] = parameters['REACTION_MODEL']
                unit_config['reaction'] = parameters
            else:
                unit_config['reaction_model'] = parameters['REACTION_MODEL']
                unit_config['reaction_bulk'] = parameters

        if not isinstance(unit.particle_reaction_model, NoReaction):
            parameters = self.get_reaction_config(unit.particle_reaction_model)
            unit_config['reaction_model_particle'] = parameters['REACTION_MODEL']
            unit_config['reaction_particle'].update(parameters)

        if isinstance(unit, Source):
            unit_config['sec_000']['const_coeff'] = unit.c[:,0]
            unit_config['sec_000']['lin_coeff'] = unit.c[:,1]
            unit_config['sec_000']['quad_coeff']= unit.c[:,2]
            unit_config['sec_000']['cube_coeff'] = unit.c[:,3]

        return unit_config

    def set_section_dependent_parameters(self, model_units, process):
        """Add time dependent model parameters to units
        """
        section_states = process.section_states.values()

        section_index = 0
        for cycle in range(0, process._n_cycles):
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
                    if isinstance(unit, Source) and param_name == 'c':
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
        unit_index = 'unit' + '_{0:03d}'.format(unit_index)
        section_index = 'sec' + '_{0:03d}'.format(sec_index)

        model_units[unit_index][section_index]['const_coeff'] = coeffs[:,0]
        model_units[unit_index][section_index]['lin_coeff'] = coeffs[:,1]
        model_units[unit_index][section_index]['quad_coeff']= coeffs[:,2]
        model_units[unit_index][section_index]['cube_coeff'] = coeffs[:,3]

    def add_parameter_section(
            self, model_units, sec_index, unit_index, unit_model, parameter, state
        ):
        """Add section value to parameter branch.
        """
        unit_index = 'unit' + '_{0:03d}'.format(unit_index)
        parameter_name = inv_unit_parameters_map[unit_model]['parameters'][parameter]

        if sec_index == 0:
            model_units[unit_index][parameter_name] = []
        model_units[unit_index][parameter_name] += list(state.ravel())

    def get_adsorption_config(self, binding):
        """Config branch /input/model/unit_xxx/adsorption for individual unit

        The parameters from the adsorption object are extracted and converted to
        CADET format

        See also
        --------
        get_unit_config
        """
        adsorption_config = AdsorptionParametersGroup(binding).to_dict()

        return adsorption_config

    def get_reaction_config(self, reaction):
        """Config branch /input/model/unit_xxx/reaction for individual unit

        Parameters
        ----------
        reaction : ReactionBaseClass
            Reaction configuration object

        See also
        --------
        get_unit_config
        """
        reaction_config = ReactionParametersGroup(reaction).to_dict()

        return reaction_config


    def get_input_solver(self, process):
        """Config branch /input/solver/

        See also
        --------
        solver_sections
        solver_time_integrator
        """
        input_solver = Dict()

        input_solver.update(self.solver_parameters.to_dict())
        input_solver.user_solution_times = process._time_complete
        input_solver.sections = self.get_solver_sections(process)
        input_solver.time_integrator = \
            self.time_integrator_parameters.to_dict()

        return input_solver

    def get_solver_sections(self, process):
        """Config branch /input/solver/sections
        """
        solver_sections = Dict()

        solver_sections.nsec = process._n_cycles * process.n_sections
        solver_sections.section_times = [
            round((cycle*process.cycle_time + evt),1)
            for cycle in range(process._n_cycles)
            for evt in process.section_times[0:-1]
        ]
        solver_sections.section_times.append(
            round(process._n_cycles * process.cycle_time,1)
        )
        solver_sections.section_continuity = [0] * (solver_sections.nsec - 1)

        return solver_sections

    def get_input_return(self, process):
        """Config branch /input/return
        """
        return_parameters = self.return_parameters.to_dict()
        unit_return_parameters = self.get_unit_return_parameters(process)
        return {**return_parameters, **unit_return_parameters}

    def get_unit_return_parameters(self, process):
        """Config branches for all units /input/return/unit_000 ... unit_xxx
        """
        unit_return_parameters = Dict()
        for unit in process.flow_sheet.units:
            unit_index = self.get_unit_index(process, unit)
            unit_return_parameters[unit_index] = \
                self.unit_return_parameters.to_dict()

        return unit_return_parameters

    def __str__(self):
        return 'CADET'

from CADETProcess.dataStructure import ParametersGroup, ParameterWrapper
class ModelSolverParametersGroup(ParametersGroup):
    """Class for defining the model_solver_parameters.

    Defines several parameters as UnsignedInteger with default values and save
    their names into a list named parameters.

    See also
    --------
    ParametersGroup
    """
    GS_TYPE = UnsignedInteger(default=1, ub=1)
    MAX_KRYLOV = UnsignedInteger(default=0)
    MAX_RESTARTS = UnsignedInteger(default=10)
    SCHUR_SAFETY = UnsignedFloat(default=1e-8)
    _parameters = ['GS_TYPE', 'MAX_KRYLOV', 'MAX_RESTARTS', 'SCHUR_SAFETY']

unit_parameters_map = {
    'GeneralRateModel': {
        'name': 'GENERAL_RATE_MODEL',
        'parameters':{
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
        'parameters':{
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
        'parameters':{
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
        'parameters':{
            'NCOMP': 'n_comp',
            'INIT_C': 'c',
            'INIT_Q': 'q',
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
        'parameters':{
            'NCOMP': 'n_comp',
            'INIT_VOLUME': 'V',
            'INIT_C': 'c',
            'INIT_Q': 'q',
            'POROSITY': 'porosity',
            'FLOWRATE_FILTER': 'flow_rate_filter',
        },
    },
    'Source': {
        'name': 'INLET',
        'parameters':{
            'NCOMP': 'n_comp',
        },
        'fixed': {
            'INLET_TYPE': 'PIECEWISE_CUBIC_POLY',
        },
    },
    'Sink': {
        'name': 'OUTLET',
        'parameters':{
            'NCOMP': 'n_comp',
        },
    },
    'MixerSplitter': {
        'name': 'CSTR',
        'parameters':{
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

class UnitParametersGroup(ParameterWrapper):
    """Class for converting UnitOperation parameters from CADETProcess to CADET.

    See also
    --------
    ParameterWrapper
    AdsorptionParametersGroup
    ReactionParametersGroup
    """
    _baseClass = UnitBaseClass

    _unit_parameters = unit_parameters_map

    _model_parameters = _unit_parameters
    _model_type = 'UNIT_TYPE'


adsorption_parameters_map = {
    'NoBinding': {
        'name': 'NONE',
        'parameters': {},
    },
    'Linear': {
        'name': 'LINEAR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'LIN_KA': 'adsorption_rate',
            'LIN_KD': 'desorption_rate'
        },
    },
    'Langmuir': {
        'name': 'MULTI_COMPONENT_LANGMUIR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'MCL_KA': 'adsorption_rate',
            'MCL_KD': 'desorption_rate',
            'MCL_QMAX': 'saturation_capacity'
        },
    },
    'BiLangmuir': {
        'name': 'MULTI_COMPONENT_BILANGMUIR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'MCBL_KA': 'adsorption_rate',
            'MCBL_KD': 'desorption_rate',
            'MCBL_QMAX': 'saturation_capacity'
        },
    },
    'StericMassAction': {
        'name': 'STERIC_MASS_ACTION',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'SMA_KA': 'adsorption_rate',
            'SMA_KD': 'desorption_rate',
            'SMA_LAMBDA': 'capacity',
            'SMA_NU': 'characteristic_charge',
            'SMA_SIGMA': 'steric_factor',
            'SMA_REF0': 'reference_liquid_phase_conc',
            'SMA_REFQ': 'reference_solid_phase_conc'
        },
    },
    'AntiLangmuir': {
        'name': 'MULTI_COMPONENT_ANTILANGMUIR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'MCAL_KA': 'adsorption_rate',
            'MCAL_KD': 'desorption_rate',
            'MCAL_QMAX': 'saturation_capacity',
            'MCAL_ANTILANGMUIR': 'antilangmuir'
        },
    },
    'MobilePhaseModulator' : {
        'name': 'MOBILE_PHASE_MODULATOR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'MPM_KA': 'adsorption_rate',
            'MPM_KD': 'desorption_rate',
            'MPM_QMAX': 'maximum_adsorption_capacity',
            'MPM_BETA': 'ion_exchange_characteristic',
            'MPM_GAMMA': 'hydrophobicity'
        },
    },
    'ExtendedMobilePhaseModulator' : {
        'name': 'EXTENDED_MOBILE_PHASE_MODULATOR',
        'parameters': {
            'IS_KINETIC' : 'is_kinetic',
            'EMPM_KA': 'adsorption_rate',
            'EMPM_KD': 'desorption_rate',
            'EMPM_QMAX': 'maximum_adsorption_capacity',
            'EMPM_BETA': 'ion_exchange_characteristic',
            'EMPM_GAMMA': 'hydrophobicity',
            'EMPM_COMP_MODE': 'component_mode',
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

class AdsorptionParametersGroup(ParameterWrapper):
    """Class for converting binding model parameters from CADETProcess to CADET.

    See also
    --------
    ParameterWrapper
    ReactionParametersGroup
    UnitParametersGroup
    """
    _baseClass = BindingBaseClass

    _adsorption_parameters = adsorption_parameters_map

    _model_parameters = _adsorption_parameters
    _model_type = 'ADSORPTION_MODEL'


class ReactionParametersGroup(ParameterWrapper):
    """Converter for particle solid reaction parameters from CADETProcess to CADET.

    See also
    --------
    ParameterWrapper
    ReactionParametersGroup
    UnitParametersGroup
    """
    _baseClass = ReactionBaseClass

    _reaction_models = {
        'NoReaction': 'NONE',
        'MassAction': 'MASS_ACTION_LAW',
                }
    _reaction_parameters = {
        'NoReaction': {
            'name': 'NONE',
            'parameters':{},
        },
        'MassActionLaw': {
            'name': 'MASS_ACTION_LAW',
            'parameters':{
                'mal_stoichiometry_bulk': 'stoich',
                'mal_exponents_bulk_fwd' : 'exponents_fwd',
                'mal_exponents_bulk_bwd' : 'exponents_bwd',
                'mal_kfwd_bulk' : 'k_fwd',
                'mal_kbwd_bulk' : 'k_bwd',
                }
        },
        'MassActionLawParticle': {
            'name': 'MASS_ACTION_LAW',
            'parameters':{
                'mal_stoichiometry_liquid': 'stoich_liquid',
                'mal_exponents_liquid_fwd' : 'exponents_fwd_liquid',
                'mal_exponents_liquid_bwd' : 'exponents_bwd_liquid',
                'mal_kfwd_liquid' : 'k_fwd_liquid',
                'mal_kbwd_liquid' : 'k_bwd_liquid',

                'mal_stoichiometry_solid': 'stoich_solid',
                'mal_exponents_solid_fwd' : 'exponents_fwd_solid',
                'mal_exponents_solid_bwd' : 'exponents_bwd_solid',
                'mal_kfwd_solid' : 'k_fwd_solid',
                'mal_kbwd_solid' : 'k_bwd_solid',

                'mal_exponents_liquid_fwd_modsolid' : 'exponents_fwd_liquid_modsolid',
                'mal_exponents_liquid_bwd_modsolid' : 'exponents_bwd_liquid_modsolid',
                'mal_exponents_solid_fwd_modliquid' : 'exponents_fwd_solid_modliquid',
                'mal_exponents_solid_bwd_modliquid' : 'exponents_bwd_solid_modliquid',
            }
        }
    }

    _model_parameters = _reaction_parameters
    _model_type = 'REACTION_MODEL'


class SolverParametersGroup(ParametersGroup):
    """Class for defining the solver parameters for cadet.

    See also
    --------
    ParametersGroup
    """
    nthreads = UnsignedInteger(default=1)
    consistent_init_mode = UnsignedInteger(default=1, ub=7)
    consistent_init_mode_sens = UnsignedInteger(default=1, ub=7)

    _parameters = [
        'nthreads', 'consistent_init_mode', 'consistent_init_mode_sens'
    ]


class SolverTimeIntegratorParametersGroup(ParametersGroup):
    """Class for defining the solver time integrator parameters for cadet.

    See also
    --------
    ParametersGroup
    """
    abstol = UnsignedFloat(default=1e-8)
    algtol = UnsignedFloat(default=1e-12)
    reltol = UnsignedFloat(default=1e-6)
    reltol_sens = UnsignedFloat(default=1e-12)
    init_step_size = UnsignedFloat(default=1e-6)
    max_steps = UnsignedInteger(default=1000000)
    max_step_size = UnsignedInteger(default=1000000)
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


class ReturnParametersGroup(ParametersGroup):
    """Class for defining the return parameters for cadet.

    See also
    --------
    ParametersGroup
    """
    write_solution_times = Bool(default=True)
    write_solution_last = Bool(default=True)
    write_sens_last = Bool(default=True)
    split_components_data = Bool(default=False)
    split_ports_data = Bool(default=False)

    _parameters = [
        'write_solution_times', 'write_solution_last', 'write_sens_last',
        'split_components_data', 'split_ports_data'
    ]


class UnitReturnParametersGroup(ParametersGroup):
    """Class for defining the unit return parameters for cadet.

    The class defines several parameters for the unit return parameters as
    boolean for cadet. The names are saved as strings in the parameters list.
    Only the WRITE_SOLUTION_OUTLET ans WRITE_SOLUTION_INLET are set True as
    default value. The remaining unit return parameters are set False for
    default value.

    See also
    --------
    ParametersGroup
    """
    write_coordinates = Bool(default=True)
    write_solution_inlet = Bool(default=True)
    write_solution_outlet = Bool(default=True)
    write_solution_bulk = Bool(default=False)
    write_solution_particle = Bool(default=False)
    write_solution_solid = Bool(default=False)
    write_solution_flux = Bool(default=False)
    write_solution_volume = Bool(default=True)
    write_soldot_inlet = Bool(default=False)
    write_soldot_outlet = Bool(default=False)
    write_soldot_bulk = Bool(default=False)
    write_soldot_particle = Bool(default=False)
    write_soldot_solid = Bool(default=False)
    write_soldot_flux = Bool(default=False)
    write_soldot_volume = Bool(default=False)
    write_sens_inlet = Bool(default=False)
    write_sens_outlet = Bool(default=False)
    write_sens_bulk = Bool(default=False)
    write_sens_particle = Bool(default=False)
    write_sens_solid = Bool(default=False)
    write_sens_flux = Bool(default=False)
    write_sens_volume = Bool(default=False)
    write_sensdot_inlet = Bool(default=False)
    write_sensdot_outlet = Bool(default=False)
    write_sensdot_bulk = Bool(default=False)
    write_sensdot_particle = Bool(default=False)
    write_sensdot_solid = Bool(default=False)
    write_sensdot_flux = Bool(default=False)
    write_sensdot_volume = Bool(default=False)
    write_solution_last_unit = Bool(default=False)

    _parameters = [
        'write_coordinates',
        'write_solution_inlet', 'write_solution_outlet', 'write_solution_bulk',
        'write_solution_particle', 'write_solution_solid', 'write_solution_flux',
        'write_solution_volume', 
        'write_soldot_inlet', 'write_soldot_outlet', 'write_soldot_bulk', 
        'write_soldot_particle', 'write_soldot_solid', 'write_soldot_flux', 
        'write_soldot_volume', 
        'write_sens_inlet', 'write_sens_outlet', 'write_sens_bulk', 
        'write_sens_particle', 'write_sens_solid', 'write_sens_flux', 
        'write_sens_volume',
        'write_sensdot_inlet', 'write_sensdot_outlet', 'write_sensdot_bulk',
        'write_sensdot_particle', 'write_sensdot_solid', 'write_sensdot_flux',
        'write_sensdot_volume'
    ]


class SensitivityParametersGroup(ParametersGroup):
    """Class for defining the sensitivity parameters.

    The sensitivity parameters NSENS and SENS_METHOD are defined with default
    values.

    See also
    --------
    ParametersGroup
    """
    nsens = UnsignedInteger(default=0)
    sens_method = Switch(default='ad1', valid=['ad1'])
    