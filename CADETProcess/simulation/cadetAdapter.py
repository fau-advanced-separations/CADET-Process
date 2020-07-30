from CADETProcess.common import settings

from subprocess import TimeoutExpired
import time
import tempfile

from addict import Dict
from cadet import Cadet as CadetAPI
CadetAPI.cadet_path = settings.cadet_path

from CADETProcess import CADETProcessError
from CADETProcess.common import Bool, Switch, UnsignedFloat, UnsignedInteger, \
    DependentlySizedUnsignedList, List
from CADETProcess.common import TimeSignal, Chromatogram

from CADETProcess.simulation import SolverBase
from CADETProcess.simulation import SimulationResults
from CADETProcess.processModel import NoBinding, BindingBaseClass
from CADETProcess.processModel import UnitBaseClass, Source
from CADETProcess.processModel import Process

class Cadet(SolverBase):
    """CADET class for running a simulation for given process objects.

    Attributes
    ----------
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_solver_parameters = ModelSolverParametersGroup()

        self.unit_discretization_parameters = UnitDiscretizationParametersGroup()
        self.discretization_weno_parameters = DiscretizationWenoParametersGroup()

        self.adsorption_consistency_solver_parameters = \
            ConsistencySolverParametersGroup()

        self.solver_parameters = SolverParametersGroup()
        self.time_integrator_parameters = SolverTimeIntegratorParametersGroup()

        self.return_parameters = ReturnParametersGroup()
        self.unit_return_parameters = UnitReturnParametersGroup()

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

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            if file_path is not None:
                cadet.filename = file_path
                cadet.save()
                
            cadet.filename = f.name
            cadet.save()
            start = time.time()
            try:
                 return_information = cadet.run(timeout=600)
            except TimeoutExpired:
                 raise CADETProcessError('Simulator timed out')
            elapsed = time.time() - start

            if return_information.returncode in [1,2,3]:
                self.logger.error('Simulation of {} with parameters {} failed.'\
                                     .format(process.name, process.config))
                raise CADETProcessError('CADET Error: {}'.format(
                    return_information.stderr))

            try:
                cadet.load()
                results = self.get_simulation_results(process, cadet, elapsed)
            except TypeError:
                raise CADETProcessError('Unexpected error reading SimulationResults.')

            if file_path is not None:
                cadet.filename = file_path
                cadet.save()

        return results
    
    def save_to_h5(self, process, file_path):
        cadet = CadetAPI()
        cadet.root = self.get_process_config(process)
        cadet.filename = file_path
        cadet.save()

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
        config = Dict()
        config.input.model = self.get_input_model(process)
        config.input.solver = self.get_input_solver(process)
        config.input['return'] = self.get_input_return(process)

        return config

    def get_simulation_results(self, process, cadet, time_elapsed):
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
            Tof simulation.

        Returns
        -------
        results : SimulationResults
            Simulation results including process and solver configuration.

        Notes
        -----
        !!! Implement method to read .h5 files that have no process associated.
        """
        solution = Dict()
        try:
            for unit in process.flow_sheet.units:
                solution[unit.name] = []

                unit_index = self.get_unit_index(process, unit)

                solution_complete = \
                    cadet.root.output.solution[unit_index].solution_outlet

                time = process.time
                for cycle in range(process._n_cycles):
                    start = cycle * (len(process.time) -1)
                    end = (cycle + 1) * (len(process.time) - 1) + 1
                    signal = solution_complete[start:end,:]
                    solution[unit.name].append(
                            TimeSignal(time, signal, name=unit.name))

            system_solution = {
                    'state': cadet.root.output.last_state_y,
                    'state_derivative': cadet.root.output.last_state_ydot
                    }

            chromatograms = [Chromatogram(
                    process.time, solution[chrom.name][-1].signal,
                    process.flow_rate_sections[chrom.name],
                    name = chrom.name)
                    for chrom in process.flow_sheet.chromatogram_sinks]

        except KeyError:
            raise CADETProcessError('Results don\'t match Process')

        results = SimulationResults(
                solver_name = str(self),
                solver_parameters = dict(),
                exit_flag = cadet.return_information.returncode,
                exit_message = cadet.return_information.stderr.decode(),
                time_elapsed = time_elapsed,
                process_name = process.name,
                process_config = process.config,
                process_meta = process.process_meta,
                solution = solution,
                system_solution = system_solution,
                chromatograms = chromatograms
                )

        return results

    def get_input_model(self, process):
        """Config branch /input/model/

        Note
        ----
        !!! External profiles not implemented yet

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
        index = 0

        def cadet_connections(flow_sheet):
            table = Dict()
            enum = 0

            for origin, flow_rates in flow_sheet.flow_rates.items():
                origin = flow_sheet[origin]
                origin_index = flow_sheet.get_unit_index(origin)
                for dest, flow_rate in flow_rates.destinations.items():
                    destination = flow_sheet[dest]
                    destination_index = flow_sheet.get_unit_index(destination)
                    if flow_rate > 0:
                        table[enum] = []
                        table[enum].append(int(origin_index))
                        table[enum].append(int(destination_index))
                        table[enum].append(-1)
                        table[enum].append(-1)
                        table[enum].append(flow_rate)
                        enum += 1

            ls = []
            for connection in table.values():
                ls += connection

            return ls


        for cycle in range(0, process._n_cycles):
            for time_step, events in process.time_line.items():
                [evt.perform() for evt in events]

                switch_index = 'switch' + '_{0:03d}'.format(index)
                model_connections[switch_index].section = index

                connections = cadet_connections(process.flow_sheet)
                model_connections[switch_index].connections = \
                    connections
                index += 1

        model_connections.nswitches = index

        return model_connections

    def get_unit_index(self, process, unit):
        """Helper function for getting unit index in CADET format unitXXX

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
        """Config branches for all units /input/model/unit_000 ... unit_xxx

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

        return model_units

    def get_unit_config(self, unit):
        """Config branch /input/model/unit_000 for individual unit

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

        unit_config['unit_type'] = unit_parameters.UNIT_TYPE

        unit_config['discretization'] = \
            self.unit_discretization_parameters.to_dict()
        unit_config['discretization']['NBOUND'] = [1] * unit.n_comp
        unit_config['discretization']['weno'] = \
            self.discretization_weno_parameters.to_dict()

        unit_config['adsorption_model'] = AdsorptionParametersGroup(
                unit.binding_model).ADSORPTION_MODEL

        if not isinstance(unit.binding_model, NoBinding):
            unit_config['adsorption'] = \
                self.get_adsorption_config(unit.binding_model)
            unit_config['consistency_solver'] = \
                self.adsorption_consistency_solver_parameters.to_dict()


        if isinstance(unit, Source):
            unit_config['inlet_type'] = 'PIECEWISE_CUBIC_POLY'
            unit_config['sec_000'] = {}
            unit_config['sec_000']['const_coeff'] = unit.c
            unit_config['sec_000']['lin_coeff'] = [0.0] * unit.n_comp
            unit_config['sec_000']['quad_coeff']= [0.0] * unit.n_comp
            unit_config['sec_000']['cube_coeff'] = [0.0] * unit.n_comp
        return unit_config


    def get_adsorption_config(self, binding):
        """Config branch /input/model/unit_xxx/adsorption for individual unit

        The parameters from the adsorption object are extracted and converted to
        CADET format

        See also
        --------
        get_unit_config
        """
        adsorption_config = AdsorptionParametersGroup(binding).to_dict()
        adsorption_config['consistency_solver'] = \
                self.adsorption_consistency_solver_parameters.to_dict()

        return adsorption_config


    def get_input_solver(self, process):
        """Config branch /input/solver/

        See also
        --------
        solver_sections
        solver_time_integrator
        """
        input_solver = Dict()

        input_solver.update(self.solver_parameters.to_dict())
        input_solver.USER_SOLUTION_TIMES = process._time_complete
        input_solver.sections = self.get_solver_sections(process)
        input_solver.time_integrator = \
            self.time_integrator_parameters.to_dict()

        return input_solver

    def get_solver_sections(self, process):
        """Config branch /input/solver/sections
        """
        solver_sections = Dict()

        solver_sections.nsec = process._n_cycles * len(process.time_line)
        solver_sections.section_continuity = [0] * (solver_sections.nsec - 1)

        solver_sections.section_times = [
                round((cycle*process.cycle_time + evt),1)
                for cycle in range(process._n_cycles)
                for evt in list(process.time_line)]

        solver_sections.section_times.append(
                round(process._n_cycles * process.cycle_time,1))

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

from CADETProcess.simulation.solver import ParametersGroup, ParameterWrapper
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
    SCHUR_SAFETY = UnsignedFloat(default=1.0e-8)
    _parameters = ['GS_TYPE', 'MAX_KRYLOV', 'MAX_RESTARTS', 'SCHUR_SAFETY']


class UnitParametersGroup(ParameterWrapper):
    """Class for converting UnitOperation parameters from CADETProcess to CADET.

    See also
    --------
    ParameterWrapper
    AdsorptionParametersGroup
    """
    _baseClass = UnitBaseClass

    UNIT_TYPE = Switch(valid=[
        'INLET','OUTLET', 'MIXER_SPLITTER', 'GENERAL_RATE_MODEL', 
        'LUMPED_RATE_MODEL_WITH_PORES', 'LUMPED_RATE_MODEL_WITHOUT_PORES', 
        'CSTR'])

    _unit_models = {
        'Source': 'INLET',
        'Sink': 'OUTLET',
        'GeneralRateModel': 'GENERAL_RATE_MODEL',
        'LumpedRateModelWithPores': 'LUMPED_RATE_MODEL_WITH_PORES',
        'LumpedRateModelWithoutPores': 'LUMPED_RATE_MODEL_WITHOUT_PORES',
        'TubularReactor': 'LUMPED_RATE_MODEL_WITHOUT_PORES',
        'Cstr': 'CSTR',
        'MixerSplitter': 'MIXER_SPLITTER',
        }

    _unit_parameters = {
        'GENERAL_RATE_MODEL': {
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
                'CROSS_SECTION_AREA': 'cross_section_area'
                },
        'LUMPED_RATE_MODEL_WITH_PORES': {
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
                'CROSS_SECTION_AREA': 'cross_section_area'
                },
        'LUMPED_RATE_MODEL_WITHOUT_PORES': {
                'NCOMP': 'n_comp',
                'INIT_C': 'c',
                'INIT_Q': 'q',
                'COL_DISPERSION': 'axial_dispersion',
                'COL_LENGTH': 'length',
                'TOTAL_POROSITY': 'total_porosity',
                'CROSS_SECTION_AREA': 'cross_section_area'
                },
        'CSTR': {
                'NCOMP': 'n_comp',
                'INIT_VOLUME': 'V',
                'INIT_C': 'c',
                'INIT_Q': 'q'
                },
        'INLET': {
                'NCOMP': 'n_comp',
                'CONST_COEFF': 'c'
                },
        'OUTLET': {
                'NCOMP': 'n_comp',
                },
        'MIXER_SPLITTER': {
                'NCOMP': 'n_comp',
                }
        }

    _model = UNIT_TYPE
    _models = _unit_models
    _model_parameters = _unit_parameters


class UnitDiscretizationParametersGroup(ParametersGroup):
    """Class for defining the unit_disrectization_parameters.

    See also
    --------
    ParametersGroup
    """
    NCOL = UnsignedInteger(default=100)
    NPAR = UnsignedInteger(default=5)
    NBOUND = List()
    PAR_DISC_TYPE = Switch(default='EQUIDISTANT_PAR', valid=[
                'EQUIDISTANT_PAR', 'EQUIVOLUME_PAR', 'USER_DEFINDED_PAR'])
    PAR_DISC_VECTOR = DependentlySizedUnsignedList(dep='NPAR', default=0)
    USE_ANALYTIC_JACOBIAN = Bool(default=True)
    RECONSTRUCTION = Switch(default='WENO', valid=['WENO'])
    GS_TYPE = Bool(default=True)
    MAX_KRYLOV = UnsignedInteger(default=0)
    MAX_RESTARTS = UnsignedInteger(default=10)
    SCHUR_SAFETY = UnsignedFloat(default=1.0e-8)

    _parameters = [
        'NCOL','NPAR','NBOUND', 'PAR_DISC_TYPE', 'PAR_DISC_VECTOR',
        'USE_ANALYTIC_JACOBIAN', 'RECONSTRUCTION', 'GS_TYPE', 'MAX_KRYLOV',
        'MAX_RESTARTS', 'SCHUR_SAFETY']

class DiscretizationWenoParametersGroup(ParametersGroup):
    """Class for defining the disrectization_weno_parameters

    Defines several parameters as UnsignedInteger, UnsignedFloat and save their
    names into a list named parameters.

    See also
    --------
    ParametersGroup
    """
    BOUNDARY_MODEL = UnsignedInteger(default=0, ub=3)
    WENO_EPS = UnsignedFloat(default=1e-10)
    WENO_ORDER = UnsignedInteger(default=3, ub=3)
    _parameters = ['BOUNDARY_MODEL', 'WENO_EPS', 'WENO_ORDER']


class AdsorptionParametersGroup(ParameterWrapper):
    """Class converting the binging model parameters from CADETProcess to CADET.

    See also
    --------
    ParameterWrapper
    UnitParametersGroup
    """
    _baseClass = BindingBaseClass

    IS_KINETIC = Bool(default=False)

    ADSORPTION_MODEL = Switch(default='NONE', valid=[
        'NONE', 'LINEAR', 'MULTI_COMPONENT_LANGMUIR',
        'MULTI_COMPONENT_ANTILANGMUIR', 'MOBILE_PHASE_MODULATOR',
        'STERIC_MASS_ACTION', 'SELF_ASSOCIATION', 'SASKA',
        'MULTI_COMPONENT_BILANGMUIR', 'KUMAR_MULTI_COMPONENT_LANGMUIR',
        'MULTI_COMPONENT_SPREADING', 'MULTISTATE_STERIC_MASS_ACTION',
        'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', 'BI_STERIC_MASS_ACTION'])

    _parameters = ['IS_KINETIC']

    _adsorption_models = {
        'NoBinding': 'NONE',
        'Linear': 'LINEAR',
        'Langmuir': 'MULTI_COMPONENT_LANGMUIR',
        'BiLangmuir': 'MULTI_COMPONENT_BILANGMUIR',
        'StericMassAction': 'STERIC_MASS_ACTION',
        'AntiLangmuir': 'MULTI_COMPONENT_ANTILANGMUIR'
                }
    _adsorption_parameters = {
        'NONE': {},
        'LINEAR': {
                'LIN_KA': 'adsorption_rate',
                'LIN_KD': 'desorption_rate'},
        'MULTI_COMPONENT_LANGMUIR': {
                'MCL_KA': 'adsorption_rate',
                'MCL_KD': 'desorption_rate',
                'MCL_QMAX': 'saturation_capacity'
                },
        'MULTI_COMPONENT_BILANGMUIR': {
                'MCBL_KA': 'adsorption_rate',
                'MCBL_KD': 'desorption_rate',
                'MCBL_QMAX': 'saturation_capacity'
                },
        'STERIC_MASS_ACTION': {
                'SMA_KA': 'adsorption_rate',
                'SMA_KD': 'desorption_rate',
                'SMA_NU': 'nu',
                'SMA_SIGMA': 'sigma',
                'SMA_LAMBDA': 'lambda_',
                'SMA_REF0': 'reference_liquid_phase_conc',
                'SMA_REFQ': 'reference_solid_phase_conc'
                },
        'MULTI_COMPONENT_ANTILANGMUIR': {
                'MCAL_KA': 'adsorption_rate',
                'MCAL_KD': 'desorption_rate',
                'MCAL_QMAX': 'saturation_capacity',
                'MCAL_ANTILANGMUIR': 'antilangmuir'
                }
        }

    _model = ADSORPTION_MODEL
    _models = _adsorption_models
    _model_parameters = _adsorption_parameters

class ConsistencySolverParametersGroup(ParametersGroup):
    """Class for defining the consistency solver parameters for cadet.

    The class defines several parameters for the consistency solver parameters
    with default values for cadet. The names are saved as strings in the
    parameters list.

    See also
    --------
    ParametersGroup
    """
    SOLVER_NAME = Switch(default='LEVMAR', valid=[
        'LEVMAR', 'ATRN_RES', 'ARTN_ERR', 'COMPOSITE'])
    INIT_DAMPING = UnsignedFloat(default=0.01)
    MIN_DAMPING = UnsignedFloat(default=0.0001)
    MAX_ITERATIONS = UnsignedInteger(default=50)
    SUBSOLVERS = Switch(default='LEVMAR', valid=[
        'LEVMAR', 'ATRN_RES', 'ARTN_ERR'])

    _parameters = ['SOLVER_NAME', 'INIT_DAMPING', 'MIN_DAMPING',
        'MAX_ITERATIONS', 'SUBSOLVERS']


class SolverParametersGroup(ParametersGroup):
    """Class for defining the solver parameters for cadet.

    The class defines several parameters for the solver parameters
    with default values for cadet. The names are saved as strings in the
    parameters list. For the CONSISTENT_INIT_MODE and CONSISTENT_INIT_MODE_SENS
    also upper bounds are defined.

    See also
    --------
    ParametersGroup
    """
    NTHREADS = UnsignedInteger(default=1)
    CONSISTENT_INIT_MODE = UnsignedInteger(default=1, ub=7)
    CONSISTENT_INIT_MODE_SENS = UnsignedInteger(default=1, ub=7)

    _parameters = [
        'NTHREADS', 'CONSISTENT_INIT_MODE', 'CONSISTENT_INIT_MODE_SENS']


class SolverTimeIntegratorParametersGroup(ParametersGroup):
    """Class for defining the solver time integrator parameters for cadet.

    The class defines several parameters for the solver time integrator
    parameters with default values for cadet. The names are saved as strings in
    the parameters list.

    See also
    --------
    ParametersGroup
    """
    ABSTOL = UnsignedFloat(default=1.0e-10)
    ALGTOL = UnsignedFloat(default=1.0e-12)
    RELTOL = UnsignedFloat(default=1.0e-10)
    RELTOL_SENS = UnsignedFloat(default=1.0e-12)
    INIT_STEP_SIZE = UnsignedFloat(default=1.0e-6)
    MAX_STEPS = UnsignedInteger(default=1000000)
    MAX_STEP_SIZE = UnsignedInteger(default=1000000)
    ERRORTEST_SENS = Bool(default=False)
    MAX_NEWTON_ITER = UnsignedInteger(default=1000000)
    MAX_ERRTEST_FAIL = UnsignedInteger(default=1000000)
    MAX_CONVTEST_FAIL = UnsignedInteger(default=1000000)
    MAX_NEWTON_ITER_SENS = UnsignedInteger(default=1000000)

    _parameters = [
        'ABSTOL', 'ALGTOL', 'RELTOL', 'RELTOL_SENS', 'INIT_STEP_SIZE',
        'MAX_STEPS', 'MAX_STEP_SIZE', 'ERRORTEST_SENS', 'MAX_NEWTON_ITER',
        'MAX_ERRTEST_FAIL', 'MAX_CONVTEST_FAIL', 'MAX_NEWTON_ITER_SENS']


class ReturnParametersGroup(ParametersGroup):
    """Class for defining the return parameters for cadet.

    The class defines several parameters for the return parameters as boolean
    for cadet. The names are saved as strings in the parameters list. Each
    default value is set True.

    See also
    --------
    ParametersGroup
    """
    WRITE_SOLUTION_TIMES = Bool(default=True)
    WRITE_SOLUTION_LAST = Bool(default=True)
    WRITE_SENS_LAST = Bool(default=True)
    SPLIT_COMPONENTS_DATA = Bool(default=False)
    SPLIT_PORTS_DATA = Bool(default=False)

    _parameters = [
        'WRITE_SOLUTION_TIMES', 'WRITE_SOLUTION_LAST', 'WRITE_SENS_LAST',
        'SPLIT_COMPONENTS_DATA', 'SPLIT_PORTS_DATA']


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
    WRITE_SOLUTION_INLET = Bool(default=True)
    WRITE_SOLUTION_OUTLET = Bool(default=True)
    WRITE_SOLUTION_BULK = Bool(default=False)
    WRITE_SOLUTION_PARTICLE = Bool(default=False)
    WRITE_SOLUTION_SOLID = Bool(default=False)
    WRITE_SOLUTION_FLUX = Bool(default=False)
    WRITE_SOLUTION_VOLUME = Bool(default=True)
    WRITE_SOLDOT_INLET = Bool(default=False)
    WRITE_SOLDOT_OUTLET = Bool(default=False)
    WRITE_SOLDOT_BULK = Bool(default=False)
    WRITE_SOLDOT_PARTICLE = Bool(default=False)
    WRITE_SOLDOT_SOLID = Bool(default=False)
    WRITE_SOLDOT_FLUX = Bool(default=False)
    WRITE_SOLDOT_VOLUME = Bool(default=False)
    WRITE_SENS_INLET = Bool(default=False)
    WRITE_SENS_OUTLET = Bool(default=False)
    WRITE_SENS_BULK = Bool(default=False)
    WRITE_SENS_PARTICLE = Bool(default=False)
    WRITE_SENS_SOLID = Bool(default=False)
    WRITE_SENS_FLUX = Bool(default=False)
    WRITE_SENS_VOLUME = Bool(default=False)
    WRITE_SENSDOT_INLET = Bool(default=False)
    WRITE_SENSDOT_OUTLET = Bool(default=False)
    WRITE_SENSDOT_BULK = Bool(default=False)
    WRITE_SENSDOT_PARTICLE = Bool(default=False)
    WRITE_SENSDOT_SOLID = Bool(default=False)
    WRITE_SENSDOT_FLUX = Bool(default=False)
    WRITE_SENSDOT_VOLUME = Bool(default=False)

    _parameters = [
        'WRITE_SOLUTION_INLET', 'WRITE_SOLUTION_OUTLET', 'WRITE_SOLUTION_BULK',
        'WRITE_SOLUTION_PARTICLE', 'WRITE_SOLUTION_SOLID', 'WRITE_SOLUTION_FLUX',
        'WRITE_SOLUTION_VOLUME', 'WRITE_SOLDOT_INLET', 'WRITE_SOLDOT_OUTLET',
        'WRITE_SOLDOT_BULK', 'WRITE_SOLDOT_PARTICLE', 'WRITE_SOLDOT_SOLID',
        'WRITE_SOLDOT_FLUX', 'WRITE_SOLDOT_VOLUME', 'WRITE_SENS_INLET',
        'WRITE_SENS_OUTLET', 'WRITE_SENS_BULK', 'WRITE_SENS_PARTICLE',
        'WRITE_SENS_SOLID', 'WRITE_SENS_FLUX', 'WRITE_SENS_VOLUME',
        'WRITE_SENSDOT_INLET', 'WRITE_SENSDOT_OUTLET', 'WRITE_SENSDOT_BULK',
        'WRITE_SENSDOT_PARTICLE', 'WRITE_SENSDOT_SOLID', 'WRITE_SENSDOT_FLUX',
        'WRITE_SENSDOT_VOLUME']


class SensitivityParametersGroup(ParametersGroup):
    """Class for defining the sensitivity parameters.

    The sensitivity parameters NSENS and SENS_METHOD are defined with default
    values.

    See also
    --------
    ParametersGroup
    """
    NSENS = UnsignedInteger(default=0)
    SENS_METHOD = Switch(default='ad1', valid=['ad1'])