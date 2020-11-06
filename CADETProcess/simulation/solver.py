from abc import abstractmethod
import os

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.common import settings
from CADETProcess.common import log, log_time, log_results, log_exceptions
from CADETProcess.common import StructMeta
from CADETProcess.common import Bool, Dict, String, List, Switch, \
                            UnsignedInteger, UnsignedFloat
from CADETProcess.processModel import Process
from CADETProcess.simulation import StationarityEvaluator
from CADETProcess.common import TimeSignal


class SolverBase(metaclass=StructMeta):
    """BaseClass for Solver APIs

    Holds the configuration of the individual solvers and gives an interface for
    calling the run method. The class has to convert the process configuration
    into the APIs configuration format and convert the results back to the
    CADETProcess format.

    Attributes
    ----------
    n_cycles : int
        Number of cycles to be simulated
    n_cycles_min : int
        If simulate_to_stationarity: Minimum number of cycles to be simulated. 
    n_cycles_max : int
        If simulate_to_stationarity: Maximum number of cycles to be simulated.
    simulate_to_stationarity : bool
        Simulate until stationarity is reached
        

    See also
    --------
    Process
    StationarityEvaluator
    """
    n_cycles = UnsignedInteger(default=1)
    evaluate_stationarity = Bool(default=False)
    n_cycles_min = UnsignedInteger(default=3)
    n_cycles_max = UnsignedInteger(default=100)

    def __init__(self, stationarity_evaluator=None):
        self.logger = log.get_logger('Simulation')

        if stationarity_evaluator is None:
            self._stationarity_evaluator = StationarityEvaluator()
        else:
            self.stationarity_evaluator = stationarity_evaluator
            self.evaluate_stationarity = True

    def simulate(self, process, previous_results=None, **kwargs):
        """Simulate process.

        Depending on the state of evaluate_stationarity, the process is
        simulated until termination criterion is reached.

        Parameters
        ----------
        process : Process
            Process to be simulated
        previous_results : SimulationResults
            Results of previous simulation run for initial conditions.

        Returns
        -------
        results : SimulationResults
            Results the final cycle of the simulation.

        Raises
        ------
        TypeError
            If process is not an instance of Process.

        See also
        --------
        simulate_n_cycles
        simulate_to_stationarity
        run
        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if not self.evaluate_stationarity:
            results = self.simulate_n_cycles(
                    process, self.n_cycles, previous_results, **kwargs)
        else:
            results = self.simulate_to_stationarity(
                    process, previous_results, **kwargs)

        return results

    @log_time('Simulation')
    @log_results('Simulation')
    @log_exceptions('Simulation')
    def simulate_n_cycles(self, process, n_cyc, previous_results=None, **kwargs):
        """Simulates process for given number of cycles.

        Parameters
        ----------
        process : Process
            Process to be simulated
        n_cyc : float
            Number of cycles
        previous_results : SimulationResults
            Results of previous simulation run.

        Returns
        -------
        results : SimulationResults
            Results the final cycle of the simulation.

        Raises
        ------
        TypeError
            If process is not an instance of Process.

        See also
        --------
        simulate_n_cycles
        simulate_to_stationarity
        StationarityEvaluator
        run
        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if previous_results is not None:
            self.set_state_from_results(process, previous_results)
        process._n_cycles = n_cyc

        return self.run(process, **kwargs)

    @log_time('Simulation')
    @log_results('Simulation')
    @log_exceptions('Simulation')
    def simulate_to_stationarity(self, process, previous_results=None, **kwargs):
        """Simulate process until stationarity is reached.

        Parameters
        ----------
        process : Process
            Process to be simulated
        previous_results : SimulationResults
            Results of previous simulation run.

        Returns
        -------
        results : SimulationResults
            Results the final cycle of the simulation.

        Raises
        ------
        TypeError
            If process is not an instance of Process.

        See also
        --------
        simulate
        run
        StationarityEvaluator
        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if previous_results is not None:
            self.set_state_from_results(process, previous_results)

        # Simulate minimum number of cycles
        n_cyc = self.n_cycles_min
        process._n_cycles = n_cyc
        results = self.run(process, **kwargs)
        process._n_cycles = 1

        # Simulate until stataionarity is reached.
        while True:
            n_cyc += 1
            self.set_state_from_results(process, results)

            results_cycle = self.run(process, **kwargs)

            if n_cyc >= self.n_cycles_max:
                self.logger.warning("Exceeded maximum number of cycles")
                break

            stationarity = False
            for chrom_old, chrom_new in zip(
                    results.chromatograms, results_cycle.chromatograms):
                stationarity = self.stationarity_evaluator.assert_stationarity(
                        chrom_old, chrom_new)

            results.update(results_cycle)

            if stationarity:
                break


        return results

    def set_state_from_results(self, process, results):
        process.system_state = results.system_solution['state']
        process.system_state_derivative = results.system_solution['state_derivative']
        return process


    @abstractmethod
    def run(process, **kwargs):
        """Abstract Method for running a simulation.

        Parameters
        ----------
        process : Process
            Process to be simulated.

        Returns
        -------
        results : SimulationResults
            Simulation results including process and solver configuration.

        Raises
        ------
        TypeError
            If process is not an instance of Process
        CADETProcessError
            If simulation doesn't terminate successfully
        """
        return

    @property
    def stationarity_evaluator(self):
        """Returns the stationarity evaluator.

        Returns
        ----------
        stationarity_evaluator : StationarityEvaluator
            Evaluator for cyclic stationarity.
        """
        return self._stationarity_evaluator

    @stationarity_evaluator.setter
    def stationarity_evaluator(self, stationarity_evaluator):
        if not isinstance(stationarity_evaluator, StationarityEvaluator):
            raise CADETProcessError('Expected StationarityEvaluator')
        self._stationarity_evaluator = stationarity_evaluator


class SimulationResults(metaclass=StructMeta):
    """Class for storing simulation results including the solver configuration

    Attributes
    ----------
    solver_name : str
        Name of the solver used to simulate the process
    solver_parameters : dict
        Dictionary with parameters used by the solver
    exit_flag : int
        Information about the solver termination.
    exit_message : str
        Additional information about the solver status
    time_elapsed : float
        Execution time of simulation.
    process_name : str
        Name of the simulated proces
    process_config : dict
        Configuration of the simulated process
    process_meta : dict
        Meta information of the process.
    solution : dict
        Time signals for all cycles of all Unit Operations.
    system_solution : dict
        Final state and state_derivative of the system.
    chromatograms : List of chromatogram
        Solution of the final cycle of the chromatogram_sinks.
    n_cycles : int
        Number of cycles that were simulated.

    Notes
    -----
    Ideally, the final state for each unit operation should be saved. However,
    CADET does currently provide this functionality.
    """
    solver_name = String()
    solver_parameters = Dict()
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    process_name = String()
    process_config = Dict()
    solution = Dict()
    system_solution = Dict()
    chromatograms = List()

    def __init__(self, solver_name, solver_parameters, exit_flag, exit_message,
                 time_elapsed, process_name, process_config, process_meta,
                 solution, system_solution, chromatograms):
        self.solver_name = solver_name
        self.solver_parameters = solver_parameters

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.process_name = process_name
        self.process_config = process_config
        self.process_meta = process_meta

        self.solution = solution
        self.system_solution = system_solution
        self.chromatograms = chromatograms

    def update(self, new_results):
        if self.process_name != new_results.process_name:
            raise CADETProcessError('Process does not match')

        self.exit_flag = new_results.exit_flag
        self.exit_message = new_results.exit_message
        self.time_elapsed += new_results.time_elapsed

        self.system_solution = new_results.system_solution

        self.chromatograms = new_results.chromatograms
        for unit in self.solution:
            self.solution[unit] += new_results.solution[unit]

    @property
    def solution_complete(self):
        time_cycle = next(iter(self.solution.values()))[0].time
        cycle_time = self.process_config['parameters']['cycle_time']

        time_complete = time_cycle
        for i in range(1, self.n_cycles):
            time_complete = np.hstack((time_complete, time_cycle[1:] + i*cycle_time))

        solution_complete = Dict()
        for unit, cycles in self.solution.items():
            unit_complete = cycles[0].signal
            for i in range(1, self.n_cycles):
                unit_complete = np.vstack((unit_complete, cycles[i].signal[1:]))
            solution_complete[unit] = TimeSignal(time_complete, unit_complete)

        return solution_complete

    @property
    def n_cycles(self):
        return len(next(iter(self.solution.values())))

    def save(self, case_dir, unit=None, start=0, end=None):
        path = os.path.join(settings.project_directory, case_dir)

        if unit is None:
            units = self.solution_complete.keys()
        else:
            units = self.solution_complete[unit]

        for unit in units:
            self.solution[unit][-1].plot(
                    save_path=path + '/' + unit + '_last.png',
                    start=start, end=end)

        for unit in units:
            self.solution_complete[unit].plot(
                    save_path=path + '/' + unit + '_complete.png',
                    start=start, end=end)

        for unit in units:
            self.solution[unit][-1].plot(
                    save_path=path + '/' + unit + '_overlay.png',
                    overlay = [cyc.signal for cyc in self.solution[unit][0:-1]],
                    start=start, end=end)


class ParametersGroup(metaclass=StructMeta):
    """Base class for grouping parameters and exporting them to a dict.
    
    Attributes
    ----------
    _parameters : List of strings
        List of paramters to be exported.
    
    See also
    --------
    Parameter
    Descriptor
    ParameterWrapper
    """
    _parameters = []

    def to_dict(self):
        """dict: Dictionary with names and values of the parameters.
        """
        return {param: getattr(self, param) for param in self._parameters
                if getattr(self, param) is not None}


class ParameterWrapper(ParametersGroup):
    """Base class for converting the config from objects such as units.

    Attributes
    ----------
    _base_class : type
        Type constraint for wrapped object
    _wrapped_object : obj
        Object whose config is to be converted

    Raises
    ------
    CADETProcessError
        If the wrapped_object is no instance of the base_class.
        
    See also
    --------
    Parameter
    Descriptor
    ParametersGroup
    """
    _base_class = object

    def __init__(self, wrapped_object):
        if not isinstance(wrapped_object, self._baseClass):
            raise CADETProcessError("Expected {}".format(self._baseClass))

        model = wrapped_object.__class__.__name__

        try:
            self._model = self._models[model]
        except KeyError:
            raise CADETProcessError("Model Type not defined")

        self._model_parameters = self._model_parameters[self._model]
        self._wrapped_object = wrapped_object

    def to_dict(self):
        """Returns the parameters for the model and solver in a dictionary.

        Defines the parameters for the model and solver and saves them into the
        respective dictionary. The cadet_parameters are get by the
        parameters_dict of the inherited functionality of the ParametersGroup.
        The keys for the model_solver_parameters are get by the attributes of
        the value of the wrapped_object, if they are not None. Both, the
        solver_parameters and the model_solver_parameters are saved into the
        parameters_dict.

        Returns
        -------
        parameters_dict : dict
            Dictionary, containing the attributes of each parameter from the
            model_parameters and the cadet_parameters.

        See also
        --------
        ParametersGroup
        """
        solver_parameters = super().to_dict()
        model_parameters = {key: getattr(self._wrapped_object, value)
            for key, value in self._model_parameters.items()
                if getattr(self._wrapped_object, value) is not None}
        return {**solver_parameters, **model_parameters}
