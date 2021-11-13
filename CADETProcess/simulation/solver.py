from abc import abstractmethod
import copy
import os

import numpy as np
import addict

from CADETProcess import CADETProcessError
from CADETProcess.common import settings
from CADETProcess.log import get_logger, log_time, log_results, log_exceptions
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    Bool, Dict, String, List, UnsignedInteger, UnsignedFloat
)
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
        self.logger = get_logger('Simulation')

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

        process.lock = True
        if not self.evaluate_stationarity:
            results = self.simulate_n_cycles(
                process, self.n_cycles, previous_results, **kwargs
            )
        else:
            results = self.simulate_to_stationarity(
                process, previous_results, **kwargs
            )
        process.lock = False

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
                results.chromatograms, results_cycle.chromatograms
            ):
                stationarity = self.stationarity_evaluator.assert_stationarity(
                    chrom_old, chrom_new
                )

            results.update(results_cycle)

            if stationarity:
                break


        return results

    def set_state_from_results(self, process, results):
        process.system_state = results.system_state['state']
        process.system_state_derivative = results.system_state['state_derivative']
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
    system_state : dict
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
    solution_cycles = Dict()
    system_state = Dict()
    chromatograms = List()

    def __init__(
            self, solver_name, solver_parameters, exit_flag, exit_message,
            time_elapsed, process_name, process_config, process_meta,
            solution_cycles, system_state, chromatograms
            ):
        self.solver_name = solver_name
        self.solver_parameters = solver_parameters

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.process_name = process_name
        self.process_config = process_config
        self.process_meta = process_meta

        self.solution_cycles = solution_cycles
        self.system_state = system_state
        self.chromatograms = chromatograms

    def update(self, new_results):
        if self.process_name != new_results.process_name:
            raise CADETProcessError('Process does not match')

        self.exit_flag = new_results.exit_flag
        self.exit_message = new_results.exit_message
        self.time_elapsed += new_results.time_elapsed

        self.system_state = new_results.system_state

        self.chromatograms = new_results.chromatograms
        for unit, solutions in self.solution_cycles.items():
            for sol in solutions:
                self.solution_cycles[unit][sol].append(new_results.solution[unit][sol])

    @property
    def solution(self):
        """Construct complete solution from individual cyles.
        """
        cycle_time = self.process_config['parameters']['cycle_time']

        time_complete = self.time_cycle
        for i in range(1, self.n_cycles):
            time_complete = np.hstack((
                time_complete, 
                self.time_cycle[1:] + i*cycle_time
            ))

        solution = addict.Dict()
        for unit, solutions in self.solution_cycles.items():
            for sol, cycles in solutions.items():
                solution[unit][sol] = copy.deepcopy(cycles[0])
                solution_complete = cycles[0].solution
                for i in range(1, self.n_cycles):
                    solution_complete = np.vstack((
                        solution_complete, cycles[i].solution[1:]
                    ))
                solution[unit][sol].time = time_complete
                solution[unit][sol].solution = solution_complete

        return solution

    @property
    def n_cycles(self):
        return len(self.solution_cycles[self._first_unit][self._first_solution])
    
    @property
    def _first_unit(self):
        return next(iter(self.solution_cycles))
    
    @property
    def _first_solution(self):
        return next(iter(self.solution_cycles[self._first_unit]))
    
    @property
    def time_cycle(self):
        """np.array: Solution times vector
        """
        return \
            self.solution_cycles[self._first_unit][self._first_solution][0].time
        
    def save(self, case_dir, unit=None, start=0, end=None):
        path = os.path.join(settings.project_directory, case_dir)

        if unit is None:
            units = self.solution.keys()
        else:
            units = self.solution[unit]

        for unit in units:
            self.solution[unit][-1].plot(
                save_path=path + '/' + unit + '_last.png',
                start=start, end=end
            )

        for unit in units:
            self.solution_complete[unit].plot(
                save_path=path + '/' + unit + '_complete.png',
                start=start, end=end
            )

        for unit in units:
            self.solution[unit][-1].plot(
                save_path=path + '/' + unit + '_overlay.png',
                overlay = [cyc.signal for cyc in self.solution[unit][0:-1]],
                start=start, end=end
            )
