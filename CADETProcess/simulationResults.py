import copy
import os

import numpy as np
import addict

from CADETProcess import CADETProcessError
from CADETProcess import settings
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    Dict, String, List, UnsignedInteger, UnsignedFloat
)


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
    process: Process
        Simulated Process.
    solution : dict
        Solution objects  for all cycles of all Unit Operations.
    solution_cycles : dict
        Solution objects  for individual cycles of all Unit Operations.
    system_state : dict
        Final state and state_derivative of the system.
    chromatograms : List of chromatogram
        Solution of the final cycle of the chromatogram_sinks.
    n_cycles : int
        Number of cycles that were simulated.

    Notes
    -----
        Ideally, the final state for each unit operation should be saved.
        However, CADET does currently provide this functionality.

    """
    solver_name = String()
    solver_parameters = Dict()
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    solution_cycles = Dict()
    system_state = Dict()
    chromatograms = List()

    def __init__(
            self,
            solver_name, solver_parameters,
            exit_flag, exit_message, time_elapsed,
            process,
            solution_cycles, system_state,
            chromatograms
            ):
        self.solver_name = solver_name
        self.solver_parameters = solver_parameters

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.process = process

        self.solution_cycles = solution_cycles
        self.system_state = system_state
        self.chromatograms = chromatograms

        self._solution = None

    def update(self, new_results):
        if self.process.name != new_results.process.name:
            raise CADETProcessError('Process does not match')

        self.exit_flag = new_results.exit_flag
        self.exit_message = new_results.exit_message
        self.time_elapsed += new_results.time_elapsed

        self.system_state = new_results.system_state

        self.chromatograms = new_results.chromatograms
        for unit, solutions in self.solution_cycles.items():
            for sol in solutions:
                solution = new_results.solution_cycles[unit][sol]
                self.solution_cycles[unit][sol] += solution

        self._solution = None

    @property
    def component_system(self):
        solution = self.solution_cycles[self._first_unit][self._first_solution]
        return solution[0].component_system

    @property
    def solution(self):
        """Construct complete solution from individual cyles."""
        if self._solution is not None:
            return self._solution

        time_complete = self.time_cycle
        for i in range(1, self.n_cycles):
            time_complete = np.hstack((
                time_complete,
                self.time_cycle[1:] + i*self.process.cycle_time
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

        self._solution = solution

        return solution

    @property
    def n_cycles(self):
        return len(
            self.solution_cycles[self._first_unit][self._first_solution]
        )

    @property
    def _first_unit(self):
        return next(iter(self.solution_cycles))

    @property
    def _first_solution(self):
        return next(iter(self.solution_cycles[self._first_unit]))

    @property
    def time_cycle(self):
        """np.array: Solution times vector"""
        return self.solution_cycles[self._first_unit][self._first_solution][0].time

    def save(self, case_dir=None, unit=None, start=0, end=None):
        path = settings.working_directory
        if case_dir is not None:
            path = os.path.join(settings.working_directory, case_dir)

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
                overlay=[cyc.signal for cyc in self.solution[unit][0:-1]],
                start=start, end=end
            )
