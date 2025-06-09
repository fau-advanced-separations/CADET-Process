"""
==========================================================
Simulation Results (:mod:`CADETProcess.simulationResults`)
==========================================================

.. currentmodule:: CADETProcess.simulationResults

This module provides a class for storing simulation results.


.. autosummary::
    :toctree: generated/

    SimulationResults

"""  # noqa

from __future__ import annotations

import copy

import numpy as np
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Dictionary,
    List,
    String,
    Structure,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.processModel import ComponentSystem, Process, UnitBaseClass
from CADETProcess.solution import SolutionBase

__all__ = ["SimulationResults"]


class SimulationResults(Structure):
    """
    Class for storing simulation results including the solver configuration.

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
        Solution objects for all cycles of all Unit Operations.
    solution_cycles : dict
        Solution objects  for individual cycles of all Unit Operations.
    sensitivity : dict
        Solution objects for all sensitivities of all cycles of all Unit Operations.
    sensitivity_cycles : dict
        Solution objects for all sensitivities of individual cycles of all Unit Operations.
    system_state : dict
        Final state and state_derivative of the system.
    chromatograms : List of chromatogram
        Solution of the final cycle of the product outlets.
    n_cycles : int
        Number of cycles that were simulated.

    Notes
    -----
        Ideally, the final state for each unit operation should be saved.
        However, CADET does currently provide this functionality.
    """

    solver_name = String()
    solver_parameters = Dictionary()
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    solution_cycles = Dictionary()
    sensitivity_cycles = Dictionary()
    system_state = Dictionary()
    chromatograms = List()

    def __init__(
        self,
        solver_name: str,
        solver_parameters: dict,
        exit_flag: int,
        exit_message: str,
        time_elapsed: float,
        process: Process,
        solution_cycles: dict,
        sensitivity_cycles: dict,
        system_state: dict,
        chromatograms: list,
    ) -> None:
        """Initialize SimulationResults."""
        self.solver_name = solver_name
        self.solver_parameters = solver_parameters

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.process = process

        self.solution_cycles = solution_cycles
        self.sensitivity_cycles = sensitivity_cycles
        self.system_state = system_state
        self.chromatograms = chromatograms

        self._time_complete = None
        self._solution = None
        self._sensitivity = None

    def update(self, new_results: SimulationResults) -> None:
        """Update the simulation results with results from a new cycle."""
        if self.process.name != new_results.process.name:
            raise CADETProcessError("Process does not match")

        self.exit_flag = new_results.exit_flag
        self.exit_message = new_results.exit_message
        self.time_elapsed += new_results.time_elapsed

        self.system_state = new_results.system_state

        self.chromatograms = new_results.chromatograms
        for unit, solutions in self.solution_cycles.items():
            for sol in solutions:
                solution = new_results.solution_cycles[unit][sol]
                self.solution_cycles[unit][sol] += solution

        self._time_complete = None
        self._solution = None
        self._sensitivity = None

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: The component system used in the simulation."""
        solution = self.solution_cycles[self._first_unit][self._first_solution]
        return solution[0].component_system

    @property
    def solution(self) -> Dict:
        """Construct complete solution from individual cyles."""
        if self._solution is not None:
            return self._solution

        time_complete = self.time_complete

        solution = Dict()
        for unit, solutions in self.solution_cycles.items():
            for sol, ports_cycles in solutions.items():
                if isinstance(ports_cycles, Dict):
                    ports = ports_cycles
                    for port, cycles in ports.items():
                        solution[unit][sol][port] = copy.deepcopy(cycles[0])
                        solution_complete = cycles[0].solution
                        if solution_complete.ndim > 1:
                            for i in range(1, self.n_cycles):
                                solution_complete = np.vstack((
                                    solution_complete, cycles[i].solution[1:]
                                ))
                        else:
                            for i in range(1, self.n_cycles):
                                solution_complete = np.hstack((
                                    solution_complete, cycles[i].solution[1:]
                                ))

                        solution[unit][sol][port].time = time_complete
                        solution[unit][sol][port].solution = solution_complete
                        solution[unit][sol][port].update_solution()
                else:
                    cycles = ports_cycles
                    solution[unit][sol] = copy.deepcopy(cycles[0])
                    solution_complete = cycles[0].solution
                    if solution_complete.ndim > 1:
                        for i in range(1, self.n_cycles):
                            solution_complete = np.vstack((
                                solution_complete, cycles[i].solution[1:]
                            ))
                    else:
                        for i in range(1, self.n_cycles):
                            solution_complete = np.hstack((
                                solution_complete, cycles[i].solution[1:]
                            ))

                    solution[unit][sol].time = time_complete
                    solution[unit][sol].solution = solution_complete
                    solution[unit][sol].update_solution()

        self._solution = solution

        return solution

    @property
    def sensitivity(self) -> Dict:
        """Construct complete sensitivity from individual cyles."""
        if self._sensitivity is not None:
            return self._sensitivity

        time_complete = self.time_complete

        sensitivity = Dict()
        for sens_name, sensitivities in self.sensitivity_cycles.items():
            for unit, sensitivities in sensitivities.items():
                for flow, ports_cycles in sensitivities.items():
                    if isinstance(ports_cycles, Dict):
                        ports = ports_cycles
                        for port, cycles in ports.items():
                            sensitivity[sens_name][unit][flow][port] = copy.deepcopy(
                                cycles[0]
                            )
                            sensitivity_complete = cycles[0].solution
                            for i in range(1, self.n_cycles):
                                sensitivity_complete = np.vstack((
                                    sensitivity_complete, cycles[i].solution[1:]
                                ))
                            sensitivity[sens_name][unit][flow][port].time = time_complete
                            sensitivity[sens_name][unit][flow][port].solution = sensitivity_complete
                            sensitivity[sens_name][unit][flow][port].update_solution()

                    else:
                        cycles = ports_cycles
                        sensitivity[sens_name][unit][flow] = copy.deepcopy(cycles[0])
                        sensitivity_complete = cycles[0].solution
                        for i in range(1, self.n_cycles):
                            sensitivity_complete = np.vstack((
                                sensitivity_complete, cycles[i].solution[1:]
                            ))
                        sensitivity[sens_name][unit][flow].time = time_complete
                        sensitivity[sens_name][unit][flow].solution = sensitivity_complete
                        sensitivity[sens_name][unit][flow].update_solution()

        self._sensitivity = sensitivity

        return sensitivity

    @property
    def n_cycles(self) -> int:
        """int: Number of simulated cycles."""
        return len(self.solution_cycles[self._first_unit][self._first_solution])

    @property
    def _first_unit(self) -> UnitBaseClass:
        return next(iter(self.solution_cycles))

    @property
    def _first_solution(self) -> SolutionBase:
        return next(iter(self.solution_cycles[self._first_unit]))

    @property
    def time_cycle(self) -> np.ndarray:
        """np.array: Solution times vector."""
        return self.solution_cycles[self._first_unit][self._first_solution][0].time

    @property
    def time_complete(self) -> np.ndarray:
        """np.ndarray: Solution times vector for all cycles."""
        if self._time_complete is not None:
            return self._time_complete

        time_complete = self.time_cycle
        for i in range(1, self.n_cycles):
            time_complete = np.hstack((
                time_complete, self.time_cycle[1:] + i * self.process.cycle_time
            ))

        self._time_complete = time_complete

        return time_complete
