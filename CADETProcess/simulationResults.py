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

        def _merge_cycles(
            cycles: list[SolutionBase],
            time_complete: np.ndarray,
        ) -> SolutionBase:
            """
            Merge cycle solutions into a complete solution.

            Parameters
            ----------
            cycles : list
                List of cycle objects.
            time_complete : np.ndarray
                Complete time array.

            Returns
            -------
            object
                A new object with merged solution and time.
            """
            merged = copy.deepcopy(cycles[0])
            merged.time = time_complete

            solution_complete = cycles[0].solution
            if solution_complete.ndim > 1:
                for cycle in cycles[1:]:
                    solution_complete = np.vstack(
                        (solution_complete, cycle.solution[1:])
                    )
            else:
                for cycle in cycles[1:]:
                    solution_complete = np.hstack(
                        (solution_complete, cycle.solution[1:])
                    )
            merged.solution = solution_complete

            if hasattr(cycles[0], "flow_rate"):
                flow_rate_complete = copy.deepcopy(cycles[0].flow_rate)
                for cycle in cycles[1:]:
                    new_flow_rate = cycle.flow_rate.offset(flow_rate_complete.end)
                    for section in new_flow_rate.sections:
                        # Find the closest time to avoid numerical issues
                        start_index = np.argmin(np.abs(time_complete - section.start))
                        section.start = time_complete[start_index]
                        end_index = np.argmin(np.abs(time_complete - section.end))
                        section.end = time_complete[end_index]
                        flow_rate_complete.add_section(section)
                merged.flow_rate = flow_rate_complete

            merged.update_solution()

            return merged

        solution = Dict()
        time_complete = self.time_complete
        for unit, solutions in self.solution_cycles.items():
            for sol, ports_cycles in solutions.items():
                if isinstance(ports_cycles, dict):
                    for port, cycles in ports_cycles.items():
                        solution[unit][sol][port] = _merge_cycles(
                            cycles,
                            time_complete,
                        )
                else:
                    solution[unit][sol] = _merge_cycles(
                        ports_cycles,
                        time_complete,
                    )

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
