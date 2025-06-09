from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from CADETProcess import CADETProcessError, SimulationResults
from CADETProcess.dataStructure import Bool, Structure, UnsignedFloat, UnsignedInteger
from CADETProcess.log import get_logger, log_exceptions, log_results, log_time
from CADETProcess.processModel import Process
from CADETProcess.stationarity import NRMSE, RelativeArea, StationarityEvaluator


class SimulatorBase(Structure):
    """
    Base class for Solver APIs.

    Holds the configuration of the individual solvers and provides an interface
    for calling the run method. The class converts the process configuration
    into the API's configuration format and converts the results back to the
    CADETProcess format.

    Attributes
    ----------
    time_resolution : float
        Time interval for user solution times. The default value is 1 second.
    resolution_cutoff : float
        Solution times closer to section times than the cutoff value are removed
        to avoid IDAS errors. The default value is 1e-3 seconds.
    n_cycles : int
        Number of cycles to be simulated. The default is 1.
    evaluate_stationarity : bool
        If True, simulate until stationarity is reached.
        The default is False
    n_cycles_min : int
        Minimum number of cycles to be simulated if evaluate_stationarity is True.
        The default is 5.
    n_cycles_max : int
        Maximum number of cycles to be simulated if evaluate_stationarity is True.
        The default is 100.
    raise_exception_on_max_cycles : bool
        Raise an exception when the maximum number of cycles is exceeded.
        The default is False

    See Also
    --------
    CADETProcess.processModel.Process
    CADETProcess.stationarity.StationarityEvaluator
    """

    time_resolution = UnsignedFloat(default=1)
    resolution_cutoff = UnsignedFloat(default=1e-3)

    n_cycles = UnsignedInteger(default=1)
    evaluate_stationarity = Bool(default=False)
    n_cycles_min = UnsignedInteger(default=1)
    n_cycles_batch = UnsignedInteger(default=5)
    n_cycles_max = UnsignedInteger(default=100)
    raise_exception_on_max_cycles = Bool(default=False)

    def __init__(
        self,
        stationarity_evaluator: Optional[StationarityEvaluator] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize Simulator Base."""
        super().__init__(*args, **kwargs)

        self.logger = get_logger("Simulation")

        if stationarity_evaluator is None:
            self.stationarity_evaluator = StationarityEvaluator()
            self.stationarity_evaluator.add_criterion(RelativeArea())
            self.stationarity_evaluator.add_criterion(NRMSE())
        else:
            self.stationarity_evaluator = stationarity_evaluator
            self.evaluate_stationarity = True

    @property
    def sig_fig(self) -> int:
        """int: Number of significant figures based on resolution_cutoff."""
        return int(-np.log10(self.resolution_cutoff))

    def get_solution_time(self, process: Process, cycle: int = 1) -> np.ndarray:
        """
        Get the time vector for one cycle of a process.

        Parameters
        ----------
        process : Process
            The process to simulate.
        cycle : int, optional
            The cycle number, by default 1.

        Returns
        -------
        np.ndarray
            Time vector for one cycle.

        See Also
        --------
        CADETProcess.processModel.Process.section_times
        get_solution_time_complete
        """
        solution_times = np.arange(
            (cycle - 1) * process.cycle_time,
            cycle * process.cycle_time,
            self.time_resolution,
        )
        section_times = self.get_section_times(process)
        solution_times = np.append(solution_times, section_times)
        solution_times = np.round(solution_times, self.sig_fig)
        solution_times = np.sort(solution_times)
        solution_times = np.unique(solution_times)

        diff = np.where(np.diff(solution_times) < self.resolution_cutoff)[0]
        indices = []
        for d in diff:
            if solution_times[d] in process.section_times:
                indices.append(d + 1)
            else:
                indices.append(d)

        solution_times = np.delete(solution_times, indices)

        return solution_times

    def get_solution_time_complete(self, process: Process) -> np.ndarray:
        """
        Get the time vector for multiple cycles of a process.

        Parameters
        ----------
        process : Process
            The process to simulate.

        Returns
        -------
        np.ndarray
            Time vector for multiple cycles of a process.

        See Also
        --------
        n_cycles
        get_section_times_complete
        get_solution_time
        """
        time = self.get_solution_time(process)

        solution_times = time

        for i in range(1, self.n_cycles):
            solution_times = np.append(solution_times, (i) * time[-1] + time[1:])
        solution_times = np.round(solution_times, self.sig_fig)

        return solution_times.tolist()

    def get_section_times(self, process: Process) -> list:
        """
        Get the section times for a single cycle of a process.

        Parameters
        ----------
        process : Process
            The process to simulate.

        Returns
        -------
        list
            Section times for a single cycle of a process.

        See Also
        --------
        get_section_times_complete
        get_solution_time_complete
        CADETProcess.processModel.Process.section_times
        """
        section_times = np.array(process.section_times)
        section_times = np.round(section_times, self.sig_fig)

        return section_times.tolist()

    def get_section_times_complete(self, process: Process) -> list:
        """
        Get the section times for multiple cycles of a process.

        Parameters
        ----------
        process : Process
            The process to simulate.

        Returns
        -------
        list
            Section times for multiple cycles of a process.

        See Also
        --------
        get_section_times
        n_cycles
        get_solution_time_complete
        CADETProcess.processModel.Process.section_times
        """
        sections = np.array(self.get_section_times(process))
        cycle_time = sections[-1]

        section_times_complete = []
        for cycle in range(self.n_cycles):
            section_cycle = cycle * cycle_time + sections[0:-1]
            section_times_complete += section_cycle.tolist()

        section_times_complete.append(self.n_cycles * sections[-1])

        section_times_complete = np.round(section_times_complete, self.sig_fig)

        return section_times_complete.tolist()

    def simulate(
        self,
        process: Process,
        previous_results: Optional[SimulationResults] = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """
        Simulate the process.

        Depending on the state of evaluate_stationarity, the process is
        simulated until the termination criterion is reached.

        Parameters
        ----------
        process : Process
            The process to be simulated.
        previous_results : SimulationResults, optional
            Results of the previous simulation run for initial conditions.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        results : SimulationResults
            Results of the final cycle of the simulation.

        Raises
        ------
        TypeError
            If the process is not an instance of Process.
        CADETProcessError
            If the process is not configured correctly.

        See Also
        --------
        simulate_n_cycles
        simulate_to_stationarity
        run
        """
        if not isinstance(process, Process):
            raise TypeError("Expected Process")

        process.lock = True
        if not process.check_config():
            raise CADETProcessError("Process is not configured correctly.")

        if self.n_cycles_max < self.n_cycles_min:
            raise ValueError(
                "n_cycles_max is set lower than n_cycles_min "
                f"({self.n_cycles_max} vs {self.n_cycles_min}). "
            )

        # If "max" is below "batch", reduce "batch" to "max" to only run "max" cycles.
        if self.n_cycles_max < self.n_cycles_batch:
            self.n_cycles_batch = self.n_cycles_max

        if not self.evaluate_stationarity:
            results = self.simulate_n_cycles(
                process, self.n_cycles, previous_results, **kwargs
            )
        else:
            results = self.simulate_to_stationarity(process, previous_results, **kwargs)
        process.lock = False

        return results

    @log_time("Simulation")
    @log_results("Simulation")
    @log_exceptions("Simulation")
    def simulate_n_cycles(
        self,
        process: Process,
        n_cyc: int,
        previous_results: Optional[SimulationResults] = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """
        Simulate the process for a given number of cycles.

        Parameters
        ----------
        process : Process
            The process to be simulated.
        n_cyc : int
            The number of cycles to simulate.
        previous_results : SimulationResults, optional
            Results of the previous simulation run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        results : SimulationResults
            Results of the final cycle of the simulation.

        Raises
        ------
        TypeError
            If the process is not an instance of Process.

        See Also
        --------
        simulate
        simulate_to_stationarity
        StationarityEvaluator
        run
        """
        n_cyc_orig = self.n_cycles
        self.n_cycles = n_cyc

        if not isinstance(process, Process):
            raise TypeError("Expected Process")

        if previous_results is not None:
            self.set_state_from_results(process, previous_results)

        return self._run(process, **kwargs)

        self.n_cycles = n_cyc_orig

    @log_time("Simulation")
    @log_results("Simulation")
    @log_exceptions("Simulation")
    def simulate_to_stationarity(
        self,
        process: Process,
        results: Optional[SimulationResults] = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """
        Simulate the process until stationarity is reached.

        Parameters
        ----------
        process : Process
            The process to be simulated.
        results : SimulationResults, optional
            Results of the previous simulation run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        results : SimulationResults
            Results of the final cycle of the simulation.

        Raises
        ------
        TypeError
            If the process is not an instance of Process.
        CADETProcessError
            If the simulation doesn't terminate successfully and
            `raise_exception_on_max_cycles` is True.

        See Also
        --------
        simulate
        run
        StationarityEvaluator
        """
        if not isinstance(process, Process):
            raise TypeError("Expected Process")

        n_cyc_orig = self.n_cycles
        self.n_cycles = max(self.n_cycles_min, self.n_cycles_batch)

        if results is not None:
            n_cyc = results.n_cycles
        else:
            n_cyc = 0

        while True:
            n_cyc += self.n_cycles_batch

            if results is not None:
                self.set_state_from_results(process, results)

            new_results = self._run(process, **kwargs)

            if results is None:
                results = new_results
            else:
                results.update(new_results)

            if n_cyc == 1:
                continue

            stationarity = self.stationarity_evaluator.assert_stationarity(results)

            if stationarity:
                break

            if n_cyc >= self.n_cycles_max:
                msg = "Exceeded maximum number of cycles."
                if self.raise_exception_on_max_cycles:
                    raise CADETProcessError(msg)
                else:
                    self.logger.warning(msg)
                break

        self.n_cycles = n_cyc_orig

        return results

    def set_state_from_results(
        self,
        process: Process,
        results: SimulationResults,
    ) -> Process:
        """
        Set the process state from the simulation results.

        Parameters
        ----------
        process : Process
            The process to set the state for.
        results : SimulationResults
            The simulation results containing the state information.

        Returns
        -------
        Process
            The process with the updated state.
        """
        process.system_state = results.system_state["state"]
        process.system_state_derivative = results.system_state["state_derivative"]

        return process

    @abstractmethod
    def _run(process: Process, **kwargs: Any) -> SimulationResults:
        """
        Abstract method for running a simulation.

        Parameters
        ----------
        process : Process
            The process to be simulated.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        SimulationResults
            The simulation results including process and solver configuration.

        Raises
        ------
        TypeError
            If the process is not an instance of Process.
        CADETProcessError
            If the simulation doesn't terminate successfully.
        """
        return

    evaluate = simulate
    __call__ = simulate

    @property
    def stationarity_evaluator(self) -> StationarityEvaluator:
        """StationarityEvaluator: The stationarity evaluator."""
        return self._stationarity_evaluator

    @stationarity_evaluator.setter
    def stationarity_evaluator(
        self,
        stationarity_evaluator: StationarityEvaluator,
    ) -> None:
        """
        Set the stationarity evaluator.

        Parameters
        ----------
        stationarity_evaluator : StationarityEvaluator
            The stationarity evaluator to set.

        Raises
        ------
        CADETProcessError
            If the stationarity evaluator is not an instance of StationarityEvaluator.
        """
        if not isinstance(stationarity_evaluator, StationarityEvaluator):
            raise CADETProcessError("Expected StationarityEvaluator")
        self._stationarity_evaluator = stationarity_evaluator
