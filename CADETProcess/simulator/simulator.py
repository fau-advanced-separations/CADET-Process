from abc import abstractmethod
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.log import get_logger, log_time, log_results, log_exceptions
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Bool, UnsignedFloat, UnsignedInteger
from CADETProcess.processModel import Process
from CADETProcess.stationarity import StationarityEvaluator, RelativeArea, NRMSE


class SimulatorBase(metaclass=StructMeta):
    """BaseClass for Solver APIs.

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the process
    configuration into the APIs configuration format and convert the results
    back to the CADETProcess format.

    Attributes
    ----------
    time_resolution : float
        Time interval for user solution times. Default is 1 s.
    resolution_cutoff : float
        To avoid IDAS errors, user solution times are removed if they are
        closer to section times than the cutoff value. Default is 1e-3 s.
    n_cycles : int
        Number of cycles to be simulated
    n_cycles_min : int
        If simulate_to_stationarity: Minimum number of cycles to be simulated.
    n_cycles_max : int
        If simulate_to_stationarity: Maximum number of cycles to be simulated.
    simulate_to_stationarity : bool
        Simulate until stationarity is reached

    See Also
    --------
    Process
    StationarityEvaluator

    """
    time_resolution = UnsignedFloat(default=1)
    resolution_cutoff = UnsignedFloat(default=1e-3)

    n_cycles = UnsignedInteger(default=1)
    evaluate_stationarity = Bool(default=False)
    n_cycles_min = UnsignedInteger(default=5)
    n_cycles_max = UnsignedInteger(default=100)

    def __init__(self, stationarity_evaluator=None):
        self.logger = get_logger('Simulation')

        if stationarity_evaluator is None:
            self.stationarity_evaluator = StationarityEvaluator()
            self.stationarity_evaluator.add_criterion(RelativeArea())
            self.stationarity_evaluator.add_criterion(NRMSE())
        else:
            self.stationarity_evaluator = stationarity_evaluator
            self.evaluate_stationarity = True

    @property
    def sig_fig(self):
        return int(-np.log10(self.resolution_cutoff))

    def get_solution_time(self, process, cycle=1):
        """np.array: Time vector for one cycle.

        See Also
        --------
        Process.section_times
        get_solution_time_complete

        """
        solution_times = np.arange((
            cycle-1)*process.cycle_time,
            cycle*process.cycle_time,
            self.time_resolution
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
                indices.append(d+1)
            else:
                indices.append(d)

        solution_times = np.delete(solution_times, indices)

        return solution_times

    def get_solution_time_complete(self, process):
        """np.array: time vector for mulitple cycles of a process.

        See Also
        --------
        n_cycles
        get_section_times_complete
        get_solution_time

        """
        time = self.get_solution_time(process)
        solution_times = np.array([])
        for i in range(self.n_cycles):
            solution_times = np.append(
                solution_times, (i)*time[-1] + time
            )
        solution_times = np.round(solution_times, self.sig_fig)

        return solution_times.tolist()

    def get_section_times(self, process):
        """list: Section times for single cycle of a process.

        Includes 0 and cycle_time if they do not coincide with event time.

        See Also
        --------
        get_section_times_complete
        get_solution_time_complete
        Process.section_times

        """
        section_times = np.array(process.section_times)
        section_times = np.round(section_times, self.sig_fig)

        return section_times.tolist()

    def get_section_times_complete(self, process):
        """list: Section times for multiple cycles of a process.

        Includes 0 and cycle_time if they do not coincide with event time.

        See Also
        --------
        get_section_times
        n_cycles
        get_solution_time_complete
        Process.section_times

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

        See Also
        --------
        simulate_n_cycles
        simulate_to_stationarity
        run

        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if not process.check_config():
            raise CADETProcessError("Process is not configured correctly.")

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
    def simulate_n_cycles(
            self, process, n_cyc, previous_results=None, **kwargs):
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

        See Also
        --------
        simulate_n_cycles
        simulate_to_stationarity
        StationarityEvaluator
        run

        """
        n_cyc_orig = self.n_cycles
        self.n_cycles = n_cyc

        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if previous_results is not None:
            self.set_state_from_results(process, previous_results)

        return self.run(process, **kwargs)

        self.n_cycles = n_cyc_orig

    @log_time('Simulation')
    @log_results('Simulation')
    @log_exceptions('Simulation')
    def simulate_to_stationarity(
            self, process, results=None, **kwargs):
        """Simulate process until stationarity is reached.

        Parameters
        ----------
        process : Process
            Process to be simulated
        results : SimulationResults
            Results of previous simulation run.

        Returns
        -------
        results : SimulationResults
            Results the final cycle of the simulation.

        Raises
        ------
        TypeError
            If process is not an instance of Process.

        See Also
        --------
        simulate
        run
        StationarityEvaluator

        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        n_cyc_orig = self.n_cycles
        self.n_cycles = self.n_cycles_min

        if results is not None:
            n_cyc = results.n_cycles
        else:
            n_cyc = 0

        while True:
            n_cyc += self.n_cycles_min

            if results is not None:
                self.set_state_from_results(process, results)

            new_results = self.run(process, **kwargs)

            if results is None:
                results = new_results
            else:
                results.update(new_results)

            if n_cyc == 1:
                continue

            stationarity = self.stationarity_evaluator.assert_stationarity(
                results
            )

            if stationarity:
                break

            if n_cyc >= self.n_cycles_max:
                self.logger.warning("Exceeded maximum number of cycles")
                break

        self.n_cycles = n_cyc_orig

        return results

    def set_state_from_results(self, process, results):
        process.system_state = results.system_state['state']
        process.system_state_derivative = \
            results.system_state['state_derivative']

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

    evaluate = simulate
    __call__ = simulate

    @property
    def stationarity_evaluator(self):
        """Returns the stationarity evaluator.

        Returns
        -------
        stationarity_evaluator : StationarityEvaluator
            Evaluator for cyclic stationarity.

        """
        return self._stationarity_evaluator

    @stationarity_evaluator.setter
    def stationarity_evaluator(self, stationarity_evaluator):
        if not isinstance(stationarity_evaluator, StationarityEvaluator):
            raise CADETProcessError('Expected StationarityEvaluator')
        self._stationarity_evaluator = stationarity_evaluator
