from abc import abstractmethod

from CADETProcess import CADETProcessError
from CADETProcess.log import get_logger, log_time, log_results, log_exceptions
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Bool, UnsignedInteger
from CADETProcess.processModel import Process
from CADETProcess.stationarity import StationarityEvaluator


class SimulatorBase(metaclass=StructMeta):
    """BaseClass for Solver APIs.

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the process
    configuration into the APIs configuration format and convert the results
    back to the CADETProcess format.

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

    See Also
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
            self.stationarity_evaluator = StationarityEvaluator()
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

        See Also
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
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        if previous_results is not None:
            self.set_state_from_results(process, previous_results)
        process._n_cycles = n_cyc

        return self.run(process, **kwargs)

    @log_time('Simulation')
    @log_results('Simulation')
    @log_exceptions('Simulation')
    def simulate_to_stationarity(
            self, process, previous_results=None, **kwargs):
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

        See Also
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
