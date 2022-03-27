from abc import abstractmethod
import json
import os

import CADETProcess
from CADETProcess import settings
from CADETProcess.log import get_logger, log_time, log_results, log_exceptions
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    List, NdArray, String, UnsignedInteger, UnsignedFloat
)
from CADETProcess.optimization import OptimizationProblem


class OptimizerBase(metaclass=StructMeta):
    """BaseClass for optimization solver APIs

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the
    OptimizationProblem configuration to the APIs configuration format and
    convert the results back to the CADET-Process format.

    """
    _options = []

    def __init__(self):
        self.logger = get_logger(str(self))

    def optimize(
            self, optimization_problem,
            working_directory='./', save_results=False,
            *args, **kwargs):
        """
        """
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')

        self.working_directory = os.path.abspath(working_directory)

        if save_results:
            self.logger = get_logger(
                str(self), log_directory=self.working_directory
            )

        log_time('Optimization', self.logger.level)(self.run)
        log_results('Optimization', self.logger.level)(self.run)
        log_exceptions('Optimization', self.logger.level)(self.run)

        results = self.run(optimization_problem, *args, **kwargs)

        if save_results:
            results.save(working_directory)

        return results

    @abstractmethod
    def run(optimization_problem, *args, **kwargs):
        """Abstract Method for solving an optimizatio problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Optimization problem to be solved.

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        Raises
        ------
        TypeError
            If optimization_problem is not an instance of OptimizationProblem
        CADETProcessError
            If solver doesn't terminate successfully

        """
        return

    @property
    def options(self):
        return {opt: getattr(self, opt) for opt in self._options}

    def __str__(self):
        return self.__class__.__name__


class OptimizationResults(metaclass=StructMeta):
    """Optimization results including the solver configuration.

    Attributes
    ----------
    optimization_problem : OptimizationProblem
        Optimization problem.
    optimizer : str
        Name of the Optimizer used to optimize the OptimizationProblem.
    optimizer_options : dict
        Dictionary with Optimizer options.
    exit_flag : int
        Information about the solver termination.
    exit_message : str
        Additional information about the solver status.
    time_elapsed : float
        Execution time of simulation.
    x : list
        Values of optimization variables at optimum.
    f : np.ndarray
        Value of objective function at x.
    g : np.ndarray
        Values of constraint function at x
    history : dict
        History of evaluations.

    """
    x0 = List()
    solver_name = String()
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    x = List()
    f = NdArray()
    g = NdArray()

    def __init__(
            self, optimization_problem,
            optimizer, optimizer_options, exit_flag, exit_message,
            time_elapsed, x, f, g, history=None):

        self.optimization_problem = optimization_problem

        self.optimizer = optimizer
        self.optimizer_options = optimizer_options

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.x = x
        self.f = f
        self.g = g

        self.history = history

        self.version = str(CADETProcess.__version__)

    def to_dict(self):
        return {
            'Optimization problem': self.optimization_problem.name,
            'Optimization problem parameters':
                self.optimization_problem.parameters,
            'Optimizer': self.optimizer,
            'Optimizer options': self.optimizer_options,
            'exit flag': self.exit_flag,
            'exit message': self.exit_message,
            'time elapsed': self.time_elapsed,
            'x': self.x,
            'f': self.f.tolist(),
            'g': self.g.tolist(),
            'x0': self.optimization_problem.x0,
            'CADET-Process version': self.version
        }

    def save(self, directory):
        path = os.path.join(
            settings.project_directory, directory, 'results.json'
        )
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def plot_solution(self):
        pass
