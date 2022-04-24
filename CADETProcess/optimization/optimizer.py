from abc import abstractmethod
import json
import os

import matplotlib.pyplot as plt

import CADETProcess
from CADETProcess import settings
from CADETProcess import log
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    List, NdArray, String, RangedInteger, UnsignedInteger, UnsignedFloat
)
from CADETProcess.optimization import OptimizationProblem, OptimizationProgress


class OptimizerBase(metaclass=StructMeta):
    """BaseClass for optimization solver APIs

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the
    OptimizationProblem configuration to the APIs configuration format and
    convert the results back to the CADET-Process format.

    """
    _options = []
    progress_frequency = RangedInteger(lb=1, default=1)

    def __init__(self, log_level="INFO", save_log=True):
        self.logger = log.get_logger(
            str(self), level=log_level, save_log=save_log
        )

    def optimize(
            self, optimization_problem,
            save_results=True,
            *args, **kwargs):
        """
        """
        backend = plt.get_backend()
        plt.switch_backend('agg')

        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')

        self.progress = OptimizationProgress(
            optimization_problem, save_results
        )

        self.setup_directories(save_results)

        log.log_time('Optimization', self.logger.level)(self.run)
        log.log_results('Optimization', self.logger.level)(self.run)
        log.log_exceptions('Optimization', self.logger.level)(self.run)

        results = self.run(optimization_problem, *args, **kwargs)

        plt.switch_backend(backend)

        if save_results:
            results.save(self.results_directory)

        return results

    def setup_directories(self, save_results, overwrite=True):
        self.working_directory = settings.working_directory

        if save_results:
            progress_dir = self.working_directory / 'progress'
            progress_dir.mkdir(exist_ok=overwrite)
        else:
            progress_dir = None
        self.progress.progress_directory = progress_dir

        if save_results:
            results_dir = self.working_directory / 'results'
            results_dir.mkdir(exist_ok=overwrite)
        else:
            results_dir = None
        self.progress.results_directory = results_dir

        self.results_directory = results_dir

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
    progress : OptimizationProgress
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
            time_elapsed, x, f, g=None, progress=None):

        self.optimization_problem = optimization_problem

        self.optimizer = optimizer
        self.optimizer_options = optimizer_options

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.x = x
        self.f = f
        self.g = g

        self.progress = progress

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
        path = os.path.join(directory, 'results.json')

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
