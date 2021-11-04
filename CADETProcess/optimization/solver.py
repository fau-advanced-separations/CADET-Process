from abc import abstractmethod
import json
import os

from CADETProcess.common import settings
from CADETProcess.log import get_logger, log_time, log_results, log_exceptions
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    Dict, Float, List,NdArray, String, UnsignedInteger, UnsignedFloat
)
from CADETProcess.optimization import OptimizationProblem


class SolverBase(metaclass=StructMeta):
    """BaseClass for optimization solver APIs

    Holds the configuration of the individual solvers and gives an interface for
    calling the run method. The class has to convert the OptimizationProblem
    configuration to the APIs configuration format and convert the results back
    to the chromapy format.
    """
    _options = []
    def __init__(self):
        self.logger = get_logger(str(self))

    def optimize(self, optimization_problem, save_results=False, *args, **kwargs):
        """
        """
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')

        if save_results:
            results_directory = optimization_problem.name
            self.logger = get_logger(str(self), log_directory=results_directory)

        log_time('Optimization', self.logger.level)(self.run)
        log_results('Optimization', self.logger.level)(self.run)
        log_exceptions('Optimization', self.logger.level)(self.run)

        results = self.run(optimization_problem, *args, **kwargs)

        if save_results:
            results.save(results_directory)

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
    """Class for storing optimization results including the solver configuration

    Attributes
    ----------
    optimization_problem : OptimizationProblem
        Optimization problem
    evaluation_object : obj
        Evaluation object in optimized state.
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
    x : list
        Values of optimization variables at optimum.
    f : np.ndarray
        Value of objective function at x.
    c : np.ndarray
        Values of constraint function at x
    """
    x0 = List()
    solver_name = String()
    solver_parameters = Dict()
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    x = List()
    f = NdArray()
    c = NdArray()
    performance = Dict()

    def __init__(
            self, optimization_problem, evaluation_object,
            solver_name, solver_parameters, exit_flag, exit_message,
            time_elapsed, x, f, c, performance, frac=None, history=None
        ):

        self.optimization_problem = optimization_problem
        self.evaluation_object = evaluation_object

        self.solver_name = solver_name
        self.solver_parameters = solver_parameters

        self.exit_flag = exit_flag
        self.exit_message = exit_message
        self.time_elapsed = time_elapsed

        self.x = x
        self.f = f
        if c is not None:
            self.c = c

        self.performance = performance

        self.frac = frac
        
        self.history = history


    def to_dict(self):
        return {
            'optimization_problem': self.optimization_problem.name,
            'optimization_problem_parameters':
                self.optimization_problem.parameters,
            'evaluation_object_parameters':
                self.evaluation_object.parameters,
            'x0': self.optimization_problem.x0,
            'solver_name': self.solver_name,
            'solver_parameters': self.solver_parameters,
            'exit_flag': self.exit_flag,
            'exit_message': self.exit_message,
            'time_elapsed': self.time_elapsed,
            'x': self.x,
            'f': self.f.tolist(),
            'c': self.c.tolist(),
            'performance': self.performance,
            'git': {
                'chromapy_branch': str(settings.repo.active_branch),
                'chromapy_commit': settings.repo.head.object.hexsha
            }
        }

    def save(self, directory):
        path = os.path.join(settings.project_directory, directory, 'results.json')
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def plot_solution(self):
        pass
