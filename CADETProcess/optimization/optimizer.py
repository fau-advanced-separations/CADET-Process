from abc import abstractmethod
from pathlib import Path
import time

import matplotlib.pyplot as plt

from CADETProcess import settings
from CADETProcess import log
from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    UnsignedInteger, RangedInteger, UnsignedFloat
)

from CADETProcess.optimization import OptimizationProblem
from CADETProcess.optimization import Individual, Population, ParetoFront
from CADETProcess.optimization import OptimizationResults


class OptimizerBase(metaclass=StructMeta):
    """BaseClass for optimization solver APIs

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the
    OptimizationProblem configuration to the APIs configuration format and
    convert the results back to the CADET-Process format.

    """
    supports_multi_objective = False
    supports_linear_constraints = False
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = False

    _options = []
    progress_frequency = RangedInteger(lb=1, default=1)
    n_cores = UnsignedInteger(default=1)
    cv_tol = UnsignedFloat(default=1e-6)

    def optimize(
            self,
            optimization_problem,
            save_results=True,
            results_directory=None,
            log_level="INFO",
            delete_cache=False,
            remove_similar=True,
            *args, **kwargs):
        """Solve OptimizationProblem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            OptimizationProblem to be solved.
        save_results : bool, optional
            If True, save results. The default is True.
        results_directory : str, optional
            Results directory. If None, working directory is used.
            Only has an effect, if save_results == True.
        log_level : str, optional
            log level. The default is "INFO".
        delete_cache : bool, optional
            If True, delete ResultsCache after finishing. The default is False.
        remove_similar : bool, optional
            If True, similar entries are removed from pareto front.
        *args : TYPE
            Additional arguments for Optimizer.
        **kwargs : TYPE
            Additional keyword arguments for Optimizer.

        Raises
        ------
        TypeError
            If optimization_problem is not an instance of OptimizationProblem.
        CADETProcessError
            If Optimizer is not suited for OptimizationProblem (e.g. multi-objective).

        Returns
        -------
        results : OptimizationResults
            Results of the Optimization.

        See Also
        --------
        OptimizationProblem
        OptimizationResults
        CADETProcess.optimization.ResultsCache

        """
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')

        if optimization_problem.n_objectives > 1 and not self.supports_multi_objective:
            raise CADETProcessError(
                "Optimizer does not support multi-objective problems"
            )
        if optimization_problem.n_linear_constraints > 0\
                and not self.supports_nonlinear_constraints:
            raise CADETProcessError(
                "Optimizer does not support problems with linear constraints."
            )
        if optimization_problem.n_linear_equality_constraints > 0 \
                and not self.supports_nonlinear_constraints:
            raise CADETProcessError(
                "Optimizer does not support problems with linear equality constraints."
            )
        if optimization_problem.n_nonlinear_constraints > 0 \
                and not self.supports_nonlinear_constraints:
            raise CADETProcessError(
                "Optimizer does not support problems with nonlinear constraints."
            )

        self.optimization_problem = optimization_problem

        if save_results:
            if results_directory is None:
                results_directory = settings.working_directory
            results_directory = Path(results_directory)
            results_directory.mkdir(exist_ok=True)
        self.results_directory = results_directory

        if self.optimization_problem.n_callbacks > 0:
            callbacks_dir = results_directory / "callbacks"
            callbacks_dir.mkdir(exist_ok=True)

            if self.optimization_problem.n_callbacks > 1:
                for callback in self.optimization_problem.callbacks:
                    callback_dir = callbacks_dir / str(callback)
                    callback_dir.mkdir(exist_ok=True)
        else:
            callbacks_dir = None
        self.callbacks_dir = callbacks_dir

        self.remove_similar = remove_similar
        self.results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=self,
            results_directory=self.results_directory,
            cv_tol=self.cv_tol
        )

        self.logger = log.get_logger(str(self), level=log_level)

        log.log_time('Optimization', self.logger.level)(self.run)
        log.log_results('Optimization', self.logger.level)(self.run)
        log.log_exceptions('Optimization', self.logger.level)(self.run)

        backend = plt.get_backend()
        plt.switch_backend('agg')

        start = time.time()

        self.run(optimization_problem, *args, **kwargs)

        self.results.time_elapsed = time.time() - start

        if delete_cache:
            optimization_problem.delete_cache()

        plt.switch_backend(backend)

        return self.results

    @abstractmethod
    def run(optimization_problem, *args, **kwargs):
        """Abstract Method for solving an optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Optimization problem to be solved.

        Returns
        -------
        results : OptimizationResults
            Optimization results including OptimizationProblem and Optimizer
            configuration.

        Raises
        ------
        CADETProcessError
            If solver doesn't terminate successfully

        """
        return

    def _run_post_processing(self, current_iteration):
        if self.remove_similar:
            self.results.pareto_front.remove_similar()

        if self.optimization_problem.n_multi_criteria_decision_functions > 0:
            X_meta_population = \
                self.optimization_problem.evaluate_multi_criteria_decision_functions(
                    self.results.pareto_front
                )

            meta_population = Population()
            for x in X_meta_population:
                meta_population.add_individual(self.results.pareto_front[x])

            self.results.meta_population = meta_population

        self.results.update_progress()

        if current_iteration % self.progress_frequency == 0:
            self.results.plot_figures(show=False)

        for callback in self.optimization_problem.callbacks:
            if self.optimization_problem.n_callbacks > 1:
                _callbacks_dir = self.callbacks_dir / str(callback)
            else:
                _callbacks_dir = self.callbacks_dir
            callback.cleanup(_callbacks_dir, current_iteration)
            callback._callbacks_dir = _callbacks_dir

        self.optimization_problem.evaluate_callbacks_population(
            self.results.meta_population,
            current_iteration,
            n_cores=self.n_cores,
        )

        self.results.save_results()

        self.optimization_problem.prune_cache()

    def run_post_evaluation_processing(
            self, x, f, g, current_evaluation, x_opt=None):
        """Run post-processing of individual evaluation.

        Parameters
        ----------
        x : list
            Optimization variable values of individual.
        f : list
            Objective function values of individual.
        g : list
            Nonlinear constraint function of individual.
        current_evaluation : int
            Current evaluation.
        x_opt : list, optional
            Best individual at current iteration.
            If None, internal pareto front is used to determine best indiviudal.

        """
        if self.optimization_problem.n_meta_scores > 0:
            m = self.optimization_problem.evaluate_meta_scores(
                x,
                untransform=True,
            )
        else:
            m = None

        x_untransformed \
            = self.optimization_problem.get_dependent_values(
                x, untransform=True
            )
        ind = Individual(
            x, f, g, m, x_untransformed,
            self.optimization_problem.independent_variable_names,
            self.optimization_problem.objective_labels,
            self.optimization_problem.nonlinear_constraint_labels,
            self.optimization_problem.meta_score_labels,
            self.optimization_problem.variable_names,
        )

        self.results.update_individual(ind)

        if x_opt is not None:
            pareto_front = Population()
            ind = self.results.population_all[x]
            pareto_front.add_individual(ind)
            self.results.pareto_front = pareto_front
        else:
            self.results.pareto_front.update_individual(ind)

        self._run_post_processing(current_evaluation)

        self.logger.info(
            f'Finished Evaluation {current_evaluation}.'
        )
        for ind in self.results.pareto_front:
            message = f'x: {ind.x}, f: {ind.f}'

            if self.optimization_problem.n_nonlinear_constraints > 0:
                message += f', g: {ind.g}'

            if self.optimization_problem.n_meta_scores > 0:
                message += f', m: {ind.m}'
            self.logger.info(message)

    def run_post_generation_processing(
            self, X, F, G, current_generation, X_opt=None):
        """Run post-processing of generation.

        Parameters
        ----------
        X : list
            Optimization Variable values of generation.
        F : TYPE
            Objective function values of generation.
        G : TYPE
            Nonlinear constraint function values of generation.
        current_generation : int
            Current generation.
        X_opt : list, optional
            (Currently) best variable values.
            If None, internal pareto front is used to determine best values.

        """
        if self.optimization_problem.n_meta_scores > 0:
            M = self.optimization_problem.evaluate_meta_scores_population(
                X,
                untransform=True,
                n_cores=self.n_cores,
            )
        else:
            M = len(X)*[None]

        population = Population()
        for x, f, g, m in zip(X, F, G, M):
            x_untransformed \
                = self.optimization_problem.get_dependent_values(
                    x, untransform=True
                )
            ind = Individual(
                x, f, g, m, x_untransformed,
                self.optimization_problem.independent_variable_names,
                self.optimization_problem.objective_labels,
                self.optimization_problem.nonlinear_constraint_labels,
                self.optimization_problem.meta_score_labels,
                self.optimization_problem.variable_names,
            )
            population.add_individual(ind)

        self.results.update_population(population)

        if X_opt is not None:
            pareto_front = Population()
            for x in X_opt:
                ind = self.results.population_all[x]
                pareto_front.add_individual(ind)

            self.results.pareto_front = pareto_front
        else:
            self.results.pareto_front.update_population(population)

        self._run_post_processing(current_generation)

        self.logger.info(
            f'Finished Generation {current_generation}.'
        )
        for ind in self.results.pareto_front:
            message = f'x: {ind.x}, f: {ind.f}'

            if self.optimization_problem.n_nonlinear_constraints > 0:
                message += f', g: {ind.g}'

            if self.optimization_problem.n_meta_scores > 0:
                message += f', m: {ind.m}'
            self.logger.info(message)

    @property
    def options(self):
        return {opt: getattr(self, opt) for opt in self._options}

    def __str__(self):
        return self.__class__.__name__
