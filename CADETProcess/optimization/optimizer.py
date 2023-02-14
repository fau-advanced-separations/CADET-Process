from abc import abstractmethod
import os
from pathlib import Path
import shutil
import time
import warnings

from cadet import H5
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


__all__ = ['OptimizerBase']


class OptimizerBase(metaclass=StructMeta):
    """BaseClass for optimization solver APIs

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the
    OptimizationProblem configuration to the APIs configuration format and
    convert the results back to the CADET-Process format.

    Attributes
    ----------
    supports_multi_objective : bool
        True, if optimizer supports multi-objective optimization.
    supports_linear_constraints : bool
        True, if optimizer supports linear constraints.
    supports_linear_equality_constraints : bool
        True, if optimizer supports linear equality constraints.
    supports_nonlinear_constraints : bool
        True, if optimizer supports nonlinear constraints.
    progress_frequency : int
        Number of generations after which optimizer reports progress.
        The default is 1.
    n_cores : int, optional
        The number of cores that the optimizer should use.
        The default is 1.
    cv_tol : float
        Tolerance for constraint violation.
        The default is 1e-6.
    similarity_tol : UnsignedFloat
        Tolerance for individuals to be considered similar.
        Similar items are removed from the Pareto front to limit its size.
        The default is None, indicating that all individuals should be kept.

    """

    supports_multi_objective = False
    supports_linear_constraints = False
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = False

    progress_frequency = RangedInteger(lb=1, default=1)
    n_cores = UnsignedInteger(default=1)
    cv_tol = UnsignedFloat(default=1e-6)
    similarity_tol = UnsignedFloat()

    _options = ['progress_frequency', 'n_cores', 'cv_tol', 'similarity_tol']

    def optimize(
            self,
            optimization_problem,
            x0=None,
            save_results=True,
            results_directory=None,
            use_checkpoint=False,
            overwrite_results_directory=False,
            exist_ok=True,
            log_level="INFO",
            reinit_cache=True,
            delete_cache=True,
            *args, **kwargs):
        """Solve OptimizationProblem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            OptimizationProblem to be solved.
        x0 : list, optional
            Initial values. If None, valid points are generated.
        save_results : bool, optional
            If True, save results. The default is True.
        results_directory : str, optional
            Results directory. If None, working directory is used.
            Only has an effect, if save_results == True.
        use_checkpoint : bool, optional
            If True, try continuing fom checkpoint. The default is True.
            Only has an effect, if save_results == True.
        overwrite_results_directory : bool, optional
            If True, overwrite existing results directory. The default is False.
        exist_ok : bool, optional
            If False, Exception is raised when results_directory is not empty.
            The default is True.
        log_level : str, optional
            log level. The default is "INFO".
        reinit_cache : bool, optional
            If True, delete ResultsCache after finishing. The default is True.
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
        self.logger = log.get_logger(str(self), level=log_level)

        # Check OptimizationProblem
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')

        if not self.check_optimization_problem(optimization_problem):
            raise CADETProcessError('Cannot solve OptimizationProblem.')

        self.optimization_problem = optimization_problem

        # Setup OptimizationResults
        self.results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=self,
            similarity_tol=self.similarity_tol,
            cv_tol=self.cv_tol,
        )

        if save_results:
            if results_directory is None:
                results_directory = settings.working_directory / f"results_{optimization_problem.name}"
            results_directory = Path(results_directory)
            if overwrite_results_directory and results_directory.exists():
                shutil.rmtree(results_directory)
            try:
                results_directory.mkdir(
                    exist_ok=exist_ok, parents=True
                )
            except FileExistsError:
                raise CADETProcessError(
                    "Results directory already exists. "
                    "To continue using same directory, 'exist_ok=True'. "
                    "To overwrite, set 'overwrite_results_directory=True. "
                )

            self.results.results_directory = results_directory

            checkpoint_path = os.path.join(results_directory, 'checkpoint.h5')
            if use_checkpoint and os.path.isfile(checkpoint_path):
                self.logger.info("Continue optimization from checkpoint.")
                data = H5()
                data.filename = checkpoint_path
                data.load()

                self.results.update_from_dict(data)
            else:
                self.results.setup_csv('results_meta')
                self.results.setup_csv('results_all')

        # Setup Callbacks
        if save_results and optimization_problem.n_callbacks > 0:
            callbacks_dir = results_directory / "callbacks"
            callbacks_dir.mkdir(exist_ok=True)

            if optimization_problem.n_callbacks > 1:
                for callback in optimization_problem.callbacks:
                    callback_dir = callbacks_dir / str(callback)
                    callback_dir.mkdir(exist_ok=True)
        else:
            callbacks_dir = None
        self.callbacks_dir = callbacks_dir

        if reinit_cache:
            self.optimization_problem.setup_cache()

        log.log_time('Optimization', self.logger.level)(self.run)
        log.log_results('Optimization', self.logger.level)(self.run)
        log.log_exceptions('Optimization', self.logger.level)(self.run)

        backend = plt.get_backend()
        plt.switch_backend('agg')

        start = time.time()

        self.run(self.optimization_problem, x0, *args, **kwargs)

        self.results.time_elapsed = time.time() - start

        if delete_cache:
            optimization_problem.delete_cache(reinit=True)

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

    def check_optimization_problem(self, optimization_problem):
        """
        Check if problem is configured correctly and supported by the optimizer.

        Parameters
        ----------
        optimization_problem: OptimizationProblem
            An optimization problem to check.

        Returns
        -------
        flag : bool
            True if the optimization problem is supported and configured correctly,
            False otherwise.

        """
        flag = True
        if not optimization_problem.check():
            warnings.warn(
                "OptimizationProblem is not configured correctly."
            )
            flag = False

        if optimization_problem.n_objectives > 1 and not self.supports_multi_objective:
            warnings.warn(
                "Optimizer does not support multi-objective problems"
            )
        if optimization_problem.n_linear_constraints > 0\
                and not self.supports_nonlinear_constraints:
            warnings.warn(
                "Optimizer does not support problems with linear constraints."
            )
            flag = False

        if optimization_problem.n_linear_equality_constraints > 0 \
                and not self.supports_nonlinear_constraints:
            warnings.warn(
                "Optimizer does not support problems with linear equality constraints."
            )
            flag = False

        if optimization_problem.n_nonlinear_constraints > 0 \
                and not self.supports_nonlinear_constraints:
            warnings.warn(
                "Optimizer does not support problems with nonlinear constraints."
            )
            flag = False

        return flag

    def _run_post_processing(self, current_iteration):
        if self.optimization_problem.n_multi_criteria_decision_functions > 0:
            pareto_front = self.results.pareto_front

            X_meta_front = \
                self.optimization_problem.evaluate_multi_criteria_decision_functions(
                    pareto_front
                )

            meta_front = Population()
            for x in X_meta_front:
                meta_front.add_individual(pareto_front[x])

            self.results.update_meta(meta_front)

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
            self.results.meta_front,
            current_iteration,
            n_cores=self.n_cores,
        )

        self.results.save_results()

        self.optimization_problem.prune_cache()

    def run_post_evaluation_processing(self, x, f, g, cv, current_evaluation, x_opt=None):
        """Run post-processing of individual evaluation.

        Parameters
        ----------
        x : list
            Optimization variable values of individual.
        f : list
            Objective function values of individual.
        g : list
            Nonlinear constraint function of individual.
        cv : list
            Nonlinear constraints violation of individual.
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
            x, f, g, m, cv, self.cv_tol, x_untransformed,
            self.optimization_problem.independent_variable_names,
            self.optimization_problem.objective_labels,
            self.optimization_problem.nonlinear_constraint_labels,
            self.optimization_problem.meta_score_labels,
            self.optimization_problem.variable_names,
        )

        self.results.update_individual(ind)

        if x_opt is None:
            self.results.update_pareto()
        else:
            pareto_front = Population()
            ind = self.results.population_all[x]
            pareto_front.add_individual(ind)

            self.results.update_pareto(pareto_front)

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
            self, X, F, G, CV, current_generation, X_opt=None):
        """Run post-processing of generation.

        Parameters
        ----------
        X : list
            Optimization Variable values of generation.
        F : list
            Objective function values of generation.
        G : list
            Nonlinear constraint function values of generation.
        CV : list
            Nonlinear constraints violation of of generation.
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
        for x, f, g, cv, m in zip(X, F, G, CV, M):
            x_untransformed \
                = self.optimization_problem.get_dependent_values(
                    x, untransform=True
                )
            ind = Individual(
                x, f, g, m, cv, self.cv_tol, x_untransformed,
                self.optimization_problem.independent_variable_names,
                self.optimization_problem.objective_labels,
                self.optimization_problem.nonlinear_constraint_labels,
                self.optimization_problem.meta_score_labels,
                self.optimization_problem.variable_names,
            )
            population.add_individual(ind)

        self.results.update_population(population)

        if X_opt is None:
            self.results.update_pareto()
        else:
            pareto_front = Population()
            for x in X_opt:
                ind = self.results.population_all[x]
                pareto_front.add_individual(ind)

            self.results.update_pareto(pareto_front)

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
        """dict: Optimizer options."""
        return {opt: getattr(self, opt) for opt in self._options}

    def __str__(self):
        return self.__class__.__name__
