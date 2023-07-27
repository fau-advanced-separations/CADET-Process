from abc import abstractmethod
import os
from pathlib import Path
import shutil
import time
import warnings

from cadet import H5
import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import settings
from CADETProcess import log
from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Typed, UnsignedInteger, RangedInteger, UnsignedFloat
)

from CADETProcess.optimization import OptimizationProblem
from CADETProcess.optimization import Individual, Population, ParetoFront
from CADETProcess.optimization import ParallelizationBackendBase, Joblib
from CADETProcess.optimization import OptimizationResults

__all__ = ['OptimizerBase']


class OptimizerBase(Structure):
    """BaseClass for optimization solver APIs.

    Holds the configuration of the individual solvers and gives an interface
    for calling the run method. The class has to convert the
    OptimizationProblem configuration to the APIs configuration format and
    convert the results back to the CADET-Process format.

    Attributes
    ----------
    is_population_based : bool
        True, if the optimizer evaluates entire populations at every step.
    supports_multi_objective : bool
        True, if the optimizer supports multi-objective optimization.
    supports_linear_constraints : bool
        True, if the optimizer supports linear constraints.
    supports_linear_equality_constraints : bool
        True, if the optimizer supports linear equality constraints.
    supports_nonlinear_constraints : bool
        True, if the optimizer supports nonlinear constraints.
    supports_bounds : bool
        True, if the optimizer supports bound constraints
    ignore_linear_constraints_config: bool
        True, if the optimizer can handle transforms and dependent variables in linear
        constraints.
    progress_frequency : int
        Number of generations after which the optimizer reports progress.
        The default is 1.
    cv_tol : float
        Tolerance for constraint violation.
        The default is 1e-6.
    similarity_tol : UnsignedFloat, optional
        Tolerance for individuals to be considered similar.
        Similar items are removed from the Pareto front to limit its size.
        The default is None, indicating that all individuals should be kept.
    n_max_evals : int, optional
        Maximum number of function evaluations.
    n_max_iter : int, optional
        Maximum number of iterations (e.g. generations).
    parallelization_backend : ParallelizationBackendBase, optional
        Class used to handle parallelized (and also sequential) evaluation of eval_fun
        functions for each individual in a given population.
        The default parallelization backend is 'Joblib', which provides parallel
        execution using multiple cores.
    n_cores
    """

    is_population_based = False

    supports_multi_objective = False
    supports_linear_constraints = False
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = False
    supports_bounds = False

    ignore_linear_constraints_config = False

    progress_frequency = RangedInteger(lb=1, default=1)

    x_tol = UnsignedFloat()
    f_tol = UnsignedFloat()
    cv_tol = UnsignedFloat(default=0)

    n_max_iter = UnsignedInteger(default=100000)
    n_max_evals = UnsignedInteger(default=100000)

    similarity_tol = UnsignedFloat()
    parallelization_backend = Typed(ty=ParallelizationBackendBase)

    _general_options = [
        'progress_frequency',
        'x_tol', 'f_tol', 'cv_tol', 'similarity_tol',
        'n_max_iter', 'n_max_evals',
    ]

    def __init__(self, *args, **kwargs):
        self.parallelization_backend = Joblib()

        super().__init__(*args, **kwargs)

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
                self.results.setup_csv()

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

        if x0 is None:
            if optimization_problem.n_linear_equality_constraints > 0:
                raise CADETProcessError(
                    "x0 should be set by the user if linear equality "
                    "constraints are specified."
                )
        else:
            x0check = np.array(x0, ndmin=2)
            for x in x0check:
                if not optimization_problem.check_bounds(x):
                    raise CADETProcessError(f"x0 = {x} does not satisfy bounds.")
                if not optimization_problem.check_linear_constraints(x):
                    raise CADETProcessError(f"x0 = {x} does not satisfy linear constraints.")
                if not optimization_problem.check_linear_equality_constraints(x):
                    raise CADETProcessError(f"x0 = {x} does not satisfy linear equality constraints.")

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
        if not optimization_problem.check_config(
                ignore_linear_constraints=self.ignore_linear_constraints_config):
            # Warnings are raised internally
            flag = False

        if optimization_problem.n_objectives > 1 and not self.supports_multi_objective:
            warnings.warn(
                "Optimizer does not support multi-objective problems"
            )
            flag = False

        if (
            not np.all(np.isinf(optimization_problem.lower_bounds_independent_transformed))
                and
            not np.all(np.isinf(optimization_problem.upper_bounds_independent_transformed))
        ) and not self.supports_bounds:
            warnings.warn(
                "Optimizer does not support bounds"
            )
            flag = False

        if optimization_problem.n_linear_constraints > 0 \
                and not self.supports_linear_constraints:
            warnings.warn(
                "Optimizer does not support problems with linear constraints."
            )
            flag = False

        if optimization_problem.n_linear_equality_constraints > 0 \
                and not self.supports_linear_equality_constraints:
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
            parallelization_backend=self.parallelization_backend,
        )

        self.results.save_results()

        self.optimization_problem.prune_cache()

    def run_post_evaluation_processing(
            self, x_transformed, f, g, cv, current_evaluation, x_opt_transformed=None):
        """Run post-processing of individual evaluation.

        Parameters
        ----------
        x_transformed : list
            Optimization variable values of individual in independent transformed space.
        f : list
            Objective function values of individual.
        g : list
            Nonlinear constraint function of individual.
        cv : list
            Nonlinear constraints violation of individual.
        current_evaluation : int
            Current evaluation.
        x_opt_transformed : list, optional
            Best individual(s) at current iteration in independent transformed space.
            If None, internal pareto front is used to determine best indiviudal.

        """
        if self.optimization_problem.n_meta_scores > 0:
            m = self.optimization_problem.evaluate_meta_scores(
                x_transformed,
                untransform=True,
            )
        else:
            m = None

        x = self.optimization_problem.get_dependent_values(
                x_transformed, untransform=True
            )

        ind = Individual(
            x, f, g, m, cv, self.cv_tol, x_transformed,
            self.optimization_problem.independent_variable_names,
            self.optimization_problem.objective_labels,
            self.optimization_problem.nonlinear_constraint_labels,
            self.optimization_problem.meta_score_labels,
            self.optimization_problem.variable_names,
        )

        self.results.update_individual(ind)

        if x_opt_transformed is None:
            self.results.update_pareto()
        else:
            x_opt = self.optimization_problem.get_dependent_values(
                x_opt_transformed, untransform=True
            )
            pareto_front = Population()
            ind = self.results.population_all[x_opt]
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
            self, X_transformed, F, G, CV, current_generation, X_opt_transformed=None):
        """Run post-processing of generation.

        Parameters
        ----------
        X_transformed : list
            Optimization variable values of generation in independent transformed space.
        F : list
            Objective function values of generation.
        G : list
            Nonlinear constraint function values of generation.
        CV : list
            Nonlinear constraints violation of of generation.
        current_generation : int
            Current generation.
        X_opt_transformed : list, optional
            (Currently) best variable values in independent transformed space.
            If None, internal pareto front is used to determine best values.

        """
        if self.optimization_problem.n_meta_scores > 0:
            M = self.optimization_problem.evaluate_meta_scores_population(
                X_transformed,
                untransform=True,
                parallelization_backend=self.parallelization_backend,
            )
        else:
            M = len(X_transformed)*[None]

        if self.optimization_problem.n_nonlinear_constraints == 0:
            G = len(X_transformed)*[None]
            CV = len(X_transformed)*[None]

        population = Population()
        for x_transformed, f, g, cv, m in zip(X_transformed, F, G, CV, M):
            x = self.optimization_problem.get_dependent_values(
                x_transformed, untransform=True
            )
            ind = Individual(
                x, f, g, m, cv, self.cv_tol, x_transformed,
                self.optimization_problem.independent_variable_names,
                self.optimization_problem.objective_labels,
                self.optimization_problem.nonlinear_constraint_labels,
                self.optimization_problem.meta_score_labels,
                self.optimization_problem.variable_names,
            )
            population.add_individual(ind)

        self.results.update_population(population)

        if X_opt_transformed is None:
            self.results.update_pareto()
        else:
            pareto_front = Population()

            for x_opt_transformed in X_opt_transformed:
                x_opt = self.optimization_problem.get_dependent_values(
                    x_opt_transformed, untransform=True
                )
                ind = self.results.population_all[x_opt]
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
        return {
            opt: getattr(self, opt)
            for opt in (self._general_options + self._specific_options)
        }

    @property
    def specific_options(self):
        """dict: Optimizer spcific options."""
        return {
            opt: getattr(self, opt)
            for opt in (self._specific_options)
        }

    @property
    def n_cores(self):
        """int: Proxy to the number of cores used by the parallelization backend.

        See Also
        --------
        parallelization_backend

        """
        return self.parallelization_backend.n_cores

    @n_cores.setter
    def n_cores(self, n_cores):
        self.parallelization_backend.n_cores = n_cores

    def __str__(self):
        return self.__class__.__name__
