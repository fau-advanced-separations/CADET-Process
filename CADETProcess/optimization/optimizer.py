import os
import shutil
import time
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError, log, settings
from CADETProcess.dataStructure import (
    RangedInteger,
    Structure,
    Typed,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.optimization import (
    Joblib,
    OptimizationProblem,
    OptimizationResults,
    ParallelizationBackendBase,
    Population,
)

__all__ = ["OptimizerBase"]


class OptimizerBase(Structure):
    """
    BaseClass for optimization solver APIs.

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
    cv_bounds_tol : float
        Tolerance for bounds constraint violation.
        The default is 0.0.
    cv_lincon_tol : float
        Tolerance for linear constraints violation.
        The default is 0.0.
    cv_lineqcon_tol : float
        Tolerance for linear equality constraints violation.
        The default is 0.0.
    cv_nonlincon_tol : float
        Tolerance for nonlinear constraints violation.
        The default is 0.0.
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
    n_cores : int, optional
        Proxy to the number of cores used by the parallelization backend.
    """

    is_population_based = False

    supports_single_objective = True
    supports_multi_objective = False
    supports_linear_constraints = False
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = False
    supports_bounds = False

    ignore_linear_constraints_config = False

    progress_frequency = RangedInteger(lb=1)

    x_tol = UnsignedFloat()
    f_tol = UnsignedFloat()

    cv_bounds_tol = UnsignedFloat(default=0.0)
    cv_lineqcon_tol = UnsignedFloat(default=0.0)
    cv_lincon_tol = UnsignedFloat(default=0.0)
    cv_nonlincon_tol = UnsignedFloat(default=0.0)

    n_max_iter = UnsignedInteger(default=100000)
    n_max_evals = UnsignedInteger(default=100000)

    similarity_tol = UnsignedFloat()
    parallelization_backend = Typed(ty=ParallelizationBackendBase)

    _general_options = [
        "progress_frequency",
        "x_tol",
        "f_tol",
        "cv_bounds_tol",
        "cv_lincon_tol",
        "cv_lineqcon_tol",
        "cv_nonlincon_tol",
        "n_max_iter",
        "n_max_evals",
        "similarity_tol",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize OptimizerBase."""
        self.parallelization_backend = Joblib()

        super().__init__(*args, **kwargs)

    def optimize(
        self,
        optimization_problem: OptimizationProblem,
        x0: Optional[list] = None,
        save_results: Optional[bool] = True,
        results_directory: Optional[str] = None,
        use_checkpoint: Optional[bool] = False,
        overwrite_results_directory: Optional[bool] = False,
        exist_ok: Optional[bool] = True,
        log_level: Optional[str] = "INFO",
        reinit_cache: Optional[bool] = True,
        delete_cache: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> OptimizationResults:
        """
        Solve OptimizationProblem.

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
            If True, reinitialize the Cache. The default is True.
        delete_cache : bool, optional
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
        self._current_cache_entries = []

        self.logger = log.get_logger(str(self), level=log_level)

        # Check OptimizationProblem
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError("Expected OptimizationProblem")

        if not self.check_optimization_problem(optimization_problem):
            raise CADETProcessError("Cannot solve OptimizationProblem.")

        self.optimization_problem = optimization_problem

        # Setup OptimizationResults
        self.results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=self,
            similarity_tol=self.similarity_tol,
        )

        if save_results:
            if results_directory is None:
                results_directory = (
                    settings.working_directory / f"results_{optimization_problem.name}"
                )
            results_directory = Path(results_directory)
            if overwrite_results_directory and results_directory.exists():
                shutil.rmtree(results_directory)
            try:
                results_directory.mkdir(exist_ok=exist_ok, parents=True)
            except FileExistsError:
                raise CADETProcessError(
                    "Results directory already exists. "
                    "To continue using same directory, 'exist_ok=True'. "
                    "To overwrite, set 'overwrite_results_directory=True. "
                )

            self.results.results_directory = results_directory

            checkpoint_path = os.path.join(results_directory, "checkpoint.h5")
            if use_checkpoint and os.path.isfile(checkpoint_path):
                self.logger.info("Continue optimization from checkpoint.")
                self.results.load_results(checkpoint_path)
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
            self.optimization_problem.setup_cache(self.n_cores)

        if x0 is not None:
            flag, x0 = self.check_x0(optimization_problem, x0)

            if not flag:
                raise ValueError("x0 contains invalid entries.")

        log.log_time("Optimization", self.logger.level)(self._run)
        log.log_results("Optimization", self.logger.level)(self._run)
        log.log_exceptions("Optimization", self.logger.level)(self._run)

        backend = plt.get_backend()
        plt.switch_backend("agg")

        start = time.time()
        self._run(self.optimization_problem, x0, *args, **kwargs)
        time_elapsed = time.time() - start

        self.results.time_elapsed = time_elapsed
        self.results.cpu_time = self.n_cores * time_elapsed

        self.run_final_processing()

        if delete_cache:
            optimization_problem.delete_cache(reinit=True)
        self._current_cache_entries = []

        plt.switch_backend(backend)

        if not self.results.success:
            raise CADETProcessError(
                f"Optimizaton failed with message: {self.results.exit_message}"
            )

        return self.results

    def load_results(
        self,
        checkpoint_path: str | Path,
        optimization_problem: Optional[OptimizationProblem] = None,
    ) -> OptimizationResults:
        """
        Load optimization results from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to the checkpoint file.
        optimization_problem : OptimizationProblem, optional
            The optimization problem associated with the results.
            If None, results are loaded without a problem reference.

        Returns
        -------
        OptimizationResults
            The loaded optimization results.
        """
        results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=self,
            similarity_tol=self.similarity_tol,
        )

        results.load_results(checkpoint_path)

        return results

    @abstractmethod
    def _run(
        optimization_problem, x0: Optional[list] = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Abstract Method for solving an optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Optimization problem to be solved.
        x0 : list, optional
            Initial population of independent variables in untransformed space.
        *args : args, optional
            Additional args that may be processed.
        **kwargs : kwargs, optional
            Additional kwargs that may be processed.

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

    def check_optimization_problem(
        self, optimization_problem: OptimizationProblem
    ) -> bool:
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
            ignore_linear_constraints=self.ignore_linear_constraints_config
        ):
            # Warnings are raised internally
            flag = False

        if (
            optimization_problem.n_objectives == 1
            and not self.supports_single_objective
        ):
            warnings.warn("Optimizer does not support single-objective problems")
            flag = False

        if optimization_problem.n_objectives > 1 and not self.supports_multi_objective:
            warnings.warn("Optimizer does not support multi-objective problems")
            flag = False

        if (
            not np.all(
                np.isinf(optimization_problem.lower_bounds_independent_transformed)
            )
            and not np.all(
                np.isinf(optimization_problem.upper_bounds_independent_transformed)
            )
        ) and not self.supports_bounds:
            warnings.warn("Optimizer does not support bounds")
            flag = False

        if (
            optimization_problem.n_linear_constraints > 0
            and not self.supports_linear_constraints
        ):
            warnings.warn(
                "Optimizer does not support problems with linear constraints."
            )
            flag = False

        if (
            optimization_problem.n_linear_equality_constraints > 0
            and not self.supports_linear_equality_constraints
        ):
            warnings.warn(
                "Optimizer does not support problems with linear equality constraints."
            )
            flag = False

        if (
            optimization_problem.n_nonlinear_constraints > 0
            and not self.supports_nonlinear_constraints
        ):
            warnings.warn(
                "Optimizer does not support problems with nonlinear constraints."
            )
            flag = False

        return flag

    def check_x0(
        self, optimization_problem: OptimizationProblem, x0: npt.ArrayLike
    ) -> tuple:
        """
        Check the initial guess x0 for an optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The optimization problem instance to which x0 is related.
        x0 : array_like
            The initial guess for the optimization problem.
            It can be a single individual or a population.

        Returns
        -------
        tuple
            A tuple containing a boolean flag indicating if x0 is valid, and the
            potentially modified x0.
        """
        flag = True

        shape = np.array(x0).shape

        is_x0_1d = len(shape) == 1

        x0 = np.array(x0, ndmin=2)

        n_dependent_variables = optimization_problem.n_dependent_variables
        n_independent_variables = optimization_problem.n_independent_variables
        n_variables = n_dependent_variables + n_independent_variables

        if x0.shape[1] != n_variables and x0.shape[1] != n_independent_variables:
            warnings.warn(
                f"x0 for optimization problem is expected to be of length "
                f"{n_independent_variables} or"
                f"{n_variables}. Got {x0.shape[1]}"
            )
            flag = False

        if n_dependent_variables > 0 and x0.shape[1] == n_variables:
            x0 = [optimization_problem.get_independent_values(ind) for ind in x0]
            warnings.warn(
                "x0 contains dependent values. "
                "Will recompute dependencies for consistency."
            )
            x0 = np.array(x0)

        for x in x0:
            if not optimization_problem.check_individual(
                x,
                get_dependent_values=True,
                cv_bounds_tol=self.cv_bounds_tol,
                cv_lincon_tol=self.cv_lincon_tol,
                cv_lineqcon_tol=self.cv_lineqcon_tol,
                check_nonlinear_constraints=False,
                silent=True,
            ):
                flag = False
                break

        if is_x0_1d:
            x0 = x0[0]

        x0 = x0.tolist()

        return flag, x0

    def _create_population(
        self,
        X_transformed: npt.ArrayLike,
        F: npt.ArrayLike,
        F_min: npt.ArrayLike,
        G: npt.ArrayLike,
        CV_nonlincon: npt.ArrayLike,
    ) -> Population:
        """Create new population from current generation for post procesing."""
        X_transformed = np.array(X_transformed, ndmin=2)
        F = np.array(F, ndmin=2)
        F_min = np.array(F_min, ndmin=2)
        G = np.array(G, ndmin=2)
        CV_nonlincon = np.array(CV_nonlincon, ndmin=2)

        if self.optimization_problem.n_meta_scores > 0:
            M_min = self.optimization_problem.evaluate_meta_scores(
                X_transformed,
                untransform=True,
                ensure_minimization=True,
                parallelization_backend=self.parallelization_backend,
            )
            M = self.optimization_problem.transform_maximization(
                M_min, scores="meta_scores"
            )
        else:
            M_min = None
            M = None

        if self.optimization_problem.n_nonlinear_constraints == 0:
            G = None
            CV_nonlincon = None

        X = self.optimization_problem.get_dependent_values(
            X_transformed, untransform=True
        )
        population = self.optimization_problem.create_population(
            X,
            F=F,
            F_min=F_min,
            G=G,
            CV_nonlincon=CV_nonlincon,
            M=M,
            M_min=M_min,
        )

        for ind in population:
            ind.is_feasible = self.optimization_problem.check_individual(
                ind.x,
                cv_bounds_tol=self.cv_bounds_tol,
                cv_lincon_tol=self.cv_lincon_tol,
                cv_lineqcon_tol=self.cv_lineqcon_tol,
                check_nonlinear_constraints=True,
                cv_nonlincon_tol=self.cv_nonlincon_tol,
                silent=True,
            )

        return population

    def _create_pareto_front(self, X_opt_transformed: npt.ArrayLike) -> Population:
        """Create new pareto front from current generation for post procesing."""
        if X_opt_transformed is None:
            pareto_front = None
        else:
            pareto_front = Population()

            for x_opt_transformed in X_opt_transformed:
                x_opt = self.optimization_problem.get_dependent_values(
                    x_opt_transformed, untransform=True
                )
                ind = self.results.population_all[x_opt]
                pareto_front.add_individual(ind)

        return pareto_front

    def _create_meta_front(self) -> Population:
        """Create new meta front from current generation for post procesing."""
        if self.optimization_problem.n_multi_criteria_decision_functions == 0:
            meta_front = None
        else:
            pareto_front = self.results.pareto_front

            X_meta_front = (
                self.optimization_problem.evaluate_multi_criteria_decision_functions(
                    pareto_front
                )
            )

            meta_front = Population()
            for x in X_meta_front:
                meta_front.add_individual(pareto_front[x])

        return meta_front

    def _evaluate_callbacks(
        self,
        current_generation: int,
        sub_dir: str = None,
    ) -> None:
        if sub_dir is not None:
            callbacks_dir = self.callbacks_dir / sub_dir
            callbacks_dir.mkdir(exist_ok=True, parents=True)
        else:
            callbacks_dir = self.callbacks_dir

        for callback in self.optimization_problem.callbacks:
            if self.optimization_problem.n_callbacks > 1:
                _callbacks_dir = callbacks_dir / str(callback)
                _callbacks_dir.mkdir(exist_ok=True, parents=True)
            else:
                _callbacks_dir = callbacks_dir

            callback.cleanup(_callbacks_dir, current_generation)
            callback._callbacks_dir = _callbacks_dir

        self.optimization_problem.evaluate_callbacks(
            self.results.meta_front,
            current_generation,
            parallelization_backend=self.parallelization_backend,
        )

    def _log_results(self, current_generation: int) -> None:
        self.logger.info(f"Finished Generation {current_generation}.")
        for ind in self.results.meta_front:
            message = f"x: {ind.x}, f: {ind.f}"

            if self.optimization_problem.n_nonlinear_constraints > 0:
                message += f", cv: {ind.cv_nonlincon}"

            if self.optimization_problem.n_meta_scores > 0:
                message += f", m: {ind.m}"
            self.logger.info(message)

    def run_post_processing(
        self,
        X_transformed: Sequence[Sequence[float]],
        F_minimized: Sequence[float | Sequence[float]],
        G: Sequence[float | Sequence[float]],
        CV_nonlincon: Sequence[float],
        current_generation: int,
        X_opt_transformed: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Run post-processing of generation.

        Notes
        -----
        This method also works for optimizers that only perform a single evaluation per
        "generation".

        Parameters
        ----------
        X_transformed : list
            Optimization variable values of generation in independent transformed space.
        F_minimized : list
            Objective function values of generation.
            This assumes that all objective function values are minimized.
        G : list
            Nonlinear constraint function values of generation.
        CV_nonlincon : list
            Nonlinear constraints violation of of generation.
        current_generation : int
            Current generation.
        X_opt_transformed : list, optional
            (Currently) best variable values in independent transformed space.
            If None, internal pareto front is used to determine best values.
        """
        F = self.optimization_problem.transform_maximization(
            F_minimized, scores="objectives"
        )
        population = self._create_population(
            X_transformed, F, F_minimized, G, CV_nonlincon
        )
        self.results.update(population)

        pareto_front = self._create_pareto_front(X_opt_transformed)
        self.results.update_pareto(pareto_front)

        meta_front = self._create_meta_front()
        if meta_front is not None:
            self.results.update_meta(meta_front)

        if (
            self.progress_frequency is not None
            and current_generation % self.progress_frequency == 0
        ):
            self.results.plot_figures(show=False)

        self._evaluate_callbacks(current_generation)

        self.results.save_results("checkpoint")

        # Remove new entries from cache that didn't make it to the meta front
        for x in population.x:
            x_key = x.tobytes()
            if x not in self.results.meta_front.x:
                self.optimization_problem.prune_cache(x_key, close=False)
            else:
                self._current_cache_entries.append(x_key)

        # Remove old meta front entries from cache that were replaced by better ones
        for x_key in self._current_cache_entries:
            x = np.frombuffer(x_key)
            if not np.all(np.isin(x, self.results.meta_front.x)):
                self.optimization_problem.prune_cache(x_key, close=False)
                self._current_cache_entries.remove(x_key)

        self._log_results(current_generation)

    def run_final_processing(self) -> None:
        """Run post processing at the end of the optimization."""
        self.results.plot_figures(show=False)
        if self.optimization_problem.n_callbacks > 0:
            self._evaluate_callbacks(0, "final")
        self.results.save_results("final")

    @property
    def options(self) -> dict:
        """dict: Optimizer options."""
        return {
            opt: getattr(self, opt)
            for opt in (self._general_options + self._specific_options)
        }

    @property
    def specific_options(self) -> dict:
        """dict: Optimizer spcific options."""
        return {opt: getattr(self, opt) for opt in (self._specific_options)}

    @property
    def n_cores(self) -> int:
        """int: Proxy to the number of cores used by the parallelization backend.

        Note, this will always return the actual number of cores used, even if negative
        values are set.

        See Also
        --------
        parallelization_backend
        """
        return self.parallelization_backend._n_cores

    @n_cores.setter
    def n_cores(self, n_cores: int) -> None:
        self.parallelization_backend.n_cores = n_cores

    def __str__(self) -> str:
        """str: String representation."""
        return self.__class__.__name__
