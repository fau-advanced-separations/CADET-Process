import warnings
from typing import Callable, Optional

import numpy as np

from CADETProcess import CADETProcessError, SimulationResults
from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import (
    COBYLA,
    OptimizationProblem,
    OptimizationResults,
    OptimizerBase,
)
from CADETProcess.performance import Mass, Performance, Purity

__all__ = ["FractionationOptimizer"]


class FractionationEvaluator:
    """Dummy Evaluator to enable caching."""

    def evaluate(self, fractionator: Fractionator) -> Performance:
        """
        Evaluate the fractionator.

        Parameters
        ----------
        fractionator: Fractionator
            The Fractionator object to be evaluated.

        Returns
        -------
        object
            The evaluation result.
        """
        return fractionator.performance

    __call__ = evaluate

    def __str__(self) -> str:
        """str: Name of the FractionationEvaluator."""
        return __class__.__name__


class FractionationOptimizer:
    """Configuration for fractionating Chromatograms."""

    def __init__(
        self,
        optimizer: Optional[OptimizerBase] = None,
        log_level: str = "WARNING",
    ) -> None:
        """
        Initialize the FractionationOptimizer.

        Parameters
        ----------
        optimizer: OptimizerBase, optional
            Optimizer for optimizing the fractionation times.
            If no value is specified, a default COBYLA optimizer will be used.
        log_level: {'WARNING', 'INFO', 'DEBUG', 'ERROR'}
            Log level for the fractionation optimization process.
            The default is 'WARNING'.
        """
        if optimizer is None:
            optimizer = COBYLA()
            optimizer.tol = 1e-4
            optimizer.catol = 5e-3
            optimizer.rhobeg = 1e-3

        self.optimizer = optimizer
        self.log_level = log_level

    @property
    def optimizer(self) -> OptimizerBase:
        """OptimizerBase: Optimizer for optimizing the fractionation times."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: OptimizerBase) -> None:
        """
        Set the optimizer.

        Parameters
        ----------
        optimizer: OptimizerBase
            The optimizer to be set.

        Raises
        ------
        TypeError
            If the optimizer is not an instance of OptimizerBase.
        """
        if not isinstance(optimizer, OptimizerBase):
            raise TypeError("Expected OptimizerBase")
        self._optimizer = optimizer

    def _setup_fractionator(
        self,
        simulation_results: SimulationResults,
        purity_required: list[float],
        components: Optional[list] = None,
        use_total_concentration_components: bool = True,
        allow_empty_fractions: bool = True,
    ) -> Fractionator:
        """
        Set up the Fractionator for optimizing the fractionation times of Chromatograms.

        Parameters
        ----------
        simulation_results: object
            Simulation results to be used for setting up the Fractionator object.
        purity_required : list of floats
            Minimum purity required for the components in the fractionation.
        components: list, optional
            List of components to consider in the fractionation process.
        use_total_concentration_components: bool, optional
            If True, use the total concentration of the components. The default is True.
        allow_empty_fractions: bool, optional
            If True, allow empty fractions. The default is True.

        Returns
        -------
        Fractionator
            The Fractionator object that has been set up using the provided arguments.
        """
        frac = Fractionator(
            simulation_results,
            components=components,
            use_total_concentration_components=use_total_concentration_components,
        )

        frac.initial_values(purity_required)

        if not np.any(frac.n_fractions_per_pool[:-1]):
            raise CADETProcessError("No areas found with sufficient purity.")

        if not allow_empty_fractions:
            empty_fractions = []
            for i, comp in enumerate(purity_required):
                if comp > 0:
                    if frac.fraction_pools[i].n_fractions == 0:
                        empty_fractions.append(i)
            if len(empty_fractions) != 0:
                raise CADETProcessError(
                    "No areas found with sufficient purity for component(s) "
                    f"{[str(frac.component_system[i]) for i in empty_fractions]}."
                )

        return frac

    def _setup_optimization_problem(
        self,
        frac: Fractionator,
        purity_required: list[float],
        allow_empty_fractions: bool = True,
        ranking: Optional[int | list[float]] = 1,
        obj_fun: Optional[Callable] = None,
        minimize: bool = True,
        bad_metrics: Optional[float | list[float]] = None,
        n_objectives: int = 1,
    ) -> tuple[OptimizationProblem, list[float]]:
        """
        Set up the OptimizationProblem for optimizing the fractionation times.

        Parameters
        ----------
        frac : Fractionator
            The Fractionator object.
        purity_required : list[float]
            Minimum purity required for the components in the fractionation.
        allow_empty_fractions: bool, optional
            If True, allow empty fractions. The default is True.
        ranking : Optional[int | list[float]] = 1,
            Weighting factors for individual components.
            If 1, the same value is assumed for all components.
            If None, no ranking is used and the problem is solved as multi-objective.
            The default is 1.
        obj_fun : callable, optional
            Alternative objective function.
            If no function is provided, the fraction mass is maximized.
            The default is None.
        bad_metrics : float or list of floats, optional
            Values to be returned if evaluation of objective function failes.
            The default is 0.
        minimize : bool, optional
            If True, the obj_fun is assumed to return a value that is to be minimized.
            The default it True.
        n_objectives : int
            Number of objectives. The default is 1.

        Raises
        ------
        CADETProcessError
            If the optimization problem setup fails.

        Returns
        -------
        OptimizationProblem
            The configured OptimizationProblem object.
        list
            The initial values for the optimization variables.
        """
        # Handle empty fractions
        n_fractions = np.array([pool.n_fractions for pool in frac.fraction_pools])
        empty_fractions = np.where(n_fractions[0:-1] == 0)[0]
        if len(empty_fractions) > 0 and allow_empty_fractions:
            for empty_fraction in empty_fractions:
                purity_required[empty_fraction] = 0

        # Setup Optimization Problem
        opt = OptimizationProblem(
            "FractionationOptimization",
            log_level=self.log_level,
            use_diskcache=False,
        )

        opt.add_evaluation_object(frac)

        frac_evaluator = FractionationEvaluator()
        opt.add_evaluator(frac_evaluator)

        if obj_fun is None:
            obj_fun = Mass(ranking=ranking)
            minimize = False
            bad_metrics = 0

        opt.add_objective(
            obj_fun,
            requires=frac_evaluator,
            n_objectives=n_objectives,
            minimize=minimize,
            bad_metrics=bad_metrics,
        )

        purity = Purity()
        purity.n_metrics = frac.component_system.n_comp
        opt.add_nonlinear_constraint(
            purity,
            n_nonlinear_constraints=len(purity_required),
            bounds=purity_required,
            comparison_operator="ge",
            requires=frac_evaluator,
        )

        for evt in frac.events:
            opt.add_variable(
                evt.name,
                parameter_path=evt.name + ".time",
                lb=-frac.cycle_time,
                ub=2 * frac.cycle_time,
                transform="linear",
            )

        for chrom_index, chrom in enumerate(frac.chromatograms):
            chrom_events = frac.chromatogram_events[chrom]
            evt_names = [evt.name for evt in chrom_events]
            for evt_index, evt in enumerate(chrom_events):
                if evt_index < len(chrom_events) - 1:
                    opt.add_linear_constraint(
                        [evt_names[evt_index], evt_names[evt_index + 1]], [1, -1]
                    )
                else:
                    opt.add_linear_constraint(
                        [evt_names[0], evt_names[-1]], [-1, 1], frac.cycle_time
                    )

        x0 = [evt.time for evt in frac.events]

        if not opt.check_nonlinear_constraints(x0):
            raise CADETProcessError("No areas found with sufficient purity.")

        return opt, x0

    def optimize_fractionation(
        self,
        simulation_results: SimulationResults,
        purity_required: float | list[float],
        components: Optional[list[str]] = None,
        use_total_concentration_components: bool = True,
        ranking: Optional[int | list[float]] = 1,
        obj_fun: Optional[Callable] = None,
        n_objectives: int = 1,
        bad_metrics: float | list[float] = 0,
        minimize: bool = True,
        allow_empty_fractions: bool = True,
        ignore_failed: bool = False,
        return_optimization_results: bool = False,
        save_results: bool = False,
    ) -> Fractionator | tuple[Fractionator, OptimizationResults]:
        """
        Optimize the fractionation times with respect to purity constraints.

        Parameters
        ----------
        simulation_results : SimulationResults
            Results containing the chromatograms for fractionation.
        purity_required :  float or array_like
            Minimum required purity for components. If is float, the same
            value is assumed for all components.
        components : list
            List of components to consider in the fractionation process.
        use_total_concentration_components : bool, Default=True
            Flag wheter to use the total concentration components.
        ranking : Optional[int | list[float]] = 1,
            Weighting factors for individual components.
            If 1, the same value is assumed for all components.
            If None, no ranking is used and the problem is solved as multi-objective.
            The default is 1.
        obj_fun : function, optional
            Objective function used for OptimizationProblem.
            If COBYLA is used, must return single objective.
            If is None, the mass of all components is maximized.
        n_objectives : int, optional
            Number of objectives returned by obj_fun. The default is 1.
        bad_metrics : float or list of floats, optional
            Values to be returned if evaluation of objective function failes.
            The default is 0.
        minimize : bool, optional
            If True, the obj_fun is assumed to return a value that is to be minimized.
            The default it True.
        allow_empty_fractions: bool, optional
            If True, allow empty fractions. The default is True.
        ignore_failed : bool, optional
            Ignore failed optimization and use initial values.
            The default is False.
        return_optimization_results : bool, optional
            If True, return optimization results.
            Otherwise, return fractionation object.
            The default is False.
        save_results : bool, optional
            If True, save optimization results. The default is False.

        Returns
        -------
        Fractionator or OptimizationResults
            The Fractionator object with optimized cut times
            or the OptimizationResults object.

        Raises
        ------
        TypeError
            If simulation_results is not an instance of SimulationResults.
        CADETProcessError
            If simulation_results do not contain chromatograms.
        Warning
            If purity requirements cannot be fulfilled.

        See Also
        --------
        _setup_fractionator
        _setup_optimization_problem
        Fractionator
        CADETProcess.solution.SolutionIO
        CADETProcess.optimization.OptimizationProblem
        CADETProcess.optimization.OptimizerBase
        """
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError("Expected SimulationResults.")

        if len(simulation_results.chromatograms) == 0:
            raise CADETProcessError("Simulation results do not contain chromatogram.")

        if isinstance(purity_required, float):
            n_comp = simulation_results.component_system.n_comp
            purity_required = n_comp * [purity_required]

        # Store previous lock state, unlock to ensure consistent values
        lock_state = simulation_results.process.lock
        simulation_results.process.lock = False

        frac = self._setup_fractionator(
            simulation_results,
            purity_required,
            components=components,
            use_total_concentration_components=use_total_concentration_components,
            allow_empty_fractions=allow_empty_fractions,
        )

        opt, x0 = self._setup_optimization_problem(
            frac,
            purity_required,
            allow_empty_fractions=allow_empty_fractions,
            ranking=ranking,
            obj_fun=obj_fun,
            n_objectives=n_objectives,
            minimize=minimize,
            bad_metrics=bad_metrics,
        )

        # Lock to enable caching
        simulation_results.process.lock = True
        try:
            results = self.optimizer.optimize(
                opt,
                x0,
                save_results=save_results,
                log_level=self.log_level,
                delete_cache=True,
            )
            opt.set_variables(results.x[0])
            frac.reset()
        except CADETProcessError as e:
            if ignore_failed:
                warnings.warn("Optimization failed. Returning initial values")
                frac.initial_values(purity_required)
            else:
                raise CADETProcessError(str(e))
        finally:
            # Restore previous lock state
            simulation_results.process.lock = lock_state

        if return_optimization_results:
            return results
        else:
            return frac

    evaluate = optimize_fractionation
    __call__ = evaluate

    def __str__(self) -> str:
        """Name of the FractionationOptimizer."""
        return self.__class__.__name__
