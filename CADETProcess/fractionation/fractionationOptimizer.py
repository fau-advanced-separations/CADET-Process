import copy
import warnings
from typing import Callable, Optional

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import (
    COBYLA,
    OptimizationProblem,
    OptimizationResults,
    OptimizerBase,
    compute_initial_radius,
)
from CADETProcess.performance import Mass, Performance, Purity
from CADETProcess.processModel.process import ProcessMeta
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import SolutionIO

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
        chromatograms: list[SolutionIO],
        process_meta: ProcessMeta,
        purity_required: list[float],
        components: Optional[list] = None,
        use_total_concentration_components: bool = True,
        allow_empty_fractions: bool = True,
    ) -> Fractionator:
        """
        Set up the Fractionator for optimizing the fractionation times of Chromatograms.

        Parameters
        ----------
        chromatograms : list of SolutionIO
            Chromatograms to be fractionated.
        process_meta : ProcessMeta
            Process meta information (cycle time, volumes, feed masses).
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
            chromatograms,
            process_meta,
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
        ranking: list[float],
        allow_empty_fractions: bool = True,
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
        ranking : list[float]
            Weighting factors for individual components.
        allow_empty_fractions: bool, optional
            If True, allow empty fractions. The default is True.
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

        Returns
        -------
        OptimizationProblem
            The configured OptimizationProblem object.
        list
            The initial values for the optimization variables.

        Raises
        ------
        CADETProcessError
            If the optimization problem setup fails.
        """
        # Handle empty fractions
        n_fractions = np.array([pool.n_fractions for pool in frac.fraction_pools])
        empty_fractions = np.where(n_fractions[0:-1] == 0)[0]
        if len(empty_fractions) > 0 and allow_empty_fractions:
            purity_required = copy.deepcopy(purity_required)
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

        x0 = []
        for chrom_index, chrom in enumerate(frac.chromatograms):
            chrom_events = frac.chromatogram_events[chrom]
            evt_names = [evt.name for evt in chrom_events]

            if len(evt_names) == 1:
                continue

            for evt in chrom_events:
                opt.add_variable(
                    evt.name,
                    parameter_path=evt.name + ".time",
                    lb=-frac.cycle_time,
                    ub=2 * frac.cycle_time,
                    transform="linear",
                )
                x0.append(evt.time)

            for evt_index, evt in enumerate(chrom_events):
                if evt_index < len(chrom_events) - 1:
                    opt.add_linear_constraint(
                        [evt_names[evt_index], evt_names[evt_index + 1]], [1, -1]
                    )
                else:
                    opt.add_linear_constraint(
                        [evt_names[0], evt_names[-1]], [-1, 1], frac.cycle_time
                    )

        if not opt.check_nonlinear_constraints(x0):
            raise CADETProcessError("No areas found with sufficient purity.")

        return opt, x0

    def optimize_fractionation(
        self,
        simulation_results: SimulationResults | SolutionIO | list[SolutionIO],
        purity_required: float | list[float],
        process_meta: Optional[ProcessMeta] = None,
        components: Optional[list[str]] = None,
        use_total_concentration_components: bool = True,
        ranking: str | list[float] | int = "equal",
        obj_fun: Optional[Callable] = None,
        n_objectives: int = 1,
        bad_metrics: float | list[float] = 0,
        minimize: bool = True,
        allow_empty_fractions: bool = True,
        scale_trust_radius: bool = False,
        ignore_failed: bool = False,
        return_optimization_results: bool = False,
        save_results: bool = False,
        c_min: Optional[float] = None,
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
        ranking : str | list[float] | int, optional, default="equal"
            Weighting factors for individual components.
            If integer, only component of that index is used.
            If None, the same value is assumed for all components.
            The default is None.
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
        scale_trust_radius: bool, optional
            If True, scale initial trust radius depending on linear constraints
            and initial values. The default is False.
        ignore_failed : bool, optional
            Ignore failed optimization and use initial values.
            The default is False.
        return_optimization_results : bool, optional
            If True, return optimization results.
            Otherwise, return fractionation object.
            The default is False.
        save_results : bool, optional
            If True, save optimization results. The default is False.
        c_min : float, optional
            Minimum total concentration below which purity is set to NaN.
            Overrides the per-chromatogram ``c_min`` attribute for all chromatograms
            passed to this call. If None, each chromatogram's existing value is used
            (default 1e-6).

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
        # Resolve chromatograms and process_meta from whatever was passed.
        _process = None
        if isinstance(simulation_results, SimulationResults):
            if len(simulation_results.chromatograms) == 0:
                raise CADETProcessError("Simulation results do not contain chromatogram.")
            chromatograms = simulation_results.chromatograms
            process_meta = simulation_results.process.process_meta
            _process = simulation_results.process  # for lock management
        elif isinstance(simulation_results, SolutionIO):
            chromatograms = [simulation_results]
        elif isinstance(simulation_results, list):
            chromatograms = simulation_results
        else:
            raise TypeError(
                "Expected SimulationResults, SolutionIO, or list of SolutionIO."
            )

        if process_meta is None:
            raise TypeError(
                "process_meta is required when not passing SimulationResults."
            )

        if len(chromatograms) == 0:
            raise CADETProcessError("No chromatograms provided.")

        if c_min is not None:
            for chrom in chromatograms:
                chrom.c_min = c_min

        # Convert inputs to lists of length n_comp
        n_comp = (
            len(components) if components
            else chromatograms[0].component_system.n_comp
        )
        if isinstance(purity_required, float):
            purity_required = n_comp * [purity_required]
        if ranking == "equal":
            ranking = n_comp * [1.0]
        if isinstance(ranking, int):
            index = ranking
            ranking = n_comp * [0.0]
            ranking[index] = 1.0

        # Synchronize zeros between purity_required and ranking
        for i in range(n_comp):
            if purity_required[i] == 0 or ranking[i] == 0:
                purity_required[i] = 0.0
                ranking[i] = 0.0

        # Store previous lock state, unlock to ensure consistent values
        lock_state = _process.lock if _process is not None else None
        if _process is not None:
            _process.lock = False

        frac = self._setup_fractionator(
            chromatograms,
            process_meta,
            purity_required,
            components=components,
            use_total_concentration_components=use_total_concentration_components,
            allow_empty_fractions=allow_empty_fractions,
        )

        opt, x0 = self._setup_optimization_problem(
            frac,
            purity_required,
            ranking=ranking,
            allow_empty_fractions=allow_empty_fractions,
            obj_fun=obj_fun,
            n_objectives=n_objectives,
            minimize=minimize,
            bad_metrics=bad_metrics,
        )

        if scale_trust_radius:
            x0_transformed = opt.transform(x0)
            tr_radius = compute_initial_radius(
                x0_transformed,
                opt.A_transformed,
                opt.n_linear_constraints * [-np.inf],
                opt.b_transformed,
                min_radius=0.01,
            )
            if isinstance(self.optimizer, COBYLA):
                self.optimizer.rhobeg = tr_radius
                self.optimizer.tol = min(0.5 * self.optimizer.rhobeg, self.optimizer.tol)

        # Lock to enable caching
        if _process is not None:
            _process.lock = True

        try:
            results = self.optimizer.optimize(
                opt,
                x0,
                save_results=save_results,
                log_level=self.log_level,
            )
            opt.set_variables(results.x[0])
            frac.reset()
        except (ValueError, CADETProcessError) as e:
            message = (
                f"Optimization failed due to {type(e).__name__}: {str(e)} "
                "Returning initial values."
            )
            if ignore_failed:
                warnings.warn(message)
                frac.initial_values(purity_required)
            else:
                raise CADETProcessError(message)
        finally:
            # Restore previous lock state
            if _process is not None:
                _process.lock = lock_state

        if return_optimization_results:
            return results
        else:
            return frac

    evaluate = optimize_fractionation
    __call__ = evaluate

    def __str__(self) -> str:
        """Name of the FractionationOptimizer."""
        return self.__class__.__name__
