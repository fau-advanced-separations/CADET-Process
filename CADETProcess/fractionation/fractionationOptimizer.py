import warnings

import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess import SimulationResults
from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import OptimizerBase, OptimizationProblem
from CADETProcess.optimization import COBYLA
from CADETProcess.performance import Mass, Purity


__all__ = ['FractionationOptimizer']


class FractionationEvaluator():
    """Dummy Evaluator to enable caching."""

    def evaluate(self, fractionator):
        """Evaluate the fractionator.

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

    def __str__(self):
        return __class__.__name__


class FractionationOptimizer():
    """Configuration for fractionating Chromatograms."""

    def __init__(self, optimizer=None, log_level='WARNING'):
        """Initialize the FractionationOptimizer.

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
            optimizer.tol = 0.1
            optimizer.catol = 1e-4
            optimizer.rhobeg = 5e-4
        self.optimizer = optimizer
        self.log_level = log_level

    @property
    def optimizer(self):
        """OptimizerBase: Optimizer for optimizing the fractionation times."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Set the optimizer.

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
            raise TypeError('Expected OptimizerBase')
        self._optimizer = optimizer

    def _setup_fractionator(
            self,
            simulation_results,
            purity_required,
            components=None,
            use_total_concentration_components=True,
            allow_empty_fractions=True):
        """Set up the Fractionator for optimizing the fractionation times of Chromatograms.

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

        Raises
        ------
        CADETProcessError
            If no areas with sufficient purity were found and `ignore_failed` is False.
        """
        frac = Fractionator(
            simulation_results,
            components=components,
            use_total_concentration_components=use_total_concentration_components,
        )

        frac.initial_values(purity_required)

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
            frac,
            purity_required,
            ranking=1,
            obj_fun=None,
            n_objectives=1):
        """Set up the OptimizationProblem for optimizing the fractionation times.

        Parameters
        ----------
        frac : Fractionator
            The Fractionator object.
        purity_required : list
            Minimum purity required for the components in the fractionation.
        ranking : {float, list, None}
            Weighting factors for individual components.
            If float, the same value is used for all components.
            If None, no ranking is used and the problem is solved as multi-objective.
        obj_fun : callable, optional
            Alternative objective function.
            If no function is provided, the fraction mass is maximized.
            The default is None.
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
        opt = OptimizationProblem(
            'FractionationOptimization',
            log_level=self.log_level,
            use_diskcache=False,
        )

        opt.add_evaluation_object(frac)

        frac_evaluator = FractionationEvaluator()
        opt.add_evaluator(frac_evaluator, cache=True)

        if obj_fun is None:
            obj_fun = Mass(ranking=ranking)
        opt.add_objective(
            obj_fun, requires=frac_evaluator, n_objectives=n_objectives,
            bad_metrics=0
        )

        purity = Purity()
        purity.n_metrics = frac.component_system.n_comp
        constraint_bounds = -np.array(purity_required, ndmin=1)
        constraint_bounds = constraint_bounds.tolist()
        opt.add_nonlinear_constraint(
            purity, n_nonlinear_constraints=len(constraint_bounds),
            bounds=constraint_bounds, requires=frac_evaluator
        )

        for evt in frac.events:
            opt.add_variable(
                evt.name, parameter_path=evt.name + '.time',
                lb=-frac.cycle_time, ub=2*frac.cycle_time,
                transform='linear'
            )

        for chrom_index, chrom in enumerate(frac.chromatograms):
            chrom_events = frac.chromatogram_events[chrom]
            evt_names = [evt.name for evt in chrom_events]
            for evt_index, evt in enumerate(chrom_events):
                if evt_index < len(chrom_events) - 1:
                    opt.add_linear_constraint(
                        [evt_names[evt_index], evt_names[evt_index+1]], [1, -1]
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
            simulation_results,
            purity_required,
            components=None,
            use_total_concentration_components=True,
            ranking=1,
            obj_fun=None,
            n_objectives=1,
            allow_empty_fractions=True,
            ignore_failed=False,
            return_optimization_results=False,
            save_results=False):
        """Optimize the fractionation times with respect to purity constraints.

        Parameters
        ----------
        simulation_results : SimulationResults
            Results containing the chromatograms for fractionation.
        purity_required :  float or array_like
            Minimum required purity for components. If is float, the same
            value is assumed for all components.
        ranking : float or array_like
            Relative value of components.
        obj_fun : function, optional
            Objective function used for OptimizationProblem.
            If COBYLA is used, must return single objective.
            If is None, the mass of all components is maximized.
        n_objectives : int, optional
            Number of objectives returned by obj_fun. The default is 1.
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
            raise TypeError('Expected SimulationResults.')

        if len(simulation_results.chromatograms) == 0:
            raise CADETProcessError(
                'Simulation results do not contain chromatogram.'
            )

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
            allow_empty_fractions=allow_empty_fractions
        )

        # Lock to enable caching
        simulation_results.process.lock = True

        try:
            opt, x0 = self._setup_optimization_problem(
                frac, purity_required, ranking, obj_fun, n_objectives
            )
            results = self.optimizer.optimize(
                opt, x0,
                save_results=save_results,
                log_level=self.log_level,
                delete_cache=True,
            )
        except CADETProcessError as e:
            if ignore_failed:
                warnings.warn('Optimization failed. Returning initial values')
                frac.initial_values(purity_required)
            else:
                raise CADETProcessError(str(e))

        opt.set_variables(results.x[0])
        frac.reset()

        # Restore previous lock state
        simulation_results.process.lock = lock_state

        if return_optimization_results:
            return results
        else:
            return frac

    evaluate = optimize_fractionation
    __call__ = evaluate

    def __str__(self):
        return self.__class__.__name__
