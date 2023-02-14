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
        return fractionator.performance

    __call__ = evaluate

    def __str__(self):
        return __class__.__name__


class FractionationOptimizer():
    """Configuration for fractionating Chromatograms."""

    def __init__(self, optimizer=None, log_level='WARNING'):
        """Initialize fractionation optimizer.

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
            optimizer.catol = 1e-3
            optimizer.rhobeg = 1
        self.optimizer = optimizer
        self.log_level = log_level

    @property
    def optimizer(self):
        """OptimizerBase: Optimizer for optimizing the fractionation times."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not isinstance(optimizer, OptimizerBase):
            raise TypeError('Expected OptimizerBase')
        self._optimizer = optimizer

    def setup_fractionator(
            self,
            simulation_results,
            purity_required,
            components=None,
            use_total_concentration_components=True,
            allow_empty_fractions=True):
        """Set up Fractionator for optimizing the fractionation times of Chromatograms.

        Parameters
        ----------
        simulation_results: object
            Simulation results to be used for setting up the Fractionator object.
        purity_required: float
            Minimum purity required for the fractionation process.
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

        frac.process.lock = False
        frac.initial_values(purity_required)
        frac.process.lock = True

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

    def setup_optimization_problem(
            self,
            frac,
            purity_required,
            ranking=1,
            obj_fun=None,
            n_objectives=1):
        """Set up OptimizationProblem for optimizing the fractionation times.

        Parameters
        ----------
        frac : Fractionator
            DESCRIPTION.
        purity_required : {list, float}
            Minimum purity required.
            If float, same value will be used for all components.
        ranking : {float, list, None}
            Weighting factors for individual components.
            If float, same value is usued for all components.
            If None, no rankining is used and the problem is solved as multi-objective.
        obj_fun : callable, optional
            Alternative objective function.
            If no function is provided, the fractiton mass is maximized.
            The default is None.
        n_objectives : int
            Number of objectives. The default is 1.

        Raises
        ------
        CADETProcessError
            DESCRIPTION.

        Returns
        -------
        opt : TYPE
            DESCRIPTION.

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
        constraint_bounds = -np.array(purity_required)
        constraint_bounds = constraint_bounds.tolist()
        opt.add_nonlinear_constraint(
            purity, n_nonlinear_constraints=len(constraint_bounds),
            bounds=constraint_bounds, requires=frac_evaluator
        )

        for evt in frac.events:
            opt.add_variable(evt.name, parameter_path=evt.name + '.time')

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
            return_optimization_results=False):
        """Optimize the fractionation times w.r.t. purity constraints.

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

        Returns
        -------
        frac : Fractionation
            Fractionation object with optimized cut times.

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
        setup_fractionator
        setup_optimization_problem
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

        frac = self.setup_fractionator(
            simulation_results,
            purity_required,
            components=components,
            use_total_concentration_components=use_total_concentration_components,
            allow_empty_fractions=allow_empty_fractions
        )

        try:
            opt, x0 = self.setup_optimization_problem(
                frac, purity_required, ranking, obj_fun, n_objectives
            )
            results = self.optimizer.optimize(
                opt, x0,
                save_results=False,
                log_level=self.log_level,
                delete_cache=True,
            )
        except CADETProcessError as e:
            if ignore_failed:
                warnings.warn('Optimization failed. Returning initial values')
                frac.initial_values(purity_required)
            else:
                raise CADETProcessError(str(e))

        frac = opt.set_variables(results.x[0])[0]

        if return_optimization_results:
            return results
        else:
            return frac

    evaluate = optimize_fractionation
    __call__ = evaluate

    def __str__(self):
        return self.__class__.__name__
