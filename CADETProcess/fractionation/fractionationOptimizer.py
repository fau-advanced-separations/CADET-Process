import warnings

import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess import SimulationResults
from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import OptimizerBase, OptimizationProblem
from CADETProcess.optimization import COBYLA
from CADETProcess.performance import Mass, Purity


class FractionationEvaluator():
    """Dummy Evaluator to enable caching."""

    def evaluate(self, fractionator):
        return fractionator.performance

    __call__ = evaluate

    def __str__(self):
        return __class__.__name__


class FractionationOptimizer():
    """Configuration for fractionating Chromatograms.

    Attributes
    ----------
    optimizer: OptimizerBase
        Optimizer for optimizing the fractionaton times.

    """

    def __init__(self, optimizer=None, log_level='WARNING', save_log=False):
        if optimizer is None:
            optimizer = COBYLA(log_level=log_level, save_log=save_log)
            optimizer.tol = 0.1
            optimizer.catol = 1
            optimizer.rhobeg = 1
        self.optimizer = optimizer
        self.log_level = log_level
        self.save_log = save_log

    @property
    def optimizer(self):
        """OptimizerBase: Optimizer for optimizing the fractionation times."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not isinstance(optimizer, OptimizerBase):
            raise TypeError('Expected OptimizerBase')
        self._optimizer = optimizer

    def setup_fractionator(self, simulation_results, purity_required):
        frac = Fractionator(simulation_results)

        frac.process.lock = False

        frac.initial_values(purity_required)

        if len(frac.events) == 0:
            raise CADETProcessError("No areas found with sufficient purity.")

        frac.process.lock = True
        return frac

    def setup_optimization_problem(
            self,
            frac,
            purity_required,
            ranking=1,
            obj_fun=None,
            n_objectives=1):
        opt = OptimizationProblem(
            'FractionationOptimization',
            log_level=self.log_level,
            save_log=self.save_log
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
            opt.add_variable(evt.name + '.time', name=evt.name)

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

        opt.x0 = [evt.time for evt in frac.events]

        if not opt.check_nonlinear_constraints(opt.x0):
            raise CADETProcessError("No areas found with sufficient purity.")

        return opt

    def optimize_fractionation(
            self,
            simulation_results,
            purity_required,
            ranking=1,
            obj_fun=None,
            n_objectives=1,
            ignore_failed=True,
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
        ignore_failed : bool, optional
            Ignore failed optimization and use initial values.
            The default is True.
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
            If simulation_results is not an instance of SimulationResult.
        CADETProcessError
            If simulation_results do not contain chromatograms
        Warning
            If purity requirements cannot be fulfilled.

        See Also
        --------
        CADETProcess.common.Chromatogram
        setup_fractionator
        Fractionator
        setup_optimization_problem
        CADETProcess.optimization.OptimizationProblem

        """
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError('Expected SimulationResults')

        if len(simulation_results.chromatograms) == 0:
            raise CADETProcessError(
                'Simulation results do not contain chromatogram'
            )

        frac = self.setup_fractionator(simulation_results, purity_required)

        try:
            opt = self.setup_optimization_problem(
                frac, purity_required, ranking, obj_fun, n_objectives
            )
            opt_results = self.optimizer.optimize(opt, save_results=False)
        except CADETProcessError as e:
            if ignore_failed:
                warnings.warn('Optimization failed. Returning initial values')
                frac.initial_values(purity_required)
            else:
                raise CADETProcessError(str(e))

        if return_optimization_results:
            return opt_results
        else:
            return frac

    evaluate = optimize_fractionation
    __call__ = evaluate

    def __str__(self):
        return self.__class__.__name__
