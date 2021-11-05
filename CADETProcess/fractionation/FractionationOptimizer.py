import logging
import warnings

from CADETProcess import CADETProcessError

from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import SolverBase, OptimizationProblem
from CADETProcess.optimization import COBYLA, TrustConstr
from CADETProcess.optimization import mass, ranked_objective_decorator
from CADETProcess.optimization import purity, nonlin_bounds_decorator


class FractionationOptimizer():
    """Configuration for fractionating Chromatograms.
    
    Attributes
    ----------
    optimizer: SolverBase
     Optimizer for optimizing the fractionaton times.
    purity_required :  float or array_like
        Minimum required purity for components. If is float, the same 
        value is assumed for all components.
    obj_fun : function, optional
        Objective function used for OptimizationProblem. If is None, the 
        mass of all components is maximized.
    """
    def __init__(self, purity_required, obj_fun=None, optimizer=None):
        self.purity_required = purity_required
        if obj_fun is None:
            obj_fun = ranked_objective_decorator(1)(mass)
        self.obj_fun = obj_fun
        if optimizer is None:
            optimizer = COBYLA()
            optimizer.tol = 0.1
            optimizer.catol = 1
            optimizer.rhobeg = 1
        self.optimizer = optimizer
        
    @property
    def optimizer(self):
        """SolverBase: Optimizer for optimizing the fractionation times.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not isinstance(optimizer, SolverBase):
            raise TypeError('Optimization SolverBase')
        self._optimizer = optimizer
        
    def setup_fractionator(self, process_meta, chromatograms):
        frac = Fractionator(process_meta)
    
        for chrom in chromatograms:
            frac.add_chromatogram(chrom)
    
        frac.initial_values(self.purity_required)
        
        if len(frac.events) == 0:
            raise CADETProcessError("No areas found with sufficient purity.")
        return frac
        
    def setup_optimization_problem(self, frac):
        opt = OptimizationProblem(frac)
        opt.logger.setLevel(logging.WARNING)
    
        opt.add_objective(self.obj_fun)
        opt.add_nonlinear_constraint(
            nonlin_bounds_decorator(self.purity_required)(purity)
        )
    
        for evt in frac.events:
            opt.add_variable(evt.name + '.time', evt.name)
    
        for chrom_index, chrom in enumerate(frac.chromatograms):
            chrom_events = frac.chromatogram_events[chrom]
            evt_names = [evt.name for evt in chrom_events]
            for evt_index, evt in enumerate(chrom_events):
                if evt_index < len(chrom_events) - 1:
                    opt.add_linear_constraint(
                        [evt_names[evt_index], evt_names[evt_index+1]], [1,-1]
                    )
                else:
                    opt.add_linear_constraint(
                        [evt_names[0], evt_names[-1]],[-1,1], frac.cycle_time
                    )
    
        opt.x0 = [evt.time for evt in frac.events]
        
        if not opt.check_nonlinear_constraints(opt.x0):
            raise CADETProcessError("No areas found with sufficient purity.")
        
        return opt
        

    def optimize_fractionation(self, chromatograms, process_meta):
        """Optimize the fractionation times w.r.t. purity constraints.
    
        Parameters
        ----------
        chromatograms : Chromatogram or list of Chromatograms
            Chromatogram to be fractionated
        process_meta : ProcessMeta
            Metainformation of the Process
    
        Returns
        -------
        performance : Performance
            FractionationPerformance    
    
        Raises
        -------
        TypeError
            If chromatogram is not an instance of Chromatogram.
        Warning
            If purity requirements cannot be fulfilled.
    
    
        See Also
        --------
        Chromatogram
        setup_fractionator
        Fractionator
        setup_optimization_problem
        OptimizationProblem
        """
        if not isinstance(chromatograms, list):
            chromatograms = [chromatograms]
        
        if (not isinstance(self.purity_required, (float, int)) 
            and chromatograms[0].n_comp != len(self.purity_required)):
            raise CADETProcessError('Number of components does not match.')
        
        frac = self.setup_fractionator(process_meta, chromatograms)
        
        try:
            opt = self.setup_optimization_problem(frac)
            opt_results = self.optimizer.optimize(opt)
        except CADETProcessError:
            warnings.warn('Optimization failed. Returning initial values')
            frac.initial_values()
    
        return frac
