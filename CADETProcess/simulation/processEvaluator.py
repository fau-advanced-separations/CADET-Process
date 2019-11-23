from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta

from CADETProcess.processModel import Process
from CADETProcess.simulation import SolverBase

from CADETProcess.optimization import mass, ranked_objective_decorator
from CADETProcess.fractionation import optimize_fractionation

from CADETProcess.common import get_bad_performance

class ProcessEvaluator(metaclass=StructMeta):
    """Wrapper for sequential simulation and fractionation of processes.

    Attributes
    ----------
    solver : Solver
        solver with stationarity configuration.

    See also
    --------
    Process
    simulation.SolverBase
    fractionation.optimize_fractionation
    """
    def __init__(self, solver, purity_required=0.95, ranking=1):
        self.solver = solver
        self.purity_required = purity_required
        self.ranking = ranking

    @property
    def solver(self):
        """Returns

        Parameters
        ----------
        solver: Solver
            Solver with interface and stationarity configuration.
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        if not isinstance(solver, SolverBase):
            raise TypeError('Expected Solver')
        self._solver = solver

    def evaluate(self, process, return_frac=False):
        """Runs the process simulation and calls the fractionation optimization.

        Parameters
        ----------
        process : Process
            Process to be simulated

        Raises
        ------
        TypeError
            If process is not an instance of Process

        Returns
        -------
        performance : Performance
            Process performance after fractionation with obj_fun and nonlin_fun
        """
        if not isinstance(process, Process):
            raise TypeError('Expected Process')

        try:
            results = self.solver.simulate(process)
        except CADETProcessError:
            return get_bad_performance(process.n_comp)

        n_comp = process.flow_sheet.n_comp
        if isinstance(self.purity_required, float):
            purity_required = [self.purity_required] * n_comp
        elif isinstance(self.purity_required, list) and \
            len(self.purity_required) != n_comp:
            raise CADETProcessError('Number of components don\' match')
        else:
            purity_required = self.purity_required

        if isinstance(self.ranking, float):
            ranking = [self.ranking] * n_comp
        elif isinstance(self.ranking, list) and \
            len(self.ranking) != n_comp:
            raise CADETProcessError('Number of components don\' match')
        else:
            ranking = self.ranking

        obj_fun = ranked_objective_decorator(ranking)(mass)
        frac = optimize_fractionation(
                results.chromatograms, process.process_meta,
                purity_required, obj_fun)

        if return_frac:
             return frac
        else:
             return frac.performance
