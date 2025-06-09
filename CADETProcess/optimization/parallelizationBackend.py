import multiprocessing
from abc import abstractmethod
from collections.abc import Iterable

from CADETProcess.dataStructure import (
    Constant,
    RangedInteger,
    Structure,
    UnsignedInteger,
)

cpu_count = multiprocessing.cpu_count()

__all__ = ["ParallelizationBackendBase", "SequentialBackend"]


class ParallelizationBackendBase(Structure):
    """
    Base class for all parallelization backend adapters.

    Attributes
    ----------
    n_cores : int
        Number of cores to be used. If set to 0 or -1, all available cores are used.
        For values less than -1, (n_cpus + 1 + n_cores) are used.
        For example, for n_cores = -2, all CPUs but one are used.
    """

    n_cores = RangedInteger(lb=-cpu_count, ub=cpu_count, default=1)

    _parameters = ["n_cores"]

    @property
    def _n_cores(self) -> int:
        if self.n_cores == 0:
            return cpu_count

        if self.n_cores < 0:
            return cpu_count + 1 + self.n_cores
        else:
            return self.n_cores

    @abstractmethod
    def evaluate(self, function: callable, population: Iterable) -> list:
        """
        Evaluate the function at all individuals in the population-list.

        This method must be implemented by subclasses to evaluate the provided function
        at each individual in the given population. The evaluation can be performed
        sequentially or in parallel, depending on the concrete implementation.

        Parameters
        ----------
        function : callable
            The function of interest to be evaluated for the population.
        population : Iterable
            A collection of individuals at which the function is to be evaluated.

        Returns
        -------
        list
            List of results of function evaluations.
        """
        pass

    def __str__(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__


class SequentialBackend(ParallelizationBackendBase):
    """
    Sequential execution backend for evaluating the target function.

    This backend does not perform parallelization. It evaluates the function at each
    individual in the population sequentially, one by one.

    Attributes
    ----------
    n_cores : int
        Number of cores to be used. This attribute is constant and always set to 1
        for the 'SequentialBackend'.
    """

    n_cores = Constant(value=1)

    def evaluate(self, function: callable, population: Iterable) -> list:
        """
        Evaluate the function sequentially at all individuals in the population.

        Since this is the 'SequentialBackend', the evaluation is done sequentially
        without any parallelization.

        Parameters
        ----------
        function : callable
            The function of interest to be evaluated for the population.
        population : Iterable
            A collection of individuals at which the function is to be evaluated.

        Returns
        -------
        list
            List of results of function evaluations.
        """
        results = []
        for ind in population:
            results.append(function(ind))
        return results


try:
    from joblib import Parallel, delayed

    __all__.append("Joblib")
except ModuleNotFoundError:
    pass


class Joblib(ParallelizationBackendBase):
    """
    Parallelization backend implementation using joblib.

    Attributes
    ----------
    verbose : int
        The verbosity level: if nonzero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it's more than 10, all iterations are reported.
    """

    verbose = UnsignedInteger(default=0)
    _parameters = ["verbose"]

    def evaluate(self, function: callable, population: Iterable) -> list:
        """
        Evaluate the function in parallalel for all individuals of the population.

        Parameters
        ----------
        function : callable
            The function of interest to be evaluated for the population.
        population : Iterable
            A collection of individuals at which the function is to be evaluated.

        Returns
        -------
        list
            List of results of function evaluations.
        """
        backend = Parallel(n_jobs=self.n_cores, verbose=self.verbose)
        results = backend(delayed(function)(x) for x in population)

        return results


try:
    import pathos

    __all__.append("Pathos")
except ModuleNotFoundError:
    pass


class Pathos(ParallelizationBackendBase):
    """Parallelization backend using the pathos library."""

    def evaluate(self, function: callable, population: Iterable) -> list:
        """
        Evaluate the function in parallalel for all individuals of the population.

        Parameters
        ----------
        function : callable
            The function of interest to be evaluated for the population.
        population : Iterable
            A collection of individuals at which the function is to be evaluated.

        Returns
        -------
        list
            List of results of function evaluations.
        """
        with pathos.pools.ProcessPool(ncpus=self.n_cores) as pool:
            results = pool.map(function, population)
        return results
