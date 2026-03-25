import copy
import functools
import importlib
from typing import Any, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError, plotting
from CADETProcess.comparison import DifferenceBase
from CADETProcess.dataStructure import String, Structure, get_nested_value
from CADETProcess.numerics import round_to_significant_digits
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import SolutionBase

__all__ = ["Comparator"]


class Comparator(Structure):
    """
    Class for comparing simulation results against reference data.

    Attributes
    ----------
    name : str
        Name of the Comparator instance.
    references : dict
        Dictionary containing the reference data to be compared against.
    solution_paths : dict
        Dictionary containing the solution path for each difference metric.
    metrics : list
        List of difference metrics to be evaluated.
    """

    name = String()

    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a new Comparator instance.

        Parameters
        ----------
        name : Optional[str]
            Name of the Comparator instance.
        """
        self.name = name
        self._metrics = []
        self.references: dict[str, SolutionBase] = {}
        self.solution_paths = {}

    def add_reference(
        self,
        reference: SolutionBase,
        update: Optional[bool] = False,
        smooth: Optional[bool] = False,
    ) -> None:
        """
        Add reference to the Comparator.

        Parameters
        ----------
        reference : SolutionBase
            Reference for comparison with SimulationResults.
        update : Optional[bool], default=False
            If True, update existing reference.
        smooth : Optional[bool], default=False
            If True, smooth data before comparison.

        Raises
        ------
        TypeError
            If reference is not an instance of SolutionBase.
        CADETProcessError
            If Reference already exists.

        """
        if not isinstance(reference, SolutionBase):
            raise TypeError("Expeced SolutionBase")

        if reference.name in self.references and not update:
            raise CADETProcessError("Reference already exists")

        reference = copy.deepcopy(reference)
        if smooth:
            reference.smooth_data()

        self.references[reference.name] = reference

    @property
    def metrics(self) -> list[DifferenceBase]:
        """list[DifferenceBase]: List of difference metrics."""
        return self._metrics

    @property
    def n_difference_metrics(self) -> int:
        """int: Number of difference metrics in the Comparator."""
        return len(self.metrics)

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics to be evaluated."""
        return sum([metric.n_metrics for metric in self.metrics])

    @property
    def bad_metrics(self) -> list[float]:
        """list[float]: Worst case metrics for all difference metrics."""
        bad_metrics = [metric.bad_metrics for metric in self.metrics]

        return np.hstack(bad_metrics).flatten().tolist()

    @property
    def labels(self) -> list[str]:
        """list[str]: List of metric labels."""
        labels = []
        for metric in self.metrics:
            try:
                metric_labels = metric.labels
            except AttributeError:
                metric_labels = [f"{metric}"]
                if metric.n_metrics > 1:
                    metric_labels = [f"{metric}_{i}" for i in range(metric.n_metrics)]

            if len(metric_labels) != metric.n_metrics:
                raise CADETProcessError(f"Must return {metric.n_labels} labels.")

            labels += metric_labels

        return labels

    @functools.wraps(DifferenceBase.__init__)
    def add_difference_metric(
        self,
        difference_metric: str,
        reference: str | SolutionBase,
        solution_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> DifferenceBase:
        """
        Add a difference metric to the Comparator.

        Parameters
        ----------
        difference_metric : str
            Name of the difference metric to be evaluated.
        reference : str | SolutionBase
            Name of the reference or reference itself.
        solution_path : str
            Path to the solution in SimulationResults.
        *args, **kwargs
            Additional arguments and keyword arguments to be passed to the
            difference metric constructor.

        Returns
        -------
        DifferenceBase
            The new difference metric instance.

        Raises
        ------
        CADETProcessError
            If the difference metric or reference is unknown.
        """
        try:
            module = importlib.import_module("CADETProcess.comparison.difference")
            cls_ = getattr(module, difference_metric)
        except KeyError:
            raise CADETProcessError("Unknown Metric Type.")

        if isinstance(reference, SolutionBase):
            reference = reference.name

        if reference not in self.references:
            raise CADETProcessError("Unknown Reference.")

        reference = self.references[reference]

        metric = cls_(reference, *args, **kwargs)

        self.solution_paths[metric] = solution_path

        self._metrics.append(metric)

        return metric

    def extract_solution(
        self,
        simulation_results: SimulationResults,
        metric: DifferenceBase,
    ) -> SolutionBase:
        """
        Extract the solution for a given metric from the SimulationResults object.

        Parameters
        ----------
        simulation_results : SimulationResults
            The SimulationResults object containing the solution.
        metric : DifferenceBase
            The difference metric object for which to extract the solution.

        Returns
        -------
        SolutionBase
            The solution for the given metric.

        Raises
        ------
        CADETProcessError
            If the solution path for the given metric is not found.
        """
        try:
            solution_path = self.solution_paths[metric]
            solution = get_nested_value(simulation_results.solution_cycles, solution_path)[-1]
        except KeyError:
            raise CADETProcessError("Could not find solution path")

        return copy.deepcopy(solution)

    def evaluate(self, simulation_results: SimulationResults) -> list[float]:
        """
        Evaluate all metrics for a given simulation and return the results as a list.

        Parameters
        ----------
        simulation_results : SimulationResults
            The SimulationResults object containing the solutions for all metrics.

        Returns
        -------
        list[float]
            A list containing the values of all difference of metrics after comparison.
        """
        metrics = []
        for metric in self.metrics:
            solution = self.extract_solution(simulation_results, metric)
            m = metric.evaluate(solution)
            metrics.append(m)

        metrics = np.hstack(metrics).tolist()

        return metrics

    __call__ = evaluate

    @plotting.figure_utils
    def plot_comparison(
        self,
        simulation_results: SimulationResults,
        x_axis_in_minutes: Optional[bool] = True,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """
        Plot the comparison of the simulation results with the reference data.

        Parameters
        ----------
        simulation_results : SimulationResults
            Simulation results to compare to reference data.
        x_axis_in_minutes: Optional[bool], default=True
            If True, the x-axis will be plotted using minutes. The default is True.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.

        Returns
        -------
        tuple
            A tuple containing:
            - list[plt.Figure]: A list of Matplotlib Figure objects.
            - npt.NDArray[plt.Axes]: An array of Axes objects with one Axes per
              difference metric.
        """
        if self.n_difference_metrics == 0:
            raise CADETProcessError("Cannot plot without difference metrics.")

        if ax is None:
            fig, axs = plotting.setup_figure(
                **setup_figure_kwargs,
                nrows=self.n_difference_metrics,
                squeeze=False,
            )
            axs = axs.reshape(-1)
        else:
            axs = ax
            fig = axs[0].get_figure()

        for ax, metric in zip(axs, self.metrics):
            ax.set_title(metric.reference.name)

            solution = self.extract_solution(simulation_results, metric)
            solution_sliced = metric.slice_and_transform(solution)

            solution_sliced.plot(
                ax=ax,
                x_axis_in_minutes=x_axis_in_minutes,
                tight_layout=False,
            )

            ref_time = metric.reference.time
            if x_axis_in_minutes:
                ref_time = ref_time / 60

            ax.plot(
                ref_time,
                metric.reference.solution,
                linestyle="--",
            )
            ax.legend(loc=1)

            m = metric.evaluate(solution_sliced, slice=False)
            m = round_to_significant_digits(m, digits=2)

            text = f"{metric}: "
            if metric.n_metrics > 1:
                try:
                    text += "\n"
                    for i, (label, m) in enumerate(zip(metric.labels, m)):
                        text += f"{label}: ${m}$"
                        if i < metric.n_metrics - 1:
                            text += " \n"
                except AttributeError:
                    text += f"{m}"
            else:
                text += str(m[0])

            plotting.add_text(ax, text)

        return fig, axs

    def __iter__(self) -> Iterator[list[DifferenceBase]]:
        """Yield metrics from the instance."""
        yield from self.metrics

    def __str__(self) -> str:
        """str: Name of the Comparator."""
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__
