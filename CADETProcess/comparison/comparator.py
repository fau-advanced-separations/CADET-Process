import copy
import importlib
import functools

import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.dataStructure import StructMeta, String
from CADETProcess.dataStructure import get_nested_value
from CADETProcess.solution import SolutionBase
from CADETProcess.comparison import DifferenceBase


class Comparator(metaclass=StructMeta):
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

    def __init__(self, name=None):
        """Initialize a new Comparator instance.

        Parameters
        ----------
        name : str, optional
            Name of the Comparator instance.
        """
        self.name = name
        self._metrics = []
        self.references = {}
        self.solution_paths = {}

    def add_reference(self, reference, update=False, smooth=False):
        """Add Reference to Comparator.

        Parameters
        ----------
        reference : ReferenceIO
            Reference for comparison with SimulationResults.
        update : bool, optional
            If True, update existing reference. The default is False.
        smooth : bool, optional
            If True, smooth data before comparison. The default is False.

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
    def metrics(self):
        """list: List of difference metrics."""
        return self._metrics

    @property
    def n_metrics(self):
        """int: Number of metrics to be evaluated."""
        return sum([metric.n_metrics for metric in self.metrics])

    @property
    def bad_metrics(self):
        """list: Worst case metrics for all difference metrics."""
        bad_metrics = [metric.bad_metrics for metric in self.metrics]

        return np.hstack(bad_metrics).flatten().tolist()

    @property
    def labels(self):
        """list: List of metric labels."""
        labels = []
        for metric in self.metrics:
            try:
                metric_labels = metric.labels
            except AttributeError:
                metric_labels = [f'{metric}']
                if metric.n_metrics > 1:
                    metric_labels = [
                        f'{metric}_{i}' for i in range(metric.n_metrics)
                    ]

            if len(metric_labels) != metric.n_metrics:
                raise CADETProcessError(
                    f"Must return {metric.n_labels} labels."
                )

            labels += metric_labels

        return labels

    @functools.wraps(DifferenceBase.__init__)
    def add_difference_metric(
            self, difference_metric, reference, solution_path,
            *args, **kwargs):
        """Add a difference metric to the Comparator.

        Parameters
        ----------
        difference_metric : str
            Name of the difference metric to be evaluated.
        reference : str or SolutionBase
            Name of the reference or reference itself.
        solution_path : str
            Path to the solution in SimulationResults.
        *args, **kwargs
            Additional arguments and keyword arguments to be passed to the
            difference metric constructor.

        Raises
        ------
        CADETProcessError
            If the difference metric or reference is unknown.
        """
        try:
            module = importlib.import_module(
                'CADETProcess.comparison.difference'
            )
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

    def extract_solution(self, simulation_results, metric):
        """Extract the solution for a given metric from the SimulationResults object.

        Parameters
        ----------
        simulation_results : SimulationResults
            The SimulationResults object containing the solution.
        metric : Metric
            The Metric object for which to extract the solution.

        Returns
        -------
        numpy.ndarray
            The solution array for the given metric.

        Raises
        ------
        CADETProcessError
            If the solution path for the given metric is not found.
        """
        try:
            solution_path = self.solution_paths[metric]
            solution = get_nested_value(
                simulation_results.solution_cycles, solution_path
            )[-1]
        except KeyError:
            raise CADETProcessError("Could not find solution path")

        return solution

    def evaluate(self, simulation_results):
        """Evaluate all metrics for a given simulation and return the results as a list.

        Parameters
        ----------
        simulation_results : SimulationResults
            The SimulationResults object containing the solutions for all metrics.

        Returns
        -------
        list
            A list of metric evaluation results, where each element is a numpy array.
        """
        metrics = []
        for metric in self.metrics:
            solution = self.extract_solution(simulation_results, metric)
            m = metric.evaluate(solution)
            metrics.append(m)

        metrics = np.hstack(metrics).tolist()

        return metrics

    __call__ = evaluate

    def setup_comparison_figure(self, plot_individual=False):
        """Set up a figure for comparing simulation results.

        Parameters
        ----------
        plot_individual : bool, optional
            If True, return figures for individual metrics.
            Otherwise, return a single figure for all metrics.
            Default is False.

        Returns
        -------
        tuple
            A tuple of the comparison figure(s) and axes object(s).
        """
        n = len(self.metrics)

        if n == 0:
            return (None, None)

        comparison_fig_all, comparison_axs_all = plt.subplots(
            nrows=n,
            figsize=(8 + 4 + 2, n*8 + 2),
            squeeze=False
        )
        plt.close(comparison_fig_all)
        comparison_axs_all = comparison_axs_all.reshape(-1)

        comparison_fig_ind = []
        comparison_axs_ind = []
        for i in range(n):
            fig, axs = plt.subplots()
            comparison_fig_ind.append(fig)
            comparison_axs_ind.append(axs)
            plt.close(fig)

        comparison_axs_ind = \
            np.array(comparison_axs_ind).reshape(comparison_axs_all.shape)

        if plot_individual:
            return comparison_fig_ind, comparison_axs_ind
        else:
            return comparison_fig_all, comparison_axs_all

    def plot_comparison(
            self, simulation_results, axs=None, figs=None,
            file_name=None, show=True, plot_individual=False):
        """Plot the comparison of the simulation results with the reference data.

        Parameters
        ----------
        simulation_results : list of SimulationResults
            List of simulation results to compare to reference data.
        axs : list of AxesSubplot, optional
            List of subplot axes to use for plotting the metrics.
        figs : list of Figure, optional
            List of figures to use for plotting the metrics.
        file_name : str, optional
            Name of the file to save the figure to.
        show : bool, optional
            If True, displays the figure(s) on the screen.
        plot_individual : bool, optional
            If True, generates a separate figure for each metric.

        Returns
        -------
        figs : list of Figure
            List of figures used for plotting the metrics.
        axs : list of AxesSubplot
            List of subplot axes used for plotting the metrics.
        """
        if axs is None:
            figs, axs = self.setup_comparison_figure(plot_individual)
        if not isinstance(figs, list):
            figs = [figs]

        for ax, metric in zip(axs, self.metrics):
            solution = self.extract_solution(simulation_results, metric)
            if metric.normalize:
                solution.normalize()
            if metric.smooth:
                solution.smooth_data()
            solution_sliced = metric.slice_and_transform(solution)

            fig, ax = solution_sliced.plot(
                ax=ax,
                show=False,
                y_max=1.1*np.max(metric.reference.solution)
            )

            plot_args = {
                'linestyle': 'dotted',
                'color': 'k',
                'label': 'reference',
            }
            plotting.add_overlay(
                ax, metric.reference.solution, metric.reference.time/60,
                **plot_args
            )
            ax.legend(loc=1)

            m = metric.evaluate(solution_sliced, slice=False)
            m = [
                np.format_float_scientific(
                    n, precision=2,
                )
                for n in m
            ]

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
                text += m[0]

            plotting.add_text(ax, text, fontsize=14)

        for fig in figs:
            fig.tight_layout()
            if not show:
                plt.close(fig)
            else:
                dummy = plt.figure(figsize=fig.get_size_inches())
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)
                plt.show()

        if file_name is not None:
            if plot_individual:
                name, suffix = file_name.split('.')
                for fig, metric in zip(figs, self.metrics):
                    fig.savefig(f'{name}_{metric}.{suffix}')
            else:
                figs[0].savefig(file_name)

        return figs, axs

    def __iter__(self):
        yield from self.metrics

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__
