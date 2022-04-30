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
    name = String()

    def __init__(self, name=None):
        self.name = name
        self._metrics = []
        self.references = {}
        self.solution_paths = {}

    def add_reference(self, reference, update=False, smooth=True):
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
        return self._metrics

    @property
    def n_metrics(self):
        return sum([metric.n_metrics for metric in self.metrics])

    @property
    def bad_metrics(self):
        bad_metrics = [metric.bad_metrics for metric in self.metrics]

        return np.hstack(bad_metrics).flatten().tolist()

    @functools.wraps(DifferenceBase.__init__)
    def add_difference_metric(
            self, difference_metric, reference, solution_path,
            *args, **kwargs):
        try:
            module = importlib.import_module('CADETProcess.comparison')
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

    def evaluate(self, simulation_results, smooth=True):
        metrics = []
        for metric in self.metrics:
            try:
                solution_path = self.solution_paths[metric]
                solution = get_nested_value(
                    simulation_results.solution_cycles, solution_path
                )[-1]
            except KeyError:
                raise CADETProcessError("Could not find solution path")

            solution = copy.deepcopy(solution)
            if smooth:
                solution.smooth_data(
                    metric.reference.s,
                    metric.reference.crit_fs,
                    metric.reference.crit_fs_der
                )
            m = metric.evaluate(solution)
            metrics.append(m)

        metrics = np.hstack(metrics).tolist()

        return metrics

    __call__ = evaluate

    def plot_comparison(
            self, simulation_results, smooth=True, file_name=None, show=True):
        axs = []
        for metric in self.metrics:
            try:
                solution_path = self.solution_paths[metric]
                solution = get_nested_value(
                    simulation_results.solution_cycles, solution_path
                )[-1]
            except KeyError:
                raise CADETProcessError("Could not find solution path")

            solution = copy.deepcopy(solution)
            if smooth:
                solution.smooth_data(
                    metric.reference.s,
                    metric.reference.crit_fs,
                    metric.reference.crit_fs_der
                )
            m = metric.evaluate(solution)
            m = [
                np.format_float_scientific(
                    n, precision=2,
                )
                for n in m
            ]

            solution = metric.slice_and_transform(solution)

            ax = solution.plot(
                show=False, start=metric.start, end=metric.end,
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
            plotting.add_text(ax, f'{metric}: {m}')

            ax.legend(loc=1)

            axs.append(ax)

        if file_name is not None:
            plt.savefig(file_name)

        if not show:
            plt.close()

        return axs

    def __iter__(self):
        yield from self.metrics

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__
