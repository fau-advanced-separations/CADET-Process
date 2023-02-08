import csv
from pathlib import Path
import warnings

from addict import Dict
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
cmap = plt.get_cmap('winter')
import numpy as np

from CADETProcess import plotting
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    NdArray, String, UnsignedInteger, UnsignedFloat
)

from CADETProcess import CADETProcessError
from CADETProcess.optimization import Individual, Population, ParetoFront


class OptimizationResults(metaclass=StructMeta):
    """Optimization results.

    Attributes
    ----------
    optimization_problem : OptimizationProblem
        Optimization problem.
    optimizer : OptimizerBase
        Optimizer used to optimize the OptimizationProblem.
    exit_flag : int
        Information about the solver termination.
    exit_message : str
        Additional information about the solver status.
    time_elapsed : float
        Execution time of simulation.
    x : list
        Values of optimization variables at optimum.
    f : np.ndarray
        Value of objective function at x.
    g : np.ndarray
        Values of constraint function at x
    population : Population
        Last population.
    pareto_front : ParetoFront
        Pareto optimal solutions.

    """
    exit_flag = UnsignedInteger(default=0)
    exit_message = String(default='Unfinished')
    time_elapsed = UnsignedFloat()

    x = NdArray()
    f = NdArray()
    g = NdArray()
    m = NdArray()

    def __init__(
            self, optimization_problem, optimizer, results_directory=None,
            cv_tol=1e-6):
        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

        self.population_all = Population()
        self._populations = []
        self.pareto_front = ParetoFront(cv_tol=cv_tol)
        self._meta_population = None

        self.progress = OptimizationProgress(optimization_problem)
        self.has_finished = False

        self.results_directory = results_directory

        if self.results_directory is not None:
            self.plot_directory = Path(self.results_directory / 'figures')
            self.plot_directory.mkdir(exist_ok=True)
            self.setup_csv('results_meta')
            self.setup_csv('results_all')
        else:
            self.plot_directory = None

    @property
    def meta_population(self):
        if self._meta_population is None:
            return self.pareto_front
        else:
            return self._meta_population

    @meta_population.setter
    def meta_population(self, meta_population):
        self._meta_population = meta_population

    @property
    def populations(self):
        return self._populations

    @property
    def population_last(self):
        return self.populations[-1]

    def update_individual(self, individual):
        """Update Results.

        Parameters
        ----------
        individual : Individual
            Latest individual.

        Raises
        ------
        CADETProcessError
            If individual is not an instance of Individual
        """
        if not isinstance(individual, Individual):
            raise CADETProcessError("Expected Individual")

        population = Population()
        population.add_individual(individual)
        self._populations.append(population)
        self.population_all.add_individual(individual, ignore_duplicate=True)

    def update_population(self, population):
        """Update Results.

        Parameters
        ----------
        population : Population
            Current population

        Raises
        ------
        CADETProcessError
            If population is not an instance of Population
        """
        if not isinstance(population, Population):
            raise CADETProcessError("Expected Population")
        self._populations.append(population)
        self.population_all.update(population)

    def update_progress(self):
        if len(self.populations) == 0:
            n_evals = 1
        else:
            n_evals = len(self.populations[-1])

        self.progress.update(
            n_evals,
            self.pareto_front.f_min,
            self.pareto_front.g_min,
            self.pareto_front.m_min,
        )

    @property
    def x(self):
        """np.array: Optimal points."""
        if self.meta_population is not None:
            return self.meta_population.x
        return self.pareto_front.x

    @property
    def x_untransformed(self):
        """np.array: Optimal points."""
        if self.meta_population is not None:
            return self.meta_population.x_untransformed
        return self.pareto_front.x_untransformed

    @property
    def f(self):
        """np.array: Optimal objective values."""
        if self.meta_population is not None:
            return self.meta_population.f
        return self.pareto_front.f

    @property
    def g(self):
        """np.array: Optimal nonlinear constraint values."""
        if self.meta_population is not None:
            return self.meta_population.g
        return self.pareto_front.g

    @property
    def m(self):
        """np.array: Optimal meta score values."""
        if self.meta_population is not None:
            return self.meta_population.m
        return self.pareto_front.m

    def plot_figures(self, show=True):
        if self.plot_directory is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.plot_convergence(
                    'objectives',
                    show=show,
                    plot_directory=self.plot_directory
                )
                if self.optimization_problem.n_nonlinear_constraints > 0:
                    self.plot_convergence(
                        'nonlinear_constraints',
                        show=show,
                        plot_directory=self.plot_directory
                    )
                if self.optimization_problem.n_meta_scores > 0:
                    self.plot_convergence(
                        'meta_scores',
                        show=show,
                        plot_directory=self.plot_directory
                    )
                self.plot_space(
                    show=show, plot_directory=self.plot_directory
                )
                if self.optimization_problem.n_variables > 1:
                    self.plot_corner(
                        show=show, plot_directory=self.plot_directory
                    )

                if self.optimization_problem.n_objectives> 1:
                    self.plot_pareto(
                        show=show, plot_directory=self.plot_directory
                    )

    def plot_space(
            self,
            include_meta=True,
            plot_individual=False,
            autoscale=True,
            show=True,
            plot_directory=None):
        axs = None
        figs = None
        _show = False
        _plot_directory = None

        cNorm = colors.Normalize(vmin=0, vmax=len(self.populations))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        for i, gen in enumerate(self.populations):
            color = scalarMap.to_rgba(i)
            if gen is self.population_last:
                _plot_directory = plot_directory
                _show = show
            axs, figs = gen.plot_space(
                axs, figs,
                include_meta=include_meta,
                plot_individual=plot_individual,
                autoscale=autoscale,
                color=color,
                show=_show,
                plot_directory=_plot_directory
            )

    def plot_pareto(
            self,
            show=True,
            plot_directory=None):
        plot = None
        _show = False
        _plot_directory = None

        cNorm = colors.Normalize(vmin=0, vmax=len(self.populations))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        for i, gen in enumerate(self.populations):
            color = scalarMap.to_rgba(i)
            if gen is self.population_last:
                _plot_directory = plot_directory
                _show = show
            plot = gen.plot_pareto(
                plot,
                color=color,
                show=_show,
                plot_directory=_plot_directory
            )

    def plot_corner(self, *args, **kwargs):
        try:
            self.population_all.plot_corner(*args, **kwargs)
        except AssertionError:
            pass

    def setup_convergence_figure(self, target, plot_individual=False):
        if target == 'objectives':
            n = self.optimization_problem.n_objectives
        elif target == 'nonlinear_constraints':
            n = self.optimization_problem.n_nonlinear_constraints
        elif target == 'meta_scores':
            n = self.optimization_problem.n_meta_scores
        else:
            raise CADETProcessError("Unknown target.")

        if n == 0:
            return (None, None)

        fig_all, axs_all = plt.subplots(
            ncols=n,
            figsize=(n*6 + 2, 6),
            squeeze=False,
        )
        axs_all = axs_all.reshape((-1,))

        plt.close(fig_all)

        figs_ind = []
        axs_ind = []
        for i in range(n):
            fig, ax = plt.subplots()
            figs_ind.append(fig)
            axs_ind.append(ax)
            plt.close(fig)

        axs_ind = np.array(axs_ind).reshape(axs_all.shape)

        if plot_individual:
            return figs_ind, axs_ind
        else:
            return fig_all, axs_all

    def plot_convergence(
            self,
            target='objectives',
            figs=None, axs=None,
            plot_individual=False,
            autoscale=True,
            show=True,
            plot_directory=None):

        if axs is None:
            figs, axs = self.setup_convergence_figure(target, plot_individual)

        if not isinstance(figs, list):
            figs = [figs]

        layout = plotting.Layout()
        layout.x_label = '$n_{Evaluations}$'

        if target == 'objectives':
            funcs = self.optimization_problem.objectives
            values = self.f_min_history
        elif target == 'nonlinear_constraints':
            funcs = self.optimization_problem.nonlinear_constraints
            values = self.g_min_history
        elif target == 'meta_scores':
            funcs = self.optimization_problem.meta_scores
            values = self.m_min_history
        else:
            raise CADETProcessError("Unknown target.")

        if len(funcs) == 0:
            return

        counter = 0
        for func in funcs:
            start = counter
            stop = counter+func.n_metrics
            v_func = values[:, start:stop]

            for i_metric in range(func.n_metrics):
                v_line = v_func[:, i_metric]

                ax = axs[counter + i_metric]
                lines = ax.get_lines()

                if len(lines) > 0:
                    lines[0].set_xdata(self.n_evals_history)
                    lines[0].set_ydata(v_line)
                else:
                    ax.plot(self.n_evals_history, v_line)

                layout.x_lim = (0, np.max(self.n_evals_history)+1)

                try:
                    label = func.labels[i_metric]
                except AttributeError:
                    label = f'{func}_{i_metric}'

                y_min = np.nanmin(v_line)
                y_max = np.nanmax(v_line)
                layout.y_label = label
                if autoscale and np.min(v_line) > 0:
                    if np.max(v_line) / np.min(v_line[v_line > 0]) > 100.0:
                        ax.set_yscale('log')
                        layout.y_label = f"$log_{{10}}$({label})"

                try:
                    plotting.set_layout(ax, layout)
                    ax.relim()
                    ax.autoscale_view()
                except ValueError:
                    pass

            counter += func.n_metrics

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

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            if plot_individual:
                for i, fig in enumerate(figs):
                    fig.savefig(
                        f'{plot_directory / target}_{i}.png'
                    )
            else:
                figs[0].savefig(
                    f'{plot_directory / target}.png'
                )

    def save_results(self):
        if self.results_directory is not None:
            self.update_csv(self.population_last, 'results_all', mode='a')
            self.update_csv(self.pareto_front, 'results_meta', mode='w')

    def setup_csv(self, file_name):
        """Create csv file for optimization results.

        Parameters
        ----------
        file_name : {str, Path}
            Path to save results.
        """
        header = [
            "id",
            *self.optimization_problem.variable_names,
            *self.optimization_problem.objective_labels
        ]

        if self.optimization_problem.n_nonlinear_constraints > 0:
            header += [*self.optimization_problem.nonlinear_constraint_labels]
        if self.optimization_problem.n_meta_scores > 0:
            header += [*self.optimization_problem.meta_score_labels]

        with open(
                f'{self.results_directory / file_name}.csv', 'w'
        ) as csvfile:

            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)

    def update_csv(self, population, file_name, mode='a'):
        """Update csv file with latest population.

        Parameters
        ----------
        population : Population
            latest Population.
        file_name : {str, Path}
            Path to save results.
        mode : {'a', 'w'}
            a: append to existing file.
            w: Create new csv.

        See also
        --------
        setup_csv
        """
        if mode == 'w':
            self.setup_csv(file_name)
            mode = 'a'

        with open(
                f'{self.results_directory / file_name}.csv', mode
        ) as csvfile:

            writer = csv.writer(csvfile, delimiter=",")

            for ind in population:
                row = [ind.id, *ind.x_untransformed, *ind.f]
                if ind.g is not None:
                    row += ind.g
                if ind.m is not None:
                    row += ind.m
                writer.writerow(row)
