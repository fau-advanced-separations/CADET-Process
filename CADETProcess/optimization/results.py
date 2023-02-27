import csv
from pathlib import Path
import warnings

from addict import Dict
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
cmap_feas = plt.get_cmap('winter_r')
cmap_infeas = plt.get_cmap('autumn_r')
import numpy as np

from cadet import H5
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
    meta_front : ParetoFront
        Reduced pareto optimal solutions using meta scores and multi-criteria decision
        functions.

    """
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()

    x = NdArray()
    f = NdArray()
    g = NdArray()
    cv = NdArray()
    m = NdArray()

    def __init__(
            self, optimization_problem, optimizer,
            remove_similar=True, cv_tol=1e-6):
        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

        self._optimizer_state = Dict()

        self._population_all = Population()
        self._populations = []
        self._remove_similar = remove_similar
        self._cv_tol = cv_tol
        self._pareto_fronts = []

        if optimization_problem.n_meta_scores > 0:
            self._meta_fronts = []
        else:
            self._meta_fronts = None

        self.results_directory = None

    @property
    def results_directory(self):
        return self._results_directory

    @results_directory.setter
    def results_directory(self, results_directory):
        if results_directory is not None:
            results_directory = Path(results_directory)
            self.plot_directory = Path(results_directory / 'figures')
            self.plot_directory.mkdir(exist_ok=True)
        else:
            self.plot_directory = None

        self._results_directory = results_directory

    @property
    def is_finished(self):
        if self.exit_flag is None:
            return False
        else:
            return True

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @property
    def populations(self):
        return self._populations

    @property
    def population_last(self):
        return self.populations[-1]

    @property
    def population_all(self):
        return self._population_all

    @property
    def pareto_fronts(self):
        return self._pareto_fronts

    @property
    def pareto_front(self):
        return self._pareto_fronts[-1]

    @property
    def meta_fronts(self):
        if self._meta_fronts is None:
            return self.pareto_fronts
        else:
            return self._meta_fronts

    @property
    def meta_front(self):
        if self._meta_fronts is None:
            return self.pareto_front
        else:
            return self._meta_fronts[-1]

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

    def update_pareto(self, pareto_new=None):
        """Update pareto front with new population.

        Parameters
        ----------
        pareto_new : Population, optional
            New pareto front. If None, update existing front with latest population.
        """
        pareto_front = ParetoFront(cv_tol=self._cv_tol)

        if pareto_new is not None:
            pareto_front.update_population(pareto_new)
        else:
            if len(self._pareto_fronts) > 0:
                pareto_front.update_population(self.pareto_front)
            pareto_front.update_population(self.population_last)

        if self._remove_similar:
            pareto_front.remove_similar()
        self._pareto_fronts.append(pareto_front)

    def update_meta(self, meta_front):
        """Update meta front with new population.

        Parameters
        ----------
        meta_front : Population
            New meta front.
        """
        if self._remove_similar:
            meta_front.remove_similar()
        self._meta_fronts.append(meta_front)

    @property
    def n_evals(self):
        """int: Number of evaluations."""
        return sum([len(pop) for pop in self.populations])

    @property
    def n_gen(self):
        """int: Number of generations."""
        return len(self.populations)

    @property
    def x(self):
        """np.array: Optimal points."""
        return self.meta_front.x

    @property
    def x_untransformed(self):
        """np.array: Optimal points."""
        return self.meta_front.x_untransformed

    @property
    def f(self):
        """np.array: Optimal objective values."""
        return self.meta_front.f

    @property
    def g(self):
        """np.array: Optimal nonlinear constraint values."""
        return self.meta_front.g

    @property
    def cv(self):
        """np.array: Optimal nonlinear constraint violations."""
        return self.meta_front.cv

    @property
    def m(self):
        """np.array: Optimal meta score values."""
        return self.meta_front.m

    @property
    def n_evals_history(self):
        """int: Number of evaluations per generation."""
        n_evals = [len(pop) for pop in self.populations]
        return np.cumsum(n_evals)

    @property
    def f_min_history(self):
        """np.array: Minimum objective values per generation."""
        return np.array([pop.f_min for pop in self.meta_fronts])

    @property
    def g_min_history(self):
        """np.array: Minimum nonlinera constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_min for pop in self.meta_fronts])

    @property
    def m_min_history(self):
        """np.array: Minimum meta score values per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        else:
            return np.array([pop.m_min for pop in self.meta_fronts])

    def plot_figures(self, show=True):
        if self.plot_directory is None:
            return

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
            self.plot_objectives(
                show=show, plot_directory=self.plot_directory
            )
            if self.optimization_problem.n_variables > 1:
                self.plot_corner(
                    show=show, plot_directory=self.plot_directory
                )

            if self.optimization_problem.n_objectives > 1:
                self.plot_pareto(
                    show=show, plot_directory=self.plot_directory
                )

    def plot_objectives(
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
        scalarMap_feas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_feas)
        scalarMap_infeas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_infeas)

        for i, gen in enumerate(self.populations):
            if gen is self.population_last:
                _plot_directory = plot_directory
                _show = show
            axs, figs = gen.plot_objectives(
                axs, figs,
                include_meta=include_meta,
                plot_individual=plot_individual,
                autoscale=autoscale,
                color_feas=scalarMap_feas.to_rgba(i),
                color_infeas=scalarMap_infeas.to_rgba(i),
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
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap_feas)

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
                    figname = f'convergence_{target}_{i}'
                    fig.savefig(
                        f'{plot_directory / figname}.png'
                    )
            else:
                figname = f'convergence_{target}'
                figs[0].savefig(
                    f'{plot_directory / figname}.png'
                )

    def save_results(self):
        if self.results_directory is not None:
            self.update_csv(self.population_last, 'results_all', mode='a')
            self.update_csv(self.meta_front, 'results_meta', mode='w')

            results = H5()
            results.root = Dict(self.to_dict())
            results.filename = self.results_directory / 'checkpoint.h5'
            results.save()

    def to_dict(self):
        """Convert Results to a dictionary.

        Returns
        -------
        addict.Dict
            Results as a dictionary with populations stored as list of dictionaries.
        """
        data = Dict()
        data.optimizer_state = self.optimizer_state
        data.population_all_id = str(self.population_all.id)
        data.populations = {i: pop.to_dict() for i, pop in enumerate(self.populations)}
        data.pareto_fronts = {
            i: front.to_dict() for i, front in enumerate(self.pareto_fronts)
        }
        if self._meta_fronts is not None:
            data.meta_fronts = {
                i: front.to_dict() for i, front in enumerate(self.meta_fronts)
            }

        return data

    def update_from_dict(self, data):
        """Update internal state from dictionary.

        Parameters
        ----------
        data : dict
            Serialized data.
        """
        self._optimizer_state = data['optimizer_state']
        self._population_all = Population(id=data['population_all_id'])

        for pop_dict in data['populations'].values():
            pop = Population.from_dict(pop_dict)
            self.update_population(pop)

        self._pareto_fronts = [
            ParetoFront.from_dict(d) for d in data['pareto_fronts'].values()
        ]
        if self._meta_fronts is not None:
            self._meta_fronts = [
                ParetoFront.from_dict(d) for d in data['meta_fronts'].values()
            ]

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
                row = [
                    ind.id,
                    *ind.x_untransformed.tolist(),
                    *ind.f.tolist()
                ]
                if ind.g is not None:
                    row += ind.g.tolist()
                if ind.m is not None:
                    row += ind.m.tolist()
                writer.writerow(row)
