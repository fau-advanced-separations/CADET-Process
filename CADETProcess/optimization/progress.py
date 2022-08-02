from collections import defaultdict
import shutil
import tempfile
import warnings

import corner
from diskcache import Cache
import numpy as np
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import List, Bool
from CADETProcess.dataStructure import DillDisk


class Individual(metaclass=StructMeta):
    x = List()
    f = List()
    g = List()
    is_valid = Bool(default=True)

    def __init__(self, x, f, g=None, x_untransformed=None):
        self.x = x
        self.f = f
        self.g = g

        if x_untransformed is None:
            x_untransformed = x
        self.x_untransformed = x_untransformed

    def dominates(self, other, objectives_filter=slice(None)):
        """
        Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        Parmeters
        ---------
        other : Individual
            Other individual
        param objectives_filter: slice
            Slice indicating on which objectives the domination is tested.
            The default value is `slice(None)`, representing all objectives.
        """
        dominates = False
        for self_value, other_value in zip(
                self.f[objectives_filter], other.f[objectives_filter]):
            if self_value < other_value:
                dominates = True
            elif self_value > other_value:
                return False
        return dominates

    def is_similar(self, other):
        """

        Return True if objectives are close to each other.

        To reduce number of entries, a rather high rtol is chosen.

        """
        similar_f = np.allclose(self.f, other.f, rtol=1e-8)

        if self.g is not None:
            similar_g = np.allclose(self.g, other.g, rtol=1e-8)
        else:
            similar_g = True

        if similar_f and similar_g:
            return True
        else:
            return False

    def __str__(self):
        return str(list(self.x))

    def __repr__(self):
        if self.g is None:
            return f'{self.__class__.__name__}({self.x}, {self.f})'
        else:
            return f'{self.__class__.__name__}({self.x}, {self.f}, {self.g})'


class OptimizationProgress():
    def __init__(
            self,
            optimization_problem,
            working_directory,
            results_directory,
            save_results=False,
            use_diskcache=True,
            cache_directory=None,
            keep_cache=True,
            overwrite=True):
        self.optimization_problem = optimization_problem

        self._individuals = []
        self._hall_of_fame = []

        self.n_evals = []
        self.f_history = np.empty((0, optimization_problem.n_objectives))
        if optimization_problem.n_nonlinear_constraints > 1:
            self.g_history = np.empty((
                0, optimization_problem.n_nonlinear_constraints
            ))

        self.save_results = save_results
        self.working_directory = working_directory
        self.results_directory = results_directory

        if self.save_results:
            progress_dir = self.working_directory / 'progress'
            progress_dir.mkdir(exist_ok=overwrite)
            self.progress_directory = progress_dir
        else:
            self.progress_directory = None

        self.keep_cache = keep_cache
        if use_diskcache:
            if self.save_results and cache_directory is None:
                cache_directory = self.working_directory / 'cache'
                cache_directory.mkdir(exist_ok=overwrite)
        self.cache = ResultsCache(use_diskcache, cache_directory)

        if self.save_results:
            self.setup_figures()

    def prune_cache(self):
        self.cache.prune()

    def delete_cache(self):
        self.cache.close()
        self.cache.delete_database()
        self.cache = None

    def setup_figures(self):
        self.setup_convergence_figure('objectives', show=False)
        if self.optimization_problem.n_nonlinear_constraints > 1:
            self.setup_convergence_figure(
                'nonlinear_constraints', show=False
            )
        self.setup_space_figure(show=False)

    def save_progress(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.plot_convergence('objectives', show=False)
            if self.optimization_problem.n_nonlinear_constraints > 1:
                self.plot_convergence('nonlinear_constraints', show=False)
            self.plot_corner(show=False)
            self.plot_space(show=False)
            self.plot_pareto(show=False)

    def save_callback(
            self, n_cores=1, current_iteration=None, untransform=False):
        if current_iteration is None:
            results_dir = self.results_directory
            current_iteration = 0
        else:
            results_dir = self.results_directory / str(current_iteration)
            results_dir.mkdir(exist_ok=True)

        self.optimization_problem.evaluate_callbacks_population(
            self.x_hof.tolist(),
            results_dir,
            cache=self.cache,
            n_cores=n_cores,
            current_iteration=current_iteration,
            untransform=untransform,
        )

    @property
    def individuals(self):
        """list: Evaluated individuals."""
        return self._individuals

    def add_individual(self, individual):
        if not isinstance(individual, Individual):
            raise TypeError("Expected Individual")
        self._individuals.append(individual)

    @property
    def hall_of_fame(self):
        return self._hall_of_fame

    @hall_of_fame.setter
    def hall_of_fame(self, hof):
        self._hall_of_fame = hof

    @property
    def x(self):
        """np.array: All evaluated points."""
        return np.array([ind.x for ind in self.individuals])

    @property
    def x_hof(self):
        """list: Optimization variable values of hall of fame entires."""
        return np.array([ind.x for ind in self.hall_of_fame])

    @property
    def x_untransformed(self):
        """np.array: All evaluated points."""
        return np.array([ind.x_untransformed for ind in self.individuals])

    @property
    def x_hof_untransformed(self):
        """list: Optimization variable values of hall of fame entires."""
        return np.array([ind._untransformed for ind in self.hall_of_fame])

    @property
    def f(self):
        """np.array: All evaluated objective function values."""
        return np.array([ind.f for ind in self.individuals])

    @property
    def f_hof(self):
        """np.array: Objective function values of hall of fame entires."""
        return np.array([ind.f for ind in self.hall_of_fame])

    @property
    def f_min(self):
        return np.min(self.f, axis=0)

    @property
    def f_min_hof(self):
        return np.min(self.f_hof, axis=0)

    @property
    def f_max(self):
        return np.max(self.f, axis=0)

    @property
    def f_max_hof(self):
        return np.max(self.f_hof, axis=0)

    @property
    def g(self):
        """np.array: All evaluated nonlinear constraint function values."""
        return np.array([ind.g for ind in self.individuals])

    @property
    def g_hof(self):
        """np.array: Constraint function values of hall of fame entires."""
        return np.array([ind.g for ind in self.hall_of_fame])

    @property
    def g_min(self):
        return np.min(self.g, axis=0)

    @property
    def g_min_hof(self):
        return np.min(self.g_hof, axis=0)

    @property
    def g_max(self):
        return np.max(self.g, axis=0)

    @property
    def g_max_hof(self):
        return np.max(self.g_hof, axis=0)

    def update_history(self):
        """Add information about progress during optimization."""
        self.n_evals.append(len(self.individuals))
        self.f_history = np.vstack((self.f_history, self.f_min_hof))
        if self.optimization_problem.n_nonlinear_constraints > 1:
            self.g_history = np.vstack((self.g_history, self.g_min_hof))

    def setup_convergence_figure(self, target, show=False):
        if target == 'objectives':
            n = self.optimization_problem.n_objectives
        elif target == 'nonlinear_constraints':
            n = self.optimization_problem.n_nonlinear_constraints
        else:
            raise CADETProcessError("Unknown target.")

        if n == 0:
            return (None, None)

        fig_all, axs_all = plt.subplots(
            ncols=n,
            figsize=(n*6 + 2, 6),
        )

        if not show:
            plt.close(fig_all)

        setattr(self, f'convergence_{target}_fig_all', fig_all)
        setattr(self, f'convergence_{target}_axs_all', axs_all)

        figs_ind = []
        axs_ind = []
        for i in range(n):
            fig, ax = plt.subplots()
            figs_ind.append(fig)
            axs_ind.append(ax)
            if not show:
                plt.close(fig)

        axs_ind = np.array(axs_ind).reshape(axs_all.shape)
        setattr(self, f'convergence_{target}_fig_ind', figs_ind)
        setattr(self, f'convergence_{target}_axs_ind', axs_ind)

    def plot_convergence(
            self,
            target,
            figs=None, axs=None,
            plot_individual=False,
            autoscale=True, show=True):

        if axs is None:
            if target == 'objectives':
                if plot_individual:
                    axs = self.convergence_objectives_axs_ind
                else:
                    axs = self.convergence_objectives_axs_all

            if target == 'nonlinear_constraints':
                if plot_individual:
                    axs = self.convergence_nonlinear_constraints_axs_ind
                else:
                    axs = self.convergence_nonlinear_constraints_axs_all

        layout = plotting.Layout()
        layout.x_label = '$n_{Evaluations}$'

        if target == 'objectives':
            funcs = self.optimization_problem.objectives
            values = self.f_history
        elif target == 'nonlinear_constraints':
            funcs = self.optimization_problem.nonlinear_constraints
            values = self.g_history
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
                    lines[0].set_xdata(self.n_evals)
                    lines[0].set_ydata(v_line)
                else:
                    ax.plot(self.n_evals, v_line)

                layout.x_lim = (0, np.max(self.n_evals)+1)
                layout.y_lim = (np.min(v_line), np.max(v_line))

                try:
                    label = func.labels[i_metric]
                except AttributeError:
                    label = f'{func}_{i_metric}'

                y_min = np.nanmin(v_line)
                y_max = np.nanmax(v_line)
                y_lim = (0.9*y_min, 1.1*y_max)
                layout.y_label = label
                if autoscale and np.min(v_line) > 0:
                    if np.max(v_line) / np.min(v_line[v_line > 0]) > 100.0:
                        ax.set_yscale('log')
                        layout.y_label = f"log10({label})"
                        y_lim = (y_min/2, y_max*2)
                if y_min != y_max:
                    layout.y_lim = y_lim

                try:
                    plotting.set_layout(ax, layout)
                except ValueError:
                    pass

            counter += func.n_metrics

        if figs is None:
            if plot_individual:
                if target == 'objectives':
                    figs = self.convergence_objectives_fig_ind
                elif target == 'nonlinear_constraints':
                    figs = self.convergence_nonlinear_constraints_fig_ind
            else:
                if target == 'objectives':
                    figs = [self.convergence_objectives_fig_all]
                elif target == 'nonlinear_constraints':
                    figs = [self.convergence_nonlinear_constraints_fig_all]

        for fig in figs:
            fig.tight_layout()
            if not show:
                plt.close(fig)
            else:
                dummy = plt.figure()
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)
                plt.show()

        if self.progress_directory is not None:
            if plot_individual:
                for i, fig in enumerate(figs):
                    fig.savefig(
                        f'{self.progress_directory / target}_{i}.png'
                    )
            else:
                figs[0].savefig(
                    f'{self.progress_directory / target}.png'
                )

    def plot_corner(self, untransformed=True, show=True):
        if untransformed:
            x = self.x
            labels = self.optimization_problem.independent_variable_names
        else:
            x = self.x_untransformed
            labels = self.optimization_problem.variable_names

        fig = corner.corner(
            x,
            labels=labels,
            bins=20,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 20},
            title_fmt=".2g",
            use_math_text=True,
            quiet=True,
        )
        fig_size = 6*len(labels)
        fig.set_size_inches((fig_size, fig_size))
        fig.tight_layout()

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / "corner.png"}')

        if not show:
            plt.close(fig)

    def setup_space_figure(self, show=False):
        n = self.optimization_problem.n_variables
        m = self.optimization_problem.n_objectives

        if n == 0:
            return (None, None)

        fig_all, axs_all = plt.subplots(
            nrows=m,
            ncols=n,
            figsize=(n*8 + 2, m*8 + 2),
        )

        if not show:
            plt.close(fig_all)

        self.space_fig_all = fig_all
        self.space_axs_all = axs_all

        figs_ind = []
        axs_ind = []
        for i in range(m*n):
            fig, ax = plt.subplots()
            figs_ind.append(fig)
            axs_ind.append(ax)
            if not show:
                plt.close(fig)

        self.space_fig_ind = figs_ind
        self.space_axs_ind = np.array(axs_ind).reshape(axs_all.shape)

    def plot_space(
            self,
            figs=None, axs=None,
            plot_individual=False,
            autoscale=True, show=True):
        if axs is None:
            if plot_individual:
                axs = self.space_axs_ind
            else:
                axs = self.space_axs_all

        layout = plotting.Layout()
        layout.y_label = '$f~/~-$'

        variables = list(self.optimization_problem.variables_dict.keys())

        values = self.f
        x = self.x_untransformed

        for i_var, var in enumerate(variables):
            x_var = x[:, i_var]

            counter = 0
            for objective in self.optimization_problem.objectives:
                start = counter
                stop = counter+objective.n_metrics
                v_var = values[:, start:stop]

                for i_metric in range(objective.n_metrics):
                    v_metric = v_var[:, i_metric]

                    ax = axs[counter+i_metric][i_var]
                    collections = ax.collections

                    if len(collections) > 0:
                        ax.collections[0].set_offsets(
                            np.vstack((x_var, v_metric)).transpose()
                        )
                    else:
                        ax.scatter(x_var, v_metric)

                    v_var = v_var.copy()
                    v_var[np.where(np.isinf(v_var))] = np.nan

                    layout.x_lim = (np.nanmin(x_var), np.nanmax(x_var))
                    layout.x_label = f"{var}"
                    if autoscale and np.min(x_var) > 0:
                        if np.max(x_var) / np.min(x_var[x_var > 0]) > 100.0:
                            ax.set_xscale('log')
                            layout.x_label = f"log10({var})"

                    try:
                        label = objective.labels[i_metric]
                    except AttributeError:
                        label = f'{objective}'
                        if objective.n_metrics > 1:
                            label = f'{objective}_{i_metric}'

                    y_min = np.nanmin(v_metric)
                    y_max = np.nanmax(v_metric)
                    y_lim = (
                        min(0.9*y_min, y_min - 0.01*(y_max-y_min)),
                        1.1*y_max
                    )
                    layout.y_label = label
                    if autoscale and np.min(v_var) > 0:
                        if np.max(v_var) / np.min(v_var[v_var > 0]) > 100.0:
                            ax.set_yscale('log')
                            layout.y_label = f"log10({label})"
                            y_lim = (y_min/2, y_max*2)
                    if y_min != y_max:
                        layout.y_lim = y_lim

                    try:
                        plotting.set_layout(ax, layout)
                    except ValueError:
                        pass

                counter += objective.n_metrics

        if figs is None:
            if plot_individual:
                figs = self.space_fig_ind
            else:
                figs = [self.space_fig_all]

        for fig in figs:
            fig.tight_layout()
            if not show:
                plt.close(fig)
            else:
                dummy = plt.figure()
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)
                plt.show()

        if self.progress_directory is not None:
            if plot_individual:
                for i, fig in enumerate(figs):
                    fig.savefig(
                        f'{self.progress_directory / "parameter_space"}_{i}.png'
                    )
            else:
                figs[0].savefig(
                    f'{self.progress_directory / "parameter_space"}.png'
                )

    def plot_pareto(self, show=True):
        n = self.optimization_problem.n_objectives

        plot = Scatter(
            figsize=(6 * n, 5 * n),
            tight_layout=True,
            plot_3D=False
        )
        plot.add(self.f, s=10)

        if self.progress_directory is not None:
            plot.save(f'{self.progress_directory / "pareto.png"}')

        if not show:
            plt.close(plot.fig)


class ResultsCache():
    """
    Internal structure:
    [evaluation_object][step][x]

    For example:
    [EvaluationObject 1][Evaluator 1][x] -> IntermediateResults 1
    [EvaluationObject 1][Evaluator 2][x] -> IntermediateResults 2
    [EvaluationObject 1][Objective 1][x] -> f1.1
    [EvaluationObject 1][Objective 2][x] -> f1.2
    [EvaluationObject 1][Constraint 1][x] -> g1.1

    [EvaluationObject 2][Evaluator 1][x] -> IntermediateResults 1
    [EvaluationObject 2][Evaluator 2][x] -> IntermediateResults 2
    [EvaluationObject 2][Objective 1][x] -> f2.1
    [EvaluationObject 2][Objective 2][x] -> f2.2
    [EvaluationObject 2][Constraint 1][x] -> g2.1

    [None][Evaluator 1][x] -> IntermediateResults 1
    [Objective 3][x] -> f3
    [Constraint 2][x] -> g2

    """

    def __init__(self, use_diskcache=True, directory=None):
        self.init_cache(use_diskcache, directory)

        self.tags = defaultdict(list)

    def init_cache(self, use_diskcache, directory):
        if use_diskcache:
            if directory is None:
                directory = tempfile.mkdtemp(prefix='diskcache-')
            self.directory = directory

            self.cache = Cache(
               directory,
               disk=DillDisk,
               disk_min_file_size=2**18,
               size_limit=2**36,
            )
            self.directory = self.cache.directory
        else:
            self.cache = {}
            self.directory = None

        self.use_diskcache = use_diskcache

    def set(self, eval_obj, step, x, result, tag=None):
        key = f'{eval_obj}.{step}.{x}'
        if tag is not None:
            self.tags[tag].append(key)

        if self.use_diskcache:
            self.cache.set(key, result, expire=None)
        else:
            self.cache[key] = result

    def get(self, eval_obj, step, x):
        key = f'{eval_obj}.{step}.{x}'

        result = self.cache[key]

        return result

    def delete(self, eval_obj, step, x):
        key = f'{eval_obj}.{step}.{x}'

        if self.use_diskcache:
            self.cache.delete(key)
        else:
            self.cache.pop(key)

    def prune(self, tag='temp'):
        try:
            keys = self.tags.pop(tag)
            for key in keys:
                eval_obj, step, x = key.split('.')
                self.delete(eval_obj, step, x)
        except KeyError:
            pass

    def close(self):
        if self.use_diskcache:
            self.cache.close()

    def delete_database(self, reinit=False):
        if self.use_diskcache:
            self.close()
            try:
                shutil.rmtree(self.directory, ignore_errors=True)
            except FileNotFoundError:
                pass

        self.cache = None

        if reinit:
            self.init_cache(self.use_diskcache, self.directory)
