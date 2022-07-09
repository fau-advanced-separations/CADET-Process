from collections import defaultdict
import shutil

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
            self, optimization_problem,
            working_directory, save_results=False, overwrite=True):
        self.optimization_problem = optimization_problem

        self._individuals = []
        self._hall_of_fame = []

        self.n_evals = []
        self.f_history = np.empty((0, optimization_problem.n_objectives))
        if optimization_problem.n_nonlinear_constraints > 1:
            self.g_history = np.empty((
                0, optimization_problem.n_nonlinear_constraints
            ))

        self.working_directory = working_directory
        self.save_results = save_results

        self.setup_directories(overwrite)

        if self.save_results:
            self.setup_figures()

        self.cache = ResultsCache(
            optimization_problem, self.cache_directory,
            autoclean=not(save_results)
        )

    def setup_directories(self, overwrite=False):
        if self.save_results:
            progress_dir = self.working_directory / 'progress'
            progress_dir.mkdir(exist_ok=overwrite)
            self.progress_directory = progress_dir
        else:
            self.progress_directory = None

        if self.save_results:
            results_dir = self.working_directory / 'results'
            progress_dir.mkdir(exist_ok=overwrite)
            self.results_directory = results_dir
        else:
            self.results_directory = None

        if self.save_results:
            cache_dir = self.working_directory / 'cache'
            results_dir.mkdir(exist_ok=overwrite)
            self.cache_directory = cache_dir
        else:
            self.cache_directory = None

    def setup_figures(self):
        self.setup_convergence_figure('objectives', show=False)
        if self.optimization_problem.n_nonlinear_constraints > 1:
            self.setup_convergence_figure(
                'nonlinear_constraints', show=False
            )
        self.setup_space_figure(show=False)

    def save_progress(self):
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

    def prune_cache(self):
        self.cache.prune()

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
            n = len(self.optimization_problem.objectives)
        elif target == 'nonlinear_constraints':
            n = len(self.optimization_problem.nonlinear_constraints)
        else:
            raise CADETProcessError("Unknown target.")

        if n == 0:
            return (None, None)
        if n == 1:
            fig, axs = plt.subplots()
            axs = [axs]
        else:
            fig, axs = plt.subplots(
                ncols=n,
                figsize=(n*4, 6),
            )

        fig.tight_layout()

        setattr(self, f'convergence_{target}_fig', fig)
        setattr(self, f'convergence_{target}_axs', axs)

        if not show:
            plt.close(fig)

        return fig, axs

    def plot_convergence(self, target, fig=None, axs=None, show=True):
        if fig is None and axs is None:
            if target == 'objectives':
                fig = self.convergence_objectives_fig
                axs = self.convergence_objectives_axs
            if target == 'nonlinear_constraints':
                fig = self.convergence_nonlinear_constraints_fig
                axs = self.convergence_nonlinear_constraints_axs

        layout = plotting.Layout()
        layout.x_label = '$n_{Evaluations}$'

        if target == 'objectives':
            funcs = self.optimization_problem.objectives
            layout.y_label = '$f~/~-$'
            values = self.f_history
        elif target == 'nonlinear_constraints':
            funcs = self.optimization_problem.nonlinear_constraints
            layout.y_label = '$g~/~-$'
            values = self.g_history
        else:
            raise CADETProcessError("Unknown target.")

        if len(funcs) == 0:
            return

        counter = 0
        for func, ax in zip(funcs, axs):
            lines = ax.get_lines()
            start = counter
            stop = counter+func.n_metrics
            v_func = values[:, start:stop]
            if len(lines) > 0:
                for i, line in enumerate(lines):
                    v_line = v_func[:, i]
                    line.set_xdata(self.n_evals)
                    layout.x_lim = (0, np.max(self.n_evals)+1)
                    line.set_ydata(v_line)
                layout.y_lim = (np.min(v_func), np.max(v_func))
            else:
                ax.plot(self.n_evals, v_func)

            layout.title = str(func)

            try:
                plotting.set_layout(ax, layout)
            except ValueError:
                pass

            counter += func.n_metrics

        fig.tight_layout()

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / target}.png')

        if not show:
            plt.close(fig)

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
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        fig.tight_layout()

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / "corner.png"}')

        if not show:
            plt.close(fig)

    def setup_space_figure(self, show=False):
        n = self.optimization_problem.n_variables

        if n == 0:
            return (None, None)
        if n == 1:
            fig, axs = plt.subplots()
            axs = [axs]
        else:
            fig, axs = plt.subplots(
                ncols=n,
                figsize=(n*4, 6),
            )

        fig.tight_layout()

        self.space_fig = fig
        self.space_axs = axs

        if not show:
            plt.close(fig)

        return fig, axs

    def plot_space(self, fig=None, axs=None, show=True):
        if fig is None and axs is None:
            fig, axs = self.space_fig, self.space_axs

        layout = plotting.Layout()
        layout.y_label = '$f~/~-$'

        variables = list(self.optimization_problem.variables_dict.keys())

        funcs = self.optimization_problem.objectives

        values = self.f
        x = self.x_untransformed

        for i_var, (var, ax) in enumerate(zip(variables, axs)):
            x_var = x[:, i_var]
            collections = copy.deepcopy(ax.collections)

            counter = 0
            for i_obj, func in enumerate(funcs):
                start = counter
                stop = counter+func.n_metrics
                v_var = values[:, start:stop]

                for i in range(func.n_metrics):
                    v_metric = v_var[:, i]
                    if len(collections) > 0:
                        ax.collections[i].set_offsets(
                            np.vstack((x_var, v_metric)).transpose()
                        )
                    else:
                        ax.scatter(x_var, v_metric)

            layout.title = var

            v_var = v_var.copy()
            v_var[np.where(np.isinf(v_var))] = np.nan
            layout.x_lim = (
                self.optimization_problem.lower_bounds[i_var],
                self.optimization_problem.upper_bounds[i_var]
            )
            layout.y_lim = (np.nanmin(v_var), np.nanmax(v_var))
            try:
                plotting.set_layout(ax, layout)
            except ValueError:
                pass

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / "parameter_space.png"}')

        if not show:
            plt.close(fig)

    def plot_pareto(self, show=True):
        plot = Scatter(tight_layout=True, plot_3D=False)
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

    def __init__(self, optimization_problem, directory=None, autoclean=False):
        self.autoclean = autoclean
        self.tags = defaultdict(list)
        self.cache = Cache(
           directory, disk=DillDisk, disk_min_file_size=2**18
        )

    @property
    def directory(self):
        return self.cache.directory

    def set(self, eval_obj, step, x, result, tag=None):
        key = f'{eval_obj}.{step}.{x}'
        if tag is not None:
            self.tags[tag].append(key)

        self.cache.set(key, result, expire=None, tag=tag)

    def get(self, eval_obj, step, x):
        key = f'{eval_obj}.{step}.{x}'

        result = self.cache[key]

        return result

    def delete(self, eval_obj, step, x):
        key = f'{eval_obj}.{step}.{x}'

        self.cache.delete(key)

    def prune(self, tag='temp'):
        keys = self.tags[tag]

        for key in keys:
            try:
                del self.cache[key]
            except KeyError:
                pass

        self.close()

    def close(self):
        self.cache.close()

    def __del__(self):
        if hasattr(self, 'autoclean') and self.autoclean:
            try:
                shutil.rmtree(self.directory, ignore_errors=True)
            except FileNotFoundError:
                pass
