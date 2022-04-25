import copy

import corner
import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import List, Bool

from pymoo.visualization.scatter import Scatter


class Individual(metaclass=StructMeta):
    x = List()
    f = List()
    g = List()
    is_valid = Bool(default=True)

    def __init__(self, x, f, g=None):
        self.x = x
        self.f = f
        self.g = g

    def __str__(self):
        return str(list(self.x))

    def __repr__(self):
        if self.g is None:
            return f'{self.__class__.__name__}({self.x}, {self.f})'
        else:
            return f'{self.__class__.__name__}({self.x}, {self.f}, {self.g})'


class OptimizationProgress():
    def __init__(self, optimization_problem, save_results=False):
        self.optimization_problem = optimization_problem

        self._individuals = []
        self._hall_of_fame = []

        self.n_evals = []
        self.f_history = np.empty((0, optimization_problem.n_objectives))
        if optimization_problem.n_nonlinear_constraints > 1:
            self.g_history = np.empty((
                0, optimization_problem.n_nonlinear_constraints
            ))

        self.cache = optimization_problem.setup_cache()

        if save_results:
            self.setup_convergence_figure('objectives', show=False)
            if optimization_problem.n_nonlinear_constraints > 1:
                self.setup_convergence_figure(
                    'nonlinear_constraints', show=False
                )
            self.setup_space_figure(show=False)

        self.progress_directory = None
        self.results_directory = None

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

    def prune_cache(self):
        """Only keep members of hall of fame."""
        problem = self.optimization_problem
        cache = problem.setup_cache(only_cached_evalutors=True)

        evaluators = problem.cached_evaluators

        x_hof = self.optimization_problem.untransform(
            self.x_hof, enforce2d=True
        )

        for ind in x_hof:
            try:
                ind = tuple(ind)
                for objective in problem.objectives:
                    if len(objective.evaluation_objects) == 0:
                        for evaluator in objective.evaluators:
                            if evaluator in evaluators:
                                cache[str(evaluator)][ind] = \
                                    self.cache[str(evaluator)][ind]
                    else:
                        for eval_obj in objective.evaluation_objects:
                            cache[str(eval_obj)][objective.name][ind] = \
                                self.cache[str(eval_obj)][objective.name][ind]

                        for evaluator in objective.evaluators:
                            if evaluator in evaluators:
                                cache[str(eval_obj)][str(evaluator)][ind] = \
                                    self.cache[str(eval_obj)][str(evaluator)][ind]

                for nonlincon in problem.nonlinear_constraints:
                    if len(nonlincon.evaluation_objects) == 0:
                        for evaluator in nonlincon.evaluators:
                            if evaluator in evaluators:
                                cache[str(evaluator)][ind] = \
                                    self.cache[str(evaluator)][ind]
                    else:
                        for eval_obj in nonlincon.evaluation_objects:
                            cache[str(eval_obj)][nonlincon.name][ind] = \
                                self.cache[str(eval_obj)][nonlincon.name][ind]

                        for evaluator in nonlincon.evaluators:
                            if evaluator in evaluators:
                                cache[str(eval_obj)][str(evaluator)][ind] = \
                                    self.cache[str(eval_obj)][str(evaluator)][ind]
            except KeyError:
                pass

        self.cache = cache

        return cache

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
            plotting.set_layout(ax, layout)

            counter += func.n_metrics

        fig.tight_layout()

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / target}.png')

        if not show:
            plt.close(fig)

    def plot_corner(self, show=True):
        fig = corner.corner(
            self.x,
            labels=self.optimization_problem.variable_names,
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
        x = self.x

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
            layout.x_lim = (np.min(x_var), np.max(x_var))
            layout.y_lim = (np.nanmin(v_var), np.nanmax(v_var))

            plotting.set_layout(ax, layout)

        if self.progress_directory is not None:
            fig.savefig(f'{self.progress_directory / "parameter_space.png"}')

        if not show:
            plt.close(fig)

    def plot_pareto(self, show=True):
        plot = Scatter(tight_layout=True)
        plot.add(self.f, s=10)

        if self.progress_directory is not None:
            plot.save(f'{self.progress_directory / "pareto.png"}')

        if not show:
            plt.close(plot.fig)
