from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess import plotting


class OptimizationProgress():
    def __init__(
            self,
            optimization_problem,
            plot_directory=None,
            overwrite=True):
        self.optimization_problem = optimization_problem

        self.n_evals = []
        self.f_history = np.empty((0, optimization_problem.n_objectives))
        if optimization_problem.n_nonlinear_constraints > 0:
            self.g_history = np.empty((
                0, optimization_problem.n_nonlinear_constraints
            ))
        if optimization_problem.n_meta_scores > 0:
            self.m_history = np.empty((
                0, optimization_problem.n_meta_scores
            ))

    def update(self, n_evals, f_min, g_min=None, m_min=None):
        """Add information about progress during optimization."""
        if len(self.n_evals) != 0:
            n_evals = self.n_evals[-1] + n_evals
        self.n_evals.append(n_evals)

        self.f_history = np.vstack((self.f_history, f_min))

        if g_min is not None:
            self.g_history = np.vstack((self.g_history, g_min))

        if m_min is not None:
            self.m_history = np.vstack((self.m_history, m_min))

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
            target,
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
            values = self.f_history
        elif target == 'nonlinear_constraints':
            funcs = self.optimization_problem.nonlinear_constraints
            values = self.g_history
        elif target == 'meta_scores':
            funcs = self.optimization_problem.meta_scores
            values = self.m_history
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
                        layout.y_label = f"$log_{{10}}$({label})"
                        y_lim = (y_min/2, y_max*2)
                if y_min != y_max:
                    layout.y_lim = y_lim

                try:
                    plotting.set_layout(ax, layout)
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
