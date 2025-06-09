from __future__ import annotations

import csv
import os
import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from addict import Dict
from cadet import H5
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from CADETProcess import CADETProcessError, plotting
from CADETProcess.dataStructure import (
    Bool,
    Dictionary,
    String,
    Structure,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.optimization import Individual, ParetoFront, Population
from CADETProcess.sysinfo import system_information

if TYPE_CHECKING:
    from CADETProcess.optimization import OptimizationProblem, OptimizerBase

cmap_feas = plt.get_cmap("winter_r")
cmap_infeas = plt.get_cmap("autumn_r")


class OptimizationResults(Structure):
    """
    Optimization results.

    Attributes
    ----------
    optimization_problem : OptimizationProblem
        Optimization problem.
    optimizer : OptimizerBase
        Optimizer used to optimize the OptimizationProblem.
    success : bool
        True if optimization was successfully terminated. False otherwise.
    exit_flag : int
        Information about the solver termination.
    exit_message : str
        Additional information about the solver status.
    time_elapsed : float
        Execution time of simulation.
    cpu_time : float
        CPU run time, taking into account the number of cores used for the optimiation.
    system_information : dict
        Information about the system on which the optimization was performed.
    x : list
        Values of optimization variables at optimum.
    f : np.ndarray
        Value of objective function at x.
    g : np.ndarray
        Values of constraint function at x
    population_last : Population
        Last population.
    pareto_front : ParetoFront
        Pareto optimal solutions.
    meta_front : ParetoFront
        Reduced pareto optimal solutions using meta scores and multi-criteria decision
        functions.
    """

    success = Bool(default=False)
    exit_flag = UnsignedInteger()
    exit_message = String()
    time_elapsed = UnsignedFloat()
    cpu_time = UnsignedFloat()
    system_information = Dictionary()

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        optimizer: OptimizerBase,
        similarity_tol: float = 0,
    ) -> None:
        """Initialize OptimizationResults object."""
        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

        self._optimizer_state = Dict()

        self._population_all = Population()
        self._populations = []
        self._similarity_tol = similarity_tol
        self._pareto_fronts = []

        if optimization_problem.n_multi_criteria_decision_functions > 0:
            self._meta_fronts = []
        else:
            self._meta_fronts = None

        self.results_directory = None

        self.system_information = system_information

    @property
    def results_directory(self) -> Path:
        """Path: Results directory path."""
        return self._results_directory

    @results_directory.setter
    def results_directory(self, results_directory: str | os.PathLike) -> None:
        if results_directory is not None:
            results_directory = Path(results_directory)
            self.plot_directory = Path(results_directory / "figures")
            self.plot_directory.mkdir(exist_ok=True)
        else:
            self.plot_directory = None

        self._results_directory = results_directory

    @property
    def is_finished(self) -> bool:
        """bool: True if optimization has finished. False otherwise."""
        if self.exit_flag is None:
            return False
        else:
            return True

    @property
    def optimizer_state(self) -> dict:
        """dict: Internal state of the optimizer."""
        return self._optimizer_state

    @property
    def populations(self) -> list[Population]:
        """list[Population]: List of populations per generation."""
        return self._populations

    @property
    def population_last(self) -> Population:
        """Population: Final population."""
        return self.populations[-1]

    @property
    def population_all(self) -> Population:
        """Population: Population with all evaluated individuals."""
        return self._population_all

    @property
    def pareto_fronts(self) -> list[ParetoFront]:
        """list[ParetoFront]: List of Pareto fronts per generation."""
        return self._pareto_fronts

    @property
    def pareto_front(self) -> ParetoFront:
        """ParetoFront: Final Pareto front."""
        return self._pareto_fronts[-1]

    @property
    def meta_fronts(self) -> list[Population]:
        """list[Population]: List of meta fronts per generation."""
        if self._meta_fronts is None:
            return self.pareto_fronts
        else:
            return self._meta_fronts

    @property
    def meta_front(self) -> Population:
        """Population: Meta front."""
        if self._meta_fronts is None:
            return self.pareto_front
        else:
            return self._meta_fronts[-1]

    def update(self, new: Individual | Population) -> None:
        """
        Update Results.

        Parameters
        ----------
        new : Individual, Population
            New results

        Raises
        ------
        CADETProcessError
            If new is not an instance of Individual or Population
        """
        if isinstance(new, Individual):
            population = Population()
            population.add_individual(new)
        elif isinstance(new, Population):
            population = new
        else:
            raise CADETProcessError("Expected Population or Individual")
        self._populations.append(population)
        self.population_all.update(population)

    def update_pareto(self, pareto_new: Population | None = None) -> None:
        """
        Update pareto front with new population.

        Parameters
        ----------
        pareto_new : Population, optional
            New pareto front. If None, update existing front with latest population.
        """
        pareto_front = ParetoFront(similarity_tol=self._similarity_tol)

        if pareto_new is not None:
            pareto_front.update_population(pareto_new)
        else:
            if len(self.pareto_fronts) > 0:
                pareto_front.update_population(self.pareto_front)
            pareto_front.update_population(self.population_last)

        if self._similarity_tol:
            pareto_front.remove_similar()
        self._pareto_fronts.append(pareto_front)

    def update_meta(self, meta_front: Population) -> None:
        """
        Update meta front with new population.

        Parameters
        ----------
        meta_front : Population
            New meta front.
        """
        if self._similarity_tol:
            meta_front.remove_similar()
        self._meta_fronts.append(meta_front)

    @property
    def n_evals(self) -> int:
        """int: Number of evaluations."""
        return sum([len(pop) for pop in self.populations])

    @property
    def n_gen(self) -> int:
        """int: Number of generations."""
        return len(self.populations)

    @property
    def x(self) -> np.ndarray:
        """np.ndarray: Optimal points in untransformed space."""
        return self.meta_front.x

    @property
    def x_transformed(self) -> np.ndarray:
        """np.ndarray: Optimal points in transformed space."""
        return self.meta_front.x_transformed

    @property
    def cv_bounds(self) -> np.ndarray:
        """np.ndarray: Bound constraint violation of optimal points."""
        return self.meta_front.cv_bounds

    @property
    def cv_lincon(self) -> np.ndarray:
        """np.ndarray: Linear constraint violation of optimal points."""
        return self.meta_front.cv_lincon

    @property
    def cv_lineqcon(self) -> np.ndarray:
        """np.ndarray: Linear equality constraint violation of optimal points."""
        return self.meta_front.cv_lineqcon

    @property
    def f(self) -> np.ndarray:
        """np.ndarray: Objective function values of optimal points."""
        return self.meta_front.f

    @property
    def g(self) -> np.ndarray:
        """np.ndarray: Nonlinear constraint function values of optimal points."""
        return self.meta_front.g

    @property
    def cv_nonlincon(self) -> np.ndarray:
        """np.ndarray: Nonlinear constraint violation values of optimal points."""
        return self.meta_front.cv_nonlincon

    @property
    def m(self) -> np.ndarray:
        """np.ndarray: Meta scores of optimal points."""
        return self.meta_front.m

    @property
    def n_evals_history(self) -> np.ndarray:
        """int: Number of evaluations per generation."""
        n_evals = [len(pop) for pop in self.populations]
        return np.cumsum(n_evals)

    @property
    def f_best_history(self) -> np.ndarray:
        """np.ndarray: Best objective values per generation."""
        return np.array([pop.f_best for pop in self.meta_fronts])

    @property
    def f_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum objective values per generation."""
        return np.array([pop.f_min for pop in self.meta_fronts])

    @property
    def f_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum objective values per generation."""
        return np.array([pop.f_max for pop in self.meta_fronts])

    @property
    def f_avg_history(self) -> np.ndarray:
        """np.ndarray: Average objective values per generation."""
        return np.array([pop.f_avg for pop in self.meta_fronts])

    @property
    def g_best_history(self) -> np.ndarray:
        """np.ndarray: Best nonlinear constraint per generation."""
        return np.array([pop.g_best for pop in self.meta_fronts])

    @property
    def g_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_min for pop in self.meta_fronts])

    @property
    def g_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_max for pop in self.meta_fronts])

    @property
    def g_avg_history(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_avg for pop in self.meta_fronts])

    @property
    def cv_nonlincon_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_min for pop in self.meta_fronts])

    @property
    def cv_nonlincon_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_max for pop in self.meta_fronts])

    @property
    def cv_nonlincon_avg_history(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_avg for pop in self.meta_fronts])

    @property
    def m_best_history(self) -> np.ndarray:
        """np.ndarray: Best meta scores per generation."""
        return np.array([pop.m_best for pop in self.meta_fronts])

    @property
    def m_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        else:
            return np.array([pop.m_min for pop in self.meta_fronts])

    @property
    def m_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        else:
            return np.array([pop.m_max for pop in self.meta_fronts])

    @property
    def m_avg_history(self) -> np.ndarray:
        """np.ndarray: Average meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        else:
            return np.array([pop.m_avg for pop in self.meta_fronts])

    def plot_figures(self, show: bool = True) -> None:
        """
        Plot result figures.

        See Also
        --------
        plot_convergence
        plot_objectives
        plot_corner
        plot_pairwise
        plot_pareto
        """
        if self.plot_directory is None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.plot_convergence(
                "objectives", show=show, plot_directory=self.plot_directory
            )
            if self.optimization_problem.n_nonlinear_constraints > 0:
                self.plot_convergence(
                    "nonlinear_constraints",
                    show=show,
                    plot_directory=self.plot_directory,
                )
            if self.optimization_problem.n_meta_scores > 0:
                self.plot_convergence(
                    "meta_scores", show=show, plot_directory=self.plot_directory
                )
            self.plot_objectives(show=show, plot_directory=self.plot_directory)
            if self.optimization_problem.n_variables > 1 and len(self.x) > 1:
                self.plot_corner(show=show, plot_directory=self.plot_directory)

            self.plot_pairwise(show=show, plot_directory=self.plot_directory)

            if self.optimization_problem.n_objectives > 1:
                self.plot_pareto(
                    show=show,
                    plot_directory=self.plot_directory,
                    plot_evolution=True,
                    plot_pareto=False,
                )

    def plot_objectives(
        self,
        include_meta: bool = True,
        plot_pareto: bool = False,
        plot_infeasible: bool = True,
        plot_individual: bool = False,
        autoscale: bool = True,
        show: bool = True,
        plot_directory: Optional[str | Path] = None,
    ) -> None:
        """
        Plot objective function values for all optimization generations.

        Parameters
        ----------
        include_meta : bool, optional
            If True, meta scores will be included in the plot. The default is True.
        plot_pareto : bool, optional
            If True, only plot Pareto front members of each generation are plotted.
            Else, all evaluated individuals are plotted.
            The default is False.
        plot_infeasible : bool, optional
            If True, plot infeasible points. The default is True.
        plot_individual : bool, optional
            If True, create separate figures for each objective. Otherwise, all
            objectives are plotted in one figure.
            The default is False.
        plot_infeasible : bool, optional
            If True, plot infeasible points. The default is False.
        autoscale : bool, optional
            If True, automatically adjust the scaling of the axes. The default is True.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved.
            The default is None.

        Returns
        -------
        tuple
            A tuple containing:
            - plt.Figure: The Matplotlib Figure object.
            - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplots.

        See Also
        --------
        CADETProcess.optimization.Population.plot_objectives
        """
        axs = None
        figs = None
        _show = False
        _plot_directory = None

        cNorm = colors.Normalize(vmin=0, vmax=self.n_gen)
        scalarMap_feas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_feas)
        scalarMap_infeas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_infeas)

        if plot_pareto:
            populations = self.pareto_fronts
            population_last = self.pareto_front
        else:
            populations = self.populations
            population_last = self.population_last

        for i, gen in enumerate(populations):
            if gen is population_last:
                _plot_directory = plot_directory
                _show = show
            figs, axs = gen.plot_objectives(
                figs,
                axs,
                include_meta=include_meta,
                plot_infeasible=plot_infeasible,
                plot_individual=plot_individual,
                autoscale=autoscale,
                color_feas=scalarMap_feas.to_rgba(i),
                color_infeas=scalarMap_infeas.to_rgba(i),
                show=_show,
                plot_directory=_plot_directory,
            )

        return figs, axs

    def plot_pareto(
        self,
        show: bool = True,
        plot_pareto: bool = True,
        plot_evolution: bool = False,
        plot_directory: Optional[str | Path] = None,
    ) -> None:
        """
        Plot Pareto fronts for each generation in the optimization.

        The Pareto front represents the optimal solutions that cannot be improved in one
        objective without sacrificing another.
        The method shows a pairwise Pareto plot, where each objective is plotted against
        every other objective in a scatter plot, allowing for a visualization of the
        trade-offs between the objectives.
        To highlight the progress, a colormap is used where later generations are
        plotted with darker blueish colors.

        Parameters
        ----------
        show : bool, optional
            If True, display the plot.
            The default is True.
        plot_pareto : bool, optional
            If True, only Pareto front members of each generation are plotted.
            Else, all evaluated individuals are plotted.
            The default is True.
        plot_evolution : bool, optional
            If True, the Pareto front is plotted for each generation.
            Else, only final Pareto front is plotted.
            The default is False.
        plot_directory : str, optional
            The directory where the plot should be saved.
            The default is None.

        Returns
        -------
        tuple
            A tuple containing:
            - plt.Figure: The Matplotlib Figure object.
            - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplots.

        See Also
        --------
        CADETProcess.optimization.Population.plot_pareto
        """
        plot = None
        _show = False
        _plot_directory = None

        cNorm = colors.Normalize(vmin=0, vmax=self.n_gen)
        scalarMap_feas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_feas)
        scalarMap_infeas = cmx.ScalarMappable(norm=cNorm, cmap=cmap_infeas)

        if plot_pareto:
            populations = self.pareto_fronts
            population_last = self.pareto_front
        else:
            populations = self.populations
            population_last = self.population_last

        if not plot_evolution:
            populations = [population_last]

        for i, gen in enumerate(populations):
            if gen is population_last:
                _plot_directory = plot_directory
                _show = show
            plot = gen.plot_pareto(
                plot,
                color_feas=scalarMap_feas.to_rgba(i),
                color_infeas=scalarMap_infeas.to_rgba(i),
                show=_show,
                plot_directory=_plot_directory,
            )

        return plot.fig, plot.ax

    @wraps(Population.plot_corner)
    def plot_corner(self, *args: Any, **kwargs: Any) -> None:
        """Create corner plot of population."""
        return self.population_all.plot_corner(*args, **kwargs)

    @wraps(Population.plot_pairwise)
    def plot_pairwise(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot population pairwise."""
        return self.population_all.plot_pairwise(*args, **kwargs)

    def setup_convergence_figure(
        self,
        target: Literal["objectives", "nonlinear_constraints", "meta_scores"],
        plot_individual: bool = False,
    ) -> tuple[list, list]:
        """
        Set up figures and axes for plotting convergence of specified targets.

        Parameters
        ----------
        target : str
            The target type for convergence plotting. Options are "objectives",
            "nonlinear_constraints", or "meta_scores".
        plot_individual : bool, optional
            If True, individual figures are created for each target. Otherwise, a single
            figure with subplots is created. Default is False.

        Returns
        -------
        tuple[list, list]
            A tuple containing lists of matplotlib Figure and Axes objects.
            Returns individual figures and axes if `plot_individual` is True.

        Raises
        ------
        CADETProcessError
            If the specified target is unknown or not supported.
        """
        if target == "objectives":
            n = self.optimization_problem.n_objectives
        elif target == "nonlinear_constraints":
            n = self.optimization_problem.n_nonlinear_constraints
        elif target == "meta_scores":
            n = self.optimization_problem.n_meta_scores
        else:
            raise CADETProcessError("Unknown target.")

        if n == 0:
            return (None, None)

        fig_all, axs_all = plt.subplots(
            ncols=n,
            figsize=(n * 6 + 2, 6),
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
        target: Literal["objectives", "nonlinear_constraints", "meta_scores"] = "objectives",
        figs: Optional[Figure] = None,
        axs: Optional[Axes] = None,
        plot_individual: bool = False,
        plot_avg: bool = True,
        autoscale: bool = True,
        show: bool = True,
        plot_directory: bool = None,
    ) -> tuple[list[Figure] | Figure, list[Axes]]:
        """
        Plot the convergence of optimization metrics over evaluations.

        Parameters
        ----------
        target : Literal["objectives", "nonlinear_constraints", "meta_scores"],
            The target metrics to plot. The default is "objectives".
        figs : plt.Figure or list of plt.Figure, optional
            Figure(s) to plot the objectives on.
        axs : plt.Axes or list of plt.Axes, optional
            Axes to plot the objectives on.
            If None, new figures and axes will be created.
        plot_individual : bool, optional
            If True, create individual figure vor each metric.
            The default is False.
        plot_avg : bool, optional
            If True, plot add trajectory of average value per generation.
            The default is True.
        autoscale : bool, optional
            If True, autoscale the y-axis. The default is True.
        show : bool, optional
            If True, show the plot. The default is True.
        plot_directory : str, optional
            A directory to save the plot, by default None.

        Returns
        -------
        tuple
            A tuple containing:
            - plt.Figure: The Matplotlib Figure object.
            - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplots.
        """
        if axs is None:
            figs, axs = self.setup_convergence_figure(target, plot_individual)

        if not isinstance(figs, list):
            figs = [figs]

        layout = plotting.Layout()
        layout.x_label = "$n_{Evaluations}$"

        if target == "objectives":
            funcs = self.optimization_problem.objectives
            values_min = self.f_best_history
            values_avg = self.f_avg_history
        elif target == "nonlinear_constraints":
            funcs = self.optimization_problem.nonlinear_constraints
            values_min = self.g_best_history
            values_avg = self.g_avg_history
        elif target == "meta_scores":
            funcs = self.optimization_problem.meta_scores
            values_min = self.m_best_history
            values_avg = self.m_avg_history
        else:
            raise CADETProcessError("Unknown target.")

        if len(funcs) == 0:
            return

        counter = 0
        for func in funcs:
            start = counter
            stop = counter + func.n_metrics
            v_func_min = values_min[:, start:stop]
            v_func_avg = values_avg[:, start:stop]

            for i_metric in range(func.n_metrics):
                v_line_min = v_func_min[:, i_metric]
                v_line_avg = v_func_avg[:, i_metric]

                ax = axs[counter + i_metric]
                lines = ax.get_lines()

                if len(lines) > 0:
                    lines[0].set_xdata(self.n_evals_history)
                    lines[0].set_ydata(v_line_min)
                    if plot_avg and self.population_last.n_individuals > 1:
                        lines[1].set_ydata(v_line_avg)
                else:
                    if plot_avg and self.population_last.n_individuals > 1:
                        label = "best"
                    else:
                        label = None

                    ax.plot(
                        self.n_evals_history, v_line_min, "--", color="k", label=label
                    )
                    if plot_avg and self.population_last.n_individuals > 1:
                        ax.plot(
                            self.n_evals_history,
                            v_line_avg,
                            "-",
                            color="k",
                            alpha=0.5,
                            label="avg",
                        )

                layout.x_lim = (0, np.max(self.n_evals_history) + 1)

                try:
                    label = func.labels[i_metric]
                except AttributeError:
                    label = f"{func}_{i_metric}"

                if plot_avg and self.population_last.n_individuals > 1:
                    y_min = np.nanmin(v_line_min)
                    y_max = np.nanmax(v_line_avg)
                else:
                    y_min = np.nanmin(v_line_min)
                    y_max = np.nanmax(v_line_min)

                layout.y_label = label
                if autoscale and y_min > 0:
                    if y_max / y_min > 100.0:
                        ax.set_yscale("log")
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
                    figname = f"convergence_{target}_{i}"
                    fig.savefig(f"{plot_directory / figname}.png")
            else:
                figname = f"convergence_{target}"
                figs[0].savefig(f"{plot_directory / figname}.png")

        if plot_individual:
            return figs, axs
        else:
            return figs[0], axs

    def save_results(self, file_name: str) -> None:
        """
        Save results to H5 file.

        Parameters
        ----------
        file_name : str
            Results file name without file extension.
        """
        if self.results_directory is not None:
            self._update_csv(self.population_last, "results_all", mode="a")
            self._update_csv(self.population_last, "results_last", mode="w")
            self._update_csv(self.pareto_front, "results_pareto", mode="w")
            if self.optimization_problem.n_meta_scores > 0:
                self._update_csv(self.meta_front, "results_meta", mode="w")

            results = H5()
            results.root = Dict(self.to_dict())
            results.filename = self.results_directory / f"{file_name}.h5"
            results.save()

    def load_results(self, file_name: str) -> None:
        """
        Update optimization results from an HDF5 checkpoint file.

        Parameters
        ----------
        file_name : str
            Path to the checkpoint file.
        """
        data = H5()
        data.filename = file_name

        # Check for CADET-Python >= v1.1, which introduced the .load_from_file interface.
        # If it's not present, assume CADET-Python <= 1.0.4 and use the old .load() interface
        # This check can be removed at some point in the future.
        if hasattr(data, "load_from_file"):
            data.load_from_file()
        else:
            data.load()

        self.update_from_dict(data.root)

    def to_dict(self) -> dict:
        """
        Convert Results to a dictionary.

        Returns
        -------
        addict.Dict
            Results as a dictionary with populations stored as list of dictionaries.
        """
        data = Dict()
        data.system_information = self.system_information
        data.optimizer_state = self.optimizer_state
        data.similarity_tol = self._similarity_tol
        data.population_all_id = str(self.population_all.id)
        data.populations = {i: pop.to_dict() for i, pop in enumerate(self.populations)}
        data.pareto_fronts = {
            i: front.to_dict() for i, front in enumerate(self.pareto_fronts)
        }
        if self._meta_fronts is not None:
            data.meta_fronts = {
                i: front.to_dict() for i, front in enumerate(self.meta_fronts)
            }
        if self.time_elapsed is not None:
            data.time_elapsed = self.time_elapsed
            data.cpu_time = self.cpu_time

        return data

    def update_from_dict(self, data: dict) -> None:
        """
        Update internal state from dictionary.

        Parameters
        ----------
        data : dict
            Serialized data.
        """
        self._optimizer_state = data["optimizer_state"]
        self._population_all = Population(id=data["population_all_id"])
        self._similarity_tol = data.get("similarity_tol")

        for pop_dict in data["populations"].values():
            pop = Population.from_dict(pop_dict)
            self.update(pop)

        self._pareto_fronts = [
            ParetoFront.from_dict(d) for d in data["pareto_fronts"].values()
        ]
        if self._meta_fronts is not None:
            self._meta_fronts = [
                ParetoFront.from_dict(d) for d in data["meta_fronts"].values()
            ]

    def setup_csv(self) -> None:
        """Create csv files for optimization results."""
        self._setup_csv("results_all")
        self._setup_csv("results_last")
        self._setup_csv("results_pareto")
        if self.optimization_problem.n_meta_scores > 0:
            self._setup_csv("results_meta")

    def _setup_csv(self, file_name: str) -> None:
        """
        Create csv file for optimization results.

        Parameters
        ----------
        file_name : str
            Results file name without file extension.
        """
        header = [
            "id",
            *self.optimization_problem.variable_names,
            *self.optimization_problem.objective_labels,
        ]

        if self.optimization_problem.n_nonlinear_constraints > 0:
            header += [*self.optimization_problem.nonlinear_constraint_labels]
        if self.optimization_problem.n_meta_scores > 0:
            header += [*self.optimization_problem.meta_score_labels]

        with open(f"{self.results_directory / file_name}.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)

    def _update_csv(
        self,
        population: Population,
        file_name: str,
        mode: Literal["w", "b"],
    ) -> None:
        """
        Update csv file with latest population.

        Parameters
        ----------
        population : Population
            latest Population.
        file_name : str
            Results file name without file extension.
        mode : {'a', 'w'}
            a: append to existing file.
            w: Create new csv.

        See Also
        --------
        setup_csv
        """
        if mode == "w":
            self._setup_csv(file_name)
            mode = "a"

        with open(f"{self.results_directory / file_name}.csv", mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            for ind in population:
                row = [ind.id, *ind.x.tolist(), *ind.f.tolist()]
                if ind.g is not None:
                    row += ind.g.tolist()
                if ind.m is not None:
                    row += ind.m.tolist()
                writer.writerow(row)
