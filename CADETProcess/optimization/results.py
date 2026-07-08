from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from addict import Dict
from cadet import H5

from CADETProcess import CADETProcessError, plotting
from CADETProcess.dataStructure import (
    Bool,
    Dictionary,
    String,
    Structure,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.optimization import (
    IndividualView,
    ParetoFront,
    Population,
)
from CADETProcess.sysinfo import system_information

if TYPE_CHECKING:
    from CADETProcess.optimization import OptimizationProblem, OptimizerBase

cmap_feas = plt.colormaps["winter_r"]
cmap_infeas = plt.colormaps["autumn_r"]

__all__ = ["OptimizationResults"]


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

        self._population_all = None
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
        if self._population_all is None:
            self._population_all = Population.concat(
                self._populations
            ).drop_duplicates()
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

    def update(self, new: IndividualView | Population) -> None:
        """
        Update Results.

        Parameters
        ----------
        new : IndividualView, Population
            New results

        Raises
        ------
        CADETProcessError
            If new is not an instance of IndividualView or Population
        """
        if isinstance(new, IndividualView):
            population = new.as_population()
        elif isinstance(new, Population):
            population = new
        else:
            raise CADETProcessError("Expected Population or IndividualView")
        self._populations.append(population)
        self._population_all = None

    def _create_pareto_front(self) -> ParetoFront:
        """Build an empty ParetoFront with problem context and optimizer tolerances."""
        problem = self.optimization_problem
        optimizer = self.optimizer
        return ParetoFront(
            similarity_tol=self._similarity_tol or 0,
            metric_space=problem.metric_space,
            parameter_space=problem.parameter_space,
            cv_bounds_tol=getattr(optimizer, "cv_bounds_tol", 0.0) or 0.0,
            cv_lincon_tol=getattr(optimizer, "cv_lincon_tol", 0.0) or 0.0,
            cv_lineqcon_tol=getattr(optimizer, "cv_lineqcon_tol", 0.0) or 0.0,
            cv_nonlincon_tol=getattr(optimizer, "cv_nonlincon_tol", 0.0) or 0.0,
        )

    def update_pareto(self, pareto_new: Population | None = None) -> None:
        """
        Update pareto front with new population.

        Parameters
        ----------
        pareto_new : Population, optional
            New pareto front. If None, update existing front with latest population.
        """
        pareto_front = self._create_pareto_front()

        if pareto_new is not None:
            pareto_front.merge(pareto_new)
        else:
            if len(self.pareto_fronts) > 0:
                pareto_front.merge(self.pareto_front)
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
            meta_front = meta_front.drop_similar(self._similarity_tol)
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
    def f_minimized(self) -> np.ndarray:
        """np.ndarray: All evaluated objective function values as if minimized."""
        return self.meta_front.f_minimized

    @property
    def f_best(self) -> np.ndarray:
        """np.ndarray: Best objective function values of optimal points."""
        return self.meta_front.f_best

    @property
    def f_best_indices(self) -> np.ndarray:
        """np.ndarray: Indices of the best objective values."""
        return self.meta_front.f_best_indices

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
        return self.meta_front.plain_metrics

    @property
    def n_evals_history(self) -> np.ndarray:
        """int: Number of evaluations per generation."""
        n_evals = [len(pop) for pop in self.populations]
        return np.cumsum(n_evals)

    @property
    def f_best_history(self) -> np.ndarray:
        """np.ndarray: Best objective values per generation."""
        return np.array([pop.f_best for pop in self.populations])

    @property
    def f_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum objective values per generation."""
        return np.array([pop.f_min for pop in self.populations])

    @property
    def f_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum objective values per generation."""
        return np.array([pop.f_max for pop in self.populations])

    @property
    def f_avg_history(self) -> np.ndarray:
        """np.ndarray: Average objective values per generation."""
        return np.array([pop.f_avg for pop in self.populations])

    @property
    def g_best_history(self) -> np.ndarray:
        """np.ndarray: Best nonlinear constraint per generation."""
        return np.array([pop.g_best for pop in self.populations])

    @property
    def g_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_min for pop in self.populations])

    @property
    def g_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_max for pop in self.populations])

    @property
    def g_avg_history(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.g_avg for pop in self.populations])

    @property
    def cv_nonlincon_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_min for pop in self.populations])

    @property
    def cv_nonlincon_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_max for pop in self.populations])

    @property
    def cv_nonlincon_avg_history(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint violation values per generation."""
        if self.optimization_problem.n_nonlinear_constraints == 0:
            return None
        else:
            return np.array([pop.cv_nonlincon_avg for pop in self.populations])

    @property
    def m_best_history(self) -> np.ndarray:
        """np.ndarray: Best meta scores per generation.

        Meta scores carry no direction annotation; they are evaluated as
        minimized, so "best" is the per-column minimum.
        """
        return np.array(
            [np.min(pop.plain_metrics, axis=0) for pop in self.populations]
        )

    @property
    def m_min_history(self) -> np.ndarray:
        """np.ndarray: Minimum meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        return np.array(
            [np.min(pop.plain_metrics, axis=0) for pop in self.populations]
        )

    @property
    def m_max_history(self) -> np.ndarray:
        """np.ndarray: Maximum meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        return np.array(
            [np.max(pop.plain_metrics, axis=0) for pop in self.populations]
        )

    @property
    def m_avg_history(self) -> np.ndarray:
        """np.ndarray: Average meta scores per generation."""
        if self.optimization_problem.n_meta_scores == 0:
            return None
        return np.array(
            [
                np.mean(np.ma.masked_invalid(pop.plain_metrics), axis=0)
                for pop in self.populations
            ]
        )

    def plot_figures(self) -> None:
        """
        Plot result figures.

        See Also
        --------
        plot_convergence
        plot_objectives
        plot_pairwise
        plot_pareto
        """
        if self.plot_directory is None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.plot_convergence(
                "objectives",
                file_name=f"{self.plot_directory / 'convergence_objectives.png'}",
            )
            if self.optimization_problem.n_nonlinear_constraints > 0:
                self.plot_convergence(
                    "nonlinear_constraints",
                    file_name=f"{self.plot_directory / 'convergence_nonlinear_constraints.png'}",
                )
            if self.optimization_problem.n_meta_scores > 0:
                self.plot_convergence(
                    "meta_scores",
                    file_name=f"{self.plot_directory / 'convergence_meta_scores.png'}",
                )
            self.plot_objectives(
                file_name=f"{self.plot_directory / 'objectives.png'}",
            )
            if self.optimization_problem.n_variables > 1 and len(self.x) > 1:
                self.plot_pairwise(
                    file_name=f"{self.plot_directory / 'pairwise.png'}",
                )

            if self.optimization_problem.n_objectives > 1:
                self.plot_pareto(
                    file_name=f"{self.plot_directory / 'pareto.png'}",
                    plot_evolution=True,
                    plot_pareto=False,
                )

    @plotting.figure_utils
    def plot_objectives(
        self,
        plot_pareto: bool = False,
        *args: Any,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot objective function values for all optimization generations.

        Parameters
        ----------
        plot_pareto : bool, default=False
            If True, only Pareto front members of each generation are plotted.
            Otherwise, all evaluated individuals are plotted.
        *args : Any
            Additional positional arguments passed to `gen.plot_objectives`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `gen.plot_objectives`.

        Returns
        -------
        tuple[plt.Figure, np.ndarray[plt.Axes]]
            A tuple containing:
            - The Matplotlib Figure object.
            - An array of Axes objects representing the subplots.

        See Also
        --------
        CADETProcess.optimization.Population.plot_objectives
        """
        populations = self.pareto_fronts if plot_pareto else self.populations
        values = np.linspace(0, 1, self.n_gen)
        for val, gen in zip(values, populations):
            fig, ax = gen.plot_objectives(
                *args,
                color_feas=cmap_feas(val),
                color_infeas=cmap_infeas(val),
                ax=ax,
                setup_figure_kwargs=setup_figure_kwargs,
                tight_layout=False,
                **kwargs,
            )
        return fig, ax

    @plotting.figure_utils
    def plot_pareto(
        self,
        plot_pareto: bool = True,
        plot_evolution: bool = False,
        *args: Any,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
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
        plot_pareto : bool, default=False
            If True, only Pareto front members of each generation are plotted.
            Otherwise, all evaluated individuals are plotted.
        plot_evolution : bool, optional
            If True, the Pareto front is plotted for each generation.
            Else, only final Pareto front is plotted.
            The default is False.
        *args : Any
            Additional positional arguments passed to `gen.plot_pareto`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `gen.plot_pareto`.

        Returns
        -------
        tuple[plt.Figure, np.ndarray[plt.Axes]]
            A tuple containing:
            - The Matplotlib Figure object.
            - An array of Axes objects representing the subplots.

        See Also
        --------
        CADETProcess.optimization.Population.plot_pareto
        """
        if plot_pareto:
            populations = self.pareto_fronts
            population_last = self.pareto_front
            population_all = Population.concat(
                self.pareto_fronts
            ).drop_duplicates()

        else:
            populations = self.populations
            population_last = self.population_last
            population_all = self.population_all

        if not plot_evolution:
            populations = [population_last]
            color_values = (1.,)  # Explicitly use the last value of the colormap
        else:
            color_values = np.linspace(0, 1, self.n_gen)

        fig, ax, = population_all.plot_pareto(
            *args,
            color_feas="grey",
            plot_scatter=False,
            ax=ax,
            tight_layout=False,
            **{"plot_infeasible": False, **kwargs},
        )

        for color_val, gen in zip(color_values, populations):
            fig, ax = gen.plot_pareto(
                *args,
                color_feas=cmap_feas(color_val),
                color_infeas=cmap_infeas(color_val),
                ax=ax,
                tight_layout=False,
                **{"update_layout": False, **kwargs},
            )
        return fig, ax

    @plotting.figure_utils
    def plot_pairwise(
        self,
        plot_evolution: bool = True,
        *args: Any,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Pairwise of all optimization variables.

        Parameters
        ----------
        plot_evolution : bool, optional
            If True, the Pareto front is plotted for each generation.
            Else, only final Pareto front is plotted.
            The default is True.
        *args : Any
            Additional positional arguments passed to `gen.plot_pareto`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `gen.plot_pareto`.

        Returns
        -------
        tuple[plt.Figure, np.ndarray[plt.Axes]]
            A tuple containing:
            - The Matplotlib Figure object.
            - An array of Axes objects representing the subplots.

        See Also
        --------
        CADETProcess.optimization.Population.plot_pairwise
        """
        populations = self.populations
        population_last = self.population_last
        population_all = self.population_all

        if not plot_evolution:
            populations = [population_last]
            color_values = (1.,)  # Explicitly use the last value of the colormap
        else:
            color_values = np.linspace(0, 1, self.n_gen)

        fig, ax, = population_all.plot_pairwise(
            *args,
            color_feas="grey",
            plot_scatter=False,
            ax=ax,
            setup_figure_kwargs=setup_figure_kwargs,
            tight_layout=False,
            **{"plot_infeasible": False, **kwargs},
        )

        for color_val, gen in zip(color_values, populations):
            fig, ax = gen.plot_pairwise(
                *args,
                color_feas=cmap_feas(color_val),
                color_infeas=cmap_infeas(color_val),
                ax=ax,
                tight_layout=False,
                **{"update_layout": False, **kwargs},
            )

        return fig, ax

    @plotting.figure_utils
    def plot_convergence(
        self,
        target: Literal["objectives", "nonlinear_constraints", "meta_scores"] = "objectives",
        plot_avg: bool = True,
        autoscale: bool = True,
        ax: npt.NDArray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """
        Plot the convergence of optimization metrics over evaluations.

        Parameters
        ----------
        target : Literal["objectives", "nonlinear_constraints", "meta_scores"]
            The target metrics to plot. The default is "objectives".
        plot_avg : bool, default=True
            If True, plot the average trajectory per generation.
        autoscale : bool, default=True
            If True, autoscale the y-axis.
        ax : npt.NDArray[plt.Axes] | None, default=None
            Axes to plot on. If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and array of Axes objects.
        """
        # Determine the number of metrics
        if target == "objectives":
            n = self.optimization_problem.n_objectives
            funcs = self.optimization_problem.objectives
            values_best = self.f_best_history
            values_avg = self.f_avg_history
        elif target == "nonlinear_constraints":
            n = self.optimization_problem.n_nonlinear_constraints
            funcs = self.optimization_problem.nonlinear_constraints
            values_best = self.g_best_history
            values_avg = self.g_avg_history
        elif target == "meta_scores":
            n = self.optimization_problem.n_meta_scores
            funcs = self.optimization_problem.meta_scores
            values_best = self.m_best_history
            values_avg = self.m_avg_history
        else:
            raise CADETProcessError("Unknown target.")

        # Setup figure and axes
        if ax is None:
            fig, axs = plotting.setup_figure(
                **setup_figure_kwargs,
                nrows=1,
                ncols=n,
                squeeze=False,
            )
            axs = axs.reshape((-1,))
        else:
            axs = ax
            fig = axs[0].get_figure()

        counter = 0
        for func in funcs:
            start = counter
            stop = counter + func.n_metrics
            v_func_best = values_best[:, start:stop]
            v_func_avg = values_avg[:, start:stop]

            for i_metric in range(func.n_metrics):
                v_line_best = v_func_best[:, i_metric]
                v_line_avg = v_func_avg[:, i_metric]

                ax_i = axs[counter + i_metric]
                lines = ax_i.get_lines()

                # Update or create lines
                if len(lines) > 0:
                    lines[0].set_xdata(self.n_evals_history)
                    lines[0].set_ydata(v_line_best)
                    if plot_avg and self.population_last.n_individuals > 1:
                        lines[1].set_ydata(v_line_avg)
                else:
                    if plot_avg and self.population_last.n_individuals > 1:
                        label = "best"
                    else:
                        label = None
                    ax_i.plot(
                        self.n_evals_history,
                        v_line_best,
                        "-",
                        color="k",
                        label=label,
                    )
                    if plot_avg and self.population_last.n_individuals > 1:
                        ax_i.plot(
                            self.n_evals_history,
                            v_line_avg,
                            "--",
                            color="k",
                            alpha=0.5,
                            label="avg",
                        )

                # Set x-axis label and limits
                ax_i.set_xlabel("$n_{Evaluations}$")

                # Set y-axis label and scaling
                try:
                    y_label = func.labels[i_metric]
                except AttributeError:
                    y_label = f"{func}_{i_metric}"
                ax_i.set_ylabel(y_label)

                # Apply autoscale and log scaling if needed
                y_min = np.nanmin(v_line_best)
                y_max = np.nanmax(v_line_best)
                if plot_avg and self.population_last.n_individuals > 1:
                    y_min = np.nanmin([v_line_best, v_line_avg])
                    y_max = np.nanmax([v_line_best, v_line_avg])
                if autoscale and y_min > 0:
                    if y_max / y_min > 100.0:
                        ax_i.set_yscale("log")

                ax_i.autoscale()

                # Add legend if needed
                if plot_avg and self.population_last.n_individuals > 1:
                    ax_i.legend()

            counter += func.n_metrics

        return fig, axs

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
        self._population_all = None
        self._similarity_tol = data.get("similarity_tol")

        problem = self.optimization_problem
        metric_space = getattr(problem, "metric_space", None)
        parameter_space = getattr(problem, "parameter_space", None)

        for pop_dict in data["populations"].values():
            pop = Population.from_dict(
                pop_dict,
                metric_space=metric_space,
                parameter_space=parameter_space,
            )
            self.update(pop)

        self._pareto_fronts = [
            ParetoFront.from_dict(
                d, metric_space=metric_space, parameter_space=parameter_space
            )
            for d in data["pareto_fronts"].values()
        ]
        if self._meta_fronts is not None:
            self._meta_fronts = [
                Population.from_dict(
                    d, metric_space=metric_space, parameter_space=parameter_space
                )
                for d in data["meta_fronts"].values()
            ]
        self.time_elapsed = data.get("time_elapsed")
        self.cpu_time = data.get("cpu_time")

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

            x = population.x
            f = population.f
            g = population.g
            m = population.plain_metrics
            for i in range(len(population)):
                row = [*np.asarray(x[i]).tolist(), *f[i].tolist()]
                if g.shape[1] > 0:
                    row += g[i].tolist()
                if m.shape[1] > 0:
                    row += m[i].tolist()
                writer.writerow(row)
