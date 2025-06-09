from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import corner
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from addict import Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pymoo.visualization.scatter import Scatter

from CADETProcess import CADETProcessError, plotting
from CADETProcess.optimization.individual import Individual, hash_array


class Population:
    """
    Collection of Individuals evaluated during Optimization.

    Attributes
    ----------
    individuals : list
        Individuals evaluated during optimization.

    See Also
    --------
    CADETProcess.optimization.Individual
    ParetoFront
    """

    def __init__(self, id: Optional[str] = None) -> None:
        """
        Initialize the Population.

        Parameters
        ----------
        id : str or None, optional
            Identifier for the population. If None, a random UUID will be generated.
        """
        self._individuals = {}

        if id is None:
            self.id = uuid.uuid4()
        else:
            if isinstance(id, bytes):
                id = id.decode(encoding="utf=8")
            self.id = uuid.UUID(id)

    @property
    def feasible(self) -> "Population":
        """Population: Population containing only feasible individuals."""
        pop = Population()
        pop._individuals = {ind.id: ind for ind in self.individuals if ind.is_feasible}

        return pop

    @property
    def infeasible(self) -> "Population":
        """Population: Population containing only infeasible individuals."""
        pop = Population()
        pop._individuals = {
            ind.id: ind for ind in self.individuals if not ind.is_feasible
        }

        return pop

    @property
    def n_x(self) -> int:
        """int: Number of optimization variables."""
        return self.individuals[0].n_x

    @property
    def n_f(self) -> int:
        """int: Number of objective metrics."""
        return self.individuals[0].n_f

    @property
    def n_g(self) -> int:
        """int: Number of nonlinear constraint metrics."""
        return self.individuals[0].n_g

    @property
    def n_m(self) -> int:
        """int: Number of meta scores."""
        return self.individuals[0].n_m

    @property
    def dimensions(self) -> tuple[int]:
        """tuple: Individual dimensions (n_x, n_f, n_g, n_m)."""
        if self.n_individuals == 0:
            return None

        return self.individuals[0].dimensions

    @property
    def objectives_minimization_factors(self) -> np.ndarray:
        """np.ndarray: Array indicating objectives transformed to minimization."""
        return self.individuals[0].objectives_minimization_factors

    @property
    def meta_scores_minimization_factors(self) -> np.ndarray:
        """np.ndarray: Array indicating meta sorces transformed to minimization."""
        return self.individuals[0].meta_scores_minimization_factors

    @property
    def variable_names(self) -> list[str]:
        """list: Names of the optimization variables."""
        if self.individuals[0].variable_names is None:
            return [f"x_{i}" for i in range(self.n_x)]
        else:
            return self.individuals[0].variable_names

    @property
    def independent_variable_names(self) -> list[str]:
        """list: Names of the independent variables."""
        return self.individuals[0].independent_variable_names

    @property
    def objective_labels(self) -> list[str]:
        """list: Labels of the objective metrics."""
        return self.individuals[0].objective_labels

    @property
    def nonlinear_constraint_labels(self) -> list[str]:
        """list: Labels of the nonlinear constraint metrics."""
        return self.individuals[0].nonlinear_constraint_labels

    @property
    def meta_score_labels(self) -> list[str]:
        """list: Labels of the meta scores."""
        return self.individuals[0].meta_score_labels

    def add_individual(
        self,
        individual: Individual,
        ignore_duplicate: bool | None = True,
    ) -> None:
        """
        Add individual to population.

        Parameters
        ----------
        individual : Individual
            Individual to be added.
        ignore_duplicate : bool, optional
            If False, an Exception is thrown if the individual already exists.

        Raises
        ------
        TypeError
            If the individual is not an instance of Individual.
        CADETProcessError
            If the individual does not match the dimensions.
            If the individual already exists.
        """
        if not isinstance(individual, Individual):
            raise TypeError("Expected Individual")

        if self.dimensions is not None and individual.dimensions != self.dimensions:
            raise CADETProcessError("Individual does not match dimensions.")

        if individual in self:
            if ignore_duplicate:
                return
            else:
                raise CADETProcessError("Individual already exists.")

        self._individuals[individual.id] = individual

    def remove_individual(self, individual: Individual) -> None:
        """
        Remove an individual from the population.

        Parameters
        ----------
        individual : Individual
            Individual to be removed.

        Raises
        ------
        TypeError
            If the individual is not an instance of Individual.
        CADETProcessError
            If the individual is not in the population.
        """
        if not isinstance(individual, Individual):
            raise TypeError("Expected Individual")

        if individual not in self:
            raise CADETProcessError("Individual is not in population.")
        self._individuals.pop(individual.id)

    def update(self, other: Population) -> None:
        """
        Update the population with individuals from another population.

        Parameters
        ----------
        other : Population
            Another population.

        Raises
        ------
        TypeError
            If other is not an instance of Population.
        CADETProcessError
            If the dimensions do not match.
        """
        if not isinstance(other, Population):
            raise TypeError("Expected Population")

        if self.dimensions is not None and self.dimensions != other.dimensions:
            raise CADETProcessError("Dimensions do not match")

        self._individuals.update(other._individuals)

    def remove_similar(self) -> None:
        """Remove similar individuals from the population."""
        for ind in self.individuals.copy():
            to_remove = []

            for ind_other in self.individuals.copy():
                if ind is ind_other:
                    continue

                if ind_other.is_similar(ind, self.similarity_tol):
                    if np.any(ind_other.f == self.f_best):
                        continue
                    to_remove.append(ind_other)

            for i in reversed(to_remove):
                try:
                    self.remove_individual(i)
                except CADETProcessError:
                    pass

    @property
    def individuals(self) -> list[Individual]:
        """list: All individuals."""
        return list(self._individuals.values())

    @property
    def n_individuals(self) -> int:
        """int: Number of indivuals."""
        return len(self.individuals)

    @property
    def x(self) -> np.ndarray:
        """np.ndarray: All evaluated points."""
        return np.array([ind.x for ind in self.individuals])

    @property
    def x_transformed(self) -> np.ndarray:
        """np.ndarray: All evaluated points in independent transformed space."""
        return np.array([ind.x_transformed for ind in self.individuals])

    @property
    def cv_bounds(self) -> np.ndarray:
        """np.ndarray: All evaluated bound constraint violations."""
        return np.array([ind.cv_bounds for ind in self.individuals])

    @property
    def cv_lincon(self) -> np.ndarray:
        """np.ndarray: All evaluated linear constraint violations."""
        return np.array([ind.cv_lincon for ind in self.individuals])

    @property
    def cv_lineqcon(self) -> np.ndarray:
        """np.ndarray: All evaluated linear equality constraint violations."""
        return np.array([ind.cv_lineqcon for ind in self.individuals])

    @property
    def f(self) -> np.ndarray:
        """np.ndarray: All evaluated objective function values."""
        return np.array([ind.f for ind in self.individuals])

    @property
    def f_minimized(self) -> np.ndarray:
        """np.ndarray: All evaluated objective function values as if minimized."""
        return np.array([ind.f_min for ind in self.individuals])

    @property
    def f_best(self) -> np.ndarray:
        """np.ndarray: Best objective values."""
        f_best = np.min(self.f_minimized, axis=0)
        return np.multiply(self.objectives_minimization_factors, f_best)

    @property
    def f_min(self) -> np.ndarray:
        """np.ndarray: Minimum objective values."""
        return np.min(self.f, axis=0)

    @property
    def f_max(self) -> np.ndarray:
        """np.ndarray: Maximum objective values."""
        return np.max(self.f, axis=0)

    @property
    def f_avg(self) -> np.ndarray:
        """np.ndarray: Average objective values."""
        return np.mean(self.f, axis=0)

    @property
    def g(self) -> np.ndarray:
        """np.ndarray: All evaluated nonlinear constraint function values."""
        if self.dimensions[2] > 0:
            return np.array([ind.g for ind in self.individuals])

    @property
    def g_best(self) -> np.ndarray:
        """np.ndarray: Best nonlinear constraint values."""
        indices = np.argmin(self.cv_nonlincon, axis=0)
        return [self.g[ind, i] for i, ind in enumerate(indices)]

    @property
    def g_min(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint values."""
        if self.dimensions[2] > 0:
            return np.min(self.g, axis=0)

    @property
    def g_max(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinear constraint values."""
        if self.dimensions[2] > 0:
            return np.max(self.g, axis=0)

    @property
    def g_avg(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint values."""
        if self.dimensions[2] > 0:
            return np.mean(self.g, axis=0)

    @property
    def cv_nonlincon(self) -> np.ndarray:
        """np.ndarray: All evaluated nonlinear constraint violation values."""
        if self.dimensions[2] > 0:
            return np.array([ind.cv_nonlincon for ind in self.individuals])

    @property
    def cv_nonlincon_min(self) -> np.ndarray:
        """np.ndarray: Minimum nonlinear constraint violation values."""
        if self.dimensions[2] > 0:
            return np.min(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_max(self) -> np.ndarray:
        """np.ndarray: Maximum nonlinearconstraint violation values."""
        if self.dimensions[2] > 0:
            return np.max(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_avg(self) -> np.ndarray:
        """np.ndarray: Average nonlinear constraint violation values."""
        if self.dimensions[2] > 0:
            return np.mean(self.cv_nonlincon, axis=0)

    @property
    def m(self) -> np.ndarray:
        """np.ndarray: All evaluated meta scores."""
        if self.dimensions[3] > 0:
            return np.array([ind.m for ind in self.individuals])

    @property
    def m_minimized(self) -> np.ndarray:
        """np.ndarray: All evaluated meta scores, transformed to be minimized."""
        if self.dimensions[3] > 0:
            return np.array([ind.m_min for ind in self.individuals])

    @property
    def m_best(self) -> np.ndarray:
        """np.ndarray: Best meta scores."""
        if self.dimensions[3] > 0:
            m_best = np.min(self.m_minimized, axis=0)
            return np.multiply(self.meta_scores_minimization_factors, m_best)

    @property
    def m_min(self) -> np.ndarray:
        """np.ndarray: Minimum meta scores."""
        if self.dimensions[3] > 0:
            return np.min(self.m, axis=0)

    @property
    def m_max(self) -> np.ndarray:
        """np.ndarray: Maximum meta scores."""
        if self.dimensions[3] > 0:
            return np.max(self.m, axis=0)

    @property
    def m_avg(self) -> np.ndarray:
        """np.ndarray: Average meta scores."""
        if self.dimensions[3] > 0:
            return np.mean(self.m, axis=0)

    @property
    def is_feasilbe(self) -> bool:
        """np.ndarray: False if any constraint is not met. True otherwise."""
        return np.array([ind.is_feasible for ind in self.individuals])

    def setup_objectives_figure(
        self,
        include_meta: Optional[bool] = True,
        plot_individual: Optional[bool] = False,
    ) -> tuple:
        """
        Set up figure and axes for plotting objectives.

        Parameters
        ----------
        include_meta : bool, optional
            If True, include meta scores in the plot. The default is True.
        plot_individual : bool, optional
            If True, create separate figures for each objective.
            Otherwise, plot all objectives in one figure. The default is True.

        Returns
        -------
        tuple
            A tuple of the figure(s) and axes object(s).
        """
        n = len(self.variable_names)
        if include_meta and self.m is not None:
            m = len(self.objective_labels) + len(self.meta_score_labels)
        else:
            m = len(self.objective_labels)

        if n == 0:
            return (None, None)

        space_fig_all, space_axs_all = plt.subplots(
            nrows=m,
            ncols=n,
            figsize=(n * 8 + 2, m * 8 + 2),
            squeeze=False,
        )
        plt.close(space_fig_all)

        space_figs_ind = []
        space_axs_ind = []
        for i in range(m * n):
            fig, ax = plt.subplots()
            space_figs_ind.append(fig)
            space_axs_ind.append(ax)
            plt.close(fig)

        space_axs_ind = np.array(space_axs_ind).reshape(space_axs_all.shape)

        if plot_individual:
            return space_figs_ind, space_axs_ind
        else:
            return space_fig_all, space_axs_all

    def plot_objectives(
        self,
        figs: Optional[Figure | list[Figure]] = None,
        axs: Optional[Axes | list[list[Axes]]] = None,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        plot_individual: bool = False,
        autoscale: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        show: bool = True,
        plot_directory: Optional[str | Path] = None,
    ) -> tuple[list[Figure], list[list[Axes]]]:
        """
        Plot the objective function values for each design variable.

        Parameters
        ----------
        figs : plt.Figure or list, optional
            Figure(s) to plot the objectives on. The default is None.
        axs : plt.Axes or list, optional
            Axes to plot the objectives on. The default is None.
        include_meta : bool, optional
            If True, include meta scores in the plot. The default is True.
        plot_infeasible : bool, optional
            If True, plot infeasible points. The default is True.
        plot_individual : bool, optional
            If True, create separate figures for each objective.
            Otherwise, plot all objectives in one figure. The default is False.
        autoscale : bool, optional
            If True, automatically adjust the scaling of the axes. The default is True.
        color_feas : str, optional
            The color for the feasible points. The default is 'blue'.
        color_infeas : str, optional
            The color for the infeasible points. The default is 'red'.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved. The default is None.

        Returns
        -------
        tuple
            A tuple of the figure(s) and axes object(s).
        """
        if axs is None:
            figs, axs = self.setup_objectives_figure(include_meta, plot_individual)

        if not isinstance(figs, list):
            figs = [figs]

        layout = plotting.Layout()
        layout.y_label = "$f~/~-$"

        variables = self.variable_names
        feasible = self.feasible
        infeasible = self.infeasible
        x_feas = feasible.x
        x_infeas = infeasible.x

        if include_meta and self.m is not None:
            if len(feasible) > 0:
                values_feas = np.hstack((feasible.f, feasible.m))
            else:
                values_infeas = np.empty((0, self.n_f + self.n_m))
            if len(infeasible) > 0:
                values_infeas = np.hstack((infeasible.f, infeasible.m))
            else:
                values_infeas = np.empty((0, self.n_f + self.n_m))

            labels = self.objective_labels + self.meta_score_labels
        else:
            values_feas = feasible.f
            values_infeas = infeasible.f
            labels = self.objective_labels

        for i_var, var in enumerate(variables):
            if len(feasible) > 0:
                x_var_feas = x_feas[:, i_var]
            if len(infeasible) > 0:
                x_var_infeas = x_infeas[:, i_var]

            for i_metric, label in enumerate(labels):
                ax = axs[i_metric][i_var]

                if len(feasible) > 0:
                    v_metric_feas = values_feas[:, i_metric]
                    ax.scatter(x_var_feas, v_metric_feas, alpha=0.5, color=color_feas)

                if len(infeasible) > 0 and plot_infeasible:
                    v_metric_infeas = values_infeas[:, i_metric]
                    ax.scatter(
                        x_var_infeas, v_metric_infeas, alpha=0.5, color=color_infeas
                    )

                points = np.vstack([col.get_offsets() for col in ax.collections])

                x_all = points[:, 0]
                v_all = points[:, 1]

                layout.x_lim = (np.nanmin(x_all), np.nanmax(x_all))
                layout.x_label = var
                if autoscale and np.min(x_all) > 0:
                    if np.max(x_all) / np.min(x_all[x_all > 0]) > 100.0:
                        ax.set_xscale("log")
                        layout.x_label = f"$log_{{10}}$({var})"

                y_min = np.nanmin(v_all)
                y_max = np.nanmax(v_all)
                y_lim = (min(0.9 * y_min, y_min - 0.01 * (y_max - y_min)), 1.1 * y_max)
                layout.y_label = label
                if autoscale and np.min(v_all) > 0:
                    if np.max(v_all) / np.min(v_all[v_all > 0]) > 100.0:
                        ax.set_yscale("log")
                        layout.y_label = f"$log_{{10}}$({label})"
                        y_lim = (y_min / 2, y_max * 2)
                if y_min != y_max:
                    layout.y_lim = y_lim

                try:
                    plotting.set_layout(ax, layout)
                except ValueError:
                    pass

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
                    fig.savefig(f"{plot_directory / 'objectives'}_{i}.png")
            else:
                figs[0].savefig(f"{plot_directory / 'objectives'}.png")

        if plot_individual:
            return figs, axs
        else:
            return figs[0], axs

    def setup_pareto(self, include_meta: bool = False) -> Scatter:
        """
        Set up base figure for plotting the Pareto front.

        Parameters
        ----------
        include_meta : bool
            If True, include meta scores in Pareto plot.

        Returns
        -------
        pymoo.visualization.scatter.Scatter
            The base figure object.
        """
        if include_meta:
            n = self.dimensions[1] + self.dimensions[3]
            labels = self.objective_labels + self.meta_score_labels
        else:
            n = self.dimensions[1]
            labels = self.objective_labels
        plot = Scatter(
            figsize=(6 * n, 5 * n),
            tight_layout=True,
            plot_3d=False,
            labels=labels,
        )
        return plot

    def plot_pareto(
        self,
        plot: Optional[Scatter] = None,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        show: bool = True,
        plot_directory: Optional[str | Path] = None,
    ) -> Scatter:
        """
        Plot pairwise Pareto fronts for each generation in the optimization.

        The Pareto front represents the optimal solutions that cannot be improved in one
        objective without sacrificing another. The method shows a pairwise Pareto plot,
        where each objective is plotted against every other objective in a scatter plot,
        allowing for a visualization of the trade-offs between the objectives.

        Parameters
        ----------
        plot : pymoo.visualization.scatter.Scatter, optional
            Base figure. If None is provided, a new one will be set up.
        include_meta : bool, optional
            If True, include meta scores in the plot. The default is True.
        plot_infeasible : bool, optional
            If True, plot infeasible points. The default is True.
        color_feas : str, optional
            The color for the feasible points. The default is 'blue'.
        color_infeas : str, optional
            The color for the infeasible points. The default is 'red'.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved. The default is None.

        Returns
        -------
        pymoo.visualization.scatter.Scatter
            The scatter plot object.
        """
        if plot is None:
            plot = self.setup_pareto(include_meta)

        feasible = self.feasible
        infeasible = self.infeasible

        if include_meta and self.m is not None:
            if len(feasible) > 0:
                values_feas = np.hstack((feasible.f, feasible.m))
            else:
                values_infeas = np.empty((0, self.n_f + self.n_m))
            if len(infeasible) > 0:
                values_infeas = np.hstack((infeasible.f, infeasible.m))
            else:
                values_infeas = np.empty((0, self.n_f + self.n_m))
        else:
            values_feas = feasible.f
            values_infeas = infeasible.f

        if len(feasible) > 0:
            plot.add(values_feas, s=10, color=color_feas)

        if plot_infeasible and len(infeasible) > 0:
            plot.add(values_infeas, s=10, color=color_infeas)

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            plot.save(f"{plot_directory / 'pareto.png'}")

        if not show:
            plt.close(plot.fig)
        else:
            plot.show()

        return plot

    def plot_pairwise(
        self,
        fig: Optional[plt.Figure] = None,
        axs: Optional[npt.NDArray[plt.Axes]] = None,
        n_bins: int = 20,
        use_transformed: bool = False,
        autoscale: bool = True,
        show: bool = True,
        plot_directory: Optional[str] = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Create a pairplot using Matplotlib.

        Parameters
        ----------
        fig : Optional[plt.Figure], default=None
            An optional Matplotlib Figure object. If none is provided, a new figure will
            be created.
        axs : Optional[npt.NDArray[plt.Axes]], default=None
            An optional array of Matplotlib Axes. If none is provided, new axes will
            be created.
        n_bins : int, default=20
            Number of bins for histogram plots.
        use_transformed : bool, optional
            If True, use the transformed independent variables. The default is False.
        autoscale : bool, optional
            If True, automatically adjust the scaling of the axes. The default is True.
        use_transformed : bool, optional
            If True, transformed values will be plotted. The default is False.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved. The default is None.

        Returns
        -------
        tuple
            A tuple containing:
            - plt.Figure: The Matplotlib Figure object.
            - np.ndarray: An array of Axes objects representing the subplot grid.
        """
        if use_transformed:
            x = self.x_transformed
            labels = self.independent_variable_names
        else:
            x = self.x
            labels = self.variable_names

        fig, axs = plot_pairwise(
            x,
            labels,
            n_bins=n_bins,
            autoscale=autoscale,
            fig=fig,
            axs=axs,
        )

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            fig.savefig(f"{plot_directory / 'pairwise.png'}")

        if not show:
            plt.close(fig)

        return fig, axs

    def plot_corner(
        self,
        use_transformed: bool = False,
        show: bool = True,
        plot_directory: Optional[str] = None,
    ) -> None:
        """
        Create a corner plot of the independent variables.

        Parameters
        ----------
        use_transformed : bool, optional
            If True, use the transformed independent variables. The default is False.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved. The default is None.
        """
        warnings.warn(
            "This method will be deprecated in the future. "
            "Use `plot_pairwise` instead.",
            FutureWarning,
        )

        if use_transformed:
            x = self.x_transformed
            labels = self.independent_variable_names
        else:
            x = self.x
            labels = self.variable_names

        # To avoid error, remove dimensions where all entries are the same value.
        singular_indices = []
        singular_labels = []

        for i, col in enumerate(x.transpose()):
            if len(np.unique(col)) == 1:
                singular_indices.append(i)
                singular_labels.append(labels[i])

        x = np.delete(x.transpose(), singular_indices, 0).transpose()
        labels = [label for label in labels if label not in singular_labels]

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
        fig_size = 6 * len(labels)
        fig.set_size_inches((fig_size, fig_size))
        fig.tight_layout()

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            fig.savefig(f"{plot_directory / 'corner.png'}")

        if not show:
            plt.close(fig)

    def __contains__(self, other: Individual | np.ndarray | list) -> bool:
        """
        Check if the population contains a specific individual.

        Parameters
        ----------
        other : Individual | np.ndarray | list
            The individual or its hashable representation.

        Returns
        -------
        bool
            True if the individual is in the population, False otherwise.
        """
        if isinstance(other, Individual):
            key = other.id
        elif isinstance(other, (np.array, list)):
            key = hash_array(other)
        else:
            key = None

        if key in self._individuals:
            return True
        else:
            return False

    def __getitem__(self, x: np.ndarray | list) -> Individual:
        """
        Get an individual from the population using its hashable representation.

        Parameters
        ----------
        x : np.ndarray | list
            The hashable representation of the individual.

        Returns
        -------
        Individual
            The individual from the population.
        """
        key = hash_array(x)

        return self._individuals[key]

    def __len__(self) -> int:
        """
        Get the number of individuals in the population.

        Returns
        -------
        int
            The number of individuals in the population.
        """
        return self.n_individuals

    def __iter__(self) -> Iterator[Individual]:
        """
        Iterate over the individuals in the population.

        Returns
        -------
        iter
            An iterator over the individuals in the population.
        """
        return iter(self.individuals)

    def to_dict(self) -> Dict:
        """
        Convert Population to a dictionary.

        Returns
        -------
        dict
            Population as a dictionary with individuals stored as list of dictionaries.
        """
        data = Dict()
        data.id = str(self.id)

        for i, ind in enumerate(self.individuals):
            data.individuals[i] = ind.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> Population:
        """
        Create a Population from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary containing population data.

        Returns
        -------
        Population
            The Population created from the data.
        """
        id = data["id"]
        if isinstance(id, bytes):
            id = id.decode(encoding="utf=8")
        population = cls(id)
        for individual_data in data["individuals"].values():
            individual = Individual.from_dict(individual_data)
            population.add_individual(individual)
        return population


class ParetoFront(Population):
    """Class representing a Pareto front in a multi-objective optimization problem."""

    def __init__(
        self,
        similarity_tol: float = 1e-1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a ParetoFront with a specified similarity tolerance.

        Parameters
        ----------
        similarity_tol : float, optional
            Tolerance for similarity between individuals. Default is 1e-1.
        *args : tuple
            Additional positional arguments for the parent class.
        **kwargs : dict
            Additional keyword arguments for the parent class.
        """
        self.similarity_tol = similarity_tol
        super().__init__(*args, **kwargs)

    def update_population(self, population: Population) -> tuple[list, bool]:
        """
        Update the Pareto front with a new population.

        Parameters
        ----------
        population : Population
            The population used to update the Pareto front.

        Returns
        -------
        tuple[list, bool]
            A tuple containing new members added to the Pareto front and a boolean indicating
            if there was a significant improvement.
        """
        new_members = []
        significant = []

        for ind_new in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []

            if not ind_new.is_feasible:
                continue

            for i, ind_pareto in enumerate(self):
                # Do not add if is dominated
                if not dominates_one and ind_pareto.dominates(ind_new):
                    is_dominated = True
                    break

                # Remove existing if infeasible
                elif not ind_pareto.is_feasible:
                    dominates_one = True
                    to_remove.append(ind_pareto)
                    significant.append(True)

                # Remove existing if new dominates
                elif ind_new.dominates(ind_pareto):
                    dominates_one = True
                    to_remove.append(ind_pareto)
                    if not ind_new.is_similar(ind_pareto, self.similarity_tol):
                        significant.append(True)

                # Ignore similar individuals
                elif ind_new.is_similar(ind_pareto, self.similarity_tol):
                    has_twin = True
                    break

            for i in reversed(to_remove):
                self.remove_individual(i)

            if not is_dominated:
                if len(self) == 0:
                    significant.append(True)
                if not has_twin:
                    significant.append(True)

                self.add_individual(ind_new)
                new_members.append(ind_new)

        if len(self) == 0:
            # Use least inveasible individuals.
            indices = np.argmin(population.cv_bounds, axis=0)
            for index in indices:
                ind_new = population.individuals[index]
                self.add_individual(ind_new)

            indices = np.argmin(population.cv_lincon, axis=0)
            for index in indices:
                ind_new = population.individuals[index]
                self.add_individual(ind_new)

            indices = np.argmin(population.cv_lineqcon, axis=0)
            for index in indices:
                ind_new = population.individuals[index]
                self.add_individual(ind_new)

            if self.n_g > 0:
                indices = np.argmin(population.cv_nonlincon, axis=0)
                for index in indices:
                    ind_new = population.individuals[index]
                    self.add_individual(ind_new)

        elif len(self) > 1:
            self.remove_infeasible()

        if self.similarity_tol:
            self.remove_similar()

        return new_members, any(significant)

    def remove_infeasible(self) -> None:
        """Remove infeasible individuals from the Pareto front."""
        for ind in self.individuals.copy():
            if not ind.is_feasible:
                self.remove_individual(ind)

    def remove_dominated(self) -> None:
        """Remove dominated individuals from the Pareto front."""
        for ind in self.individuals.copy():
            dominates_one = False
            to_remove = []

            for ind_other in self.individuals.copy():
                if not dominates_one and ind_other.dominates(ind):
                    to_remove.append(ind)
                    break
                elif ind.dominates(ind_other):
                    dominates_one = True
                    to_remove.append(ind_other)

            for i in reversed(to_remove):
                try:
                    self.remove_individual(i)
                except CADETProcessError:
                    pass

    def to_dict(self) -> dict:
        """
        Convert the ParetoFront to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the ParetoFront, including individuals and
            similarity tolerance if set.
        """
        front = super().to_dict()
        if self.similarity_tol:
            front["similarity_tol"] = self.similarity_tol

        return front

    @classmethod
    def from_dict(cls, data: dict) -> ParetoFront:
        """
        Create a ParetoFront instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the ParetoFront data.

        Returns
        -------
        ParetoFront
            An instance of ParetoFront created from the dictionary.
        """
        front = cls(similarity_tol=data.get("similarity_tol"), id=data["id"])
        for individual_data in data["individuals"].values():
            individual = Individual.from_dict(individual_data)
            front.add_individual(individual)

        return front


def plot_pairwise(
    population: npt.ArrayLike,
    variable_names: Optional[list[str]] = None,
    n_bins: int = 20,
    autoscale: bool = True,
    fig: Optional[plt.Figure] = None,
    axs: Optional[np.ndarray[plt.Axes]] = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Create a pairwise scatter plot for all variables of a population.

    Parameters
    ----------
    population : npt.ArrayLike
        3D array-like structure containing numerical variables with shape
        (n_chains, n_samples, n_variables)
    variable_names : list of str, optional
        list of variable names corresponding to columns in the data.
        If None, default names will be assigned.
    n_bins : int, default=20
        Number of bins for histogram plots.
    autoscale : bool, default=True
        If True, automatically adjust the scaling of the axes.
    fig : Optional[plt.Figure], default=None
        An optional Matplotlib Figure object. If none is provided, a new figure will be
        created.
    axs : Optional[npt.NDArray[plt.Axes]], default=None
        An optional array of Matplotlib Axes. If none is provided, new axes will be
        created.

    Returns
    -------
    tuple
        A tuple containing:
        - plt.Figure: The Matplotlib Figure object.
        - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplot grid.
    """
    population = np.array(population)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    n_samples, n_variables = population.shape

    if variable_names is None:
        variable_names = [f"$x_{i}$" for i in range(n_variables)]

    if fig is None and axs is None:
        fig, axs = plt.subplots(
            n_variables,
            n_variables,
            figsize=(6 * n_variables, 5 * n_variables),
            sharex="col",
            sharey="row",
            squeeze=False,
        )

    if axs.shape != (n_variables, n_variables):
        raise ValueError(
            "Inconsistent shape for provided axes."
            f"Expected {(n_variables, n_variables)}, got {axs.shape}."
        )

    # Rows
    for i in range(n_variables):
        scale_i = False
        if autoscale and np.all(population[:, i] > 0):
            value_range = population[:, i].max() / population[:, i].min()
            if value_range > 100.0:
                scale_i = True

        # Columns
        for j in range(n_variables):
            scale_j = False
            if autoscale and np.all(population[:, j] > 0):
                value_range = population[:, j].max() / population[:, j].min()
                if value_range > 100.0:
                    scale_j = True

            ax = axs[i, j]
            if i == j:
                # Create a twin axis for histograms to avoid sharing y-axis
                ax_hist = ax.twinx()

                # Determine binning strategy
                if scale_i:
                    bins = np.geomspace(
                        population[:, i].min(), population[:, i].max(), n_bins + 1
                    )
                else:
                    bins = np.linspace(
                        population[:, i].min(), population[:, i].max(), n_bins + 1
                    )

                ax_hist.hist(
                    population[:, i],
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                    align="mid",
                )
                ax_hist.set_yticks([])  # Hide y-ticks for the histogram
            else:
                # Scatter plot for non-diagonal elements
                ax.scatter(population[:, j], population[:, i], alpha=0.5, s=10)

            # Apply log scale based on autoscale logic
            if scale_j:
                ax.set_xscale("log")
            if scale_i:
                ax.set_yscale("log")

            # Ensure axis labels and ticks are visible only on the
            # first column
            if j == 0:
                ax.yaxis.set_tick_params(labelleft=True)
                if not scale_i:
                    ax.ticklabel_format(axis="y", useMathText=True, scilimits=[-3, 3])
            else:
                ax.yaxis.set_tick_params(labelleft=False)

            # and last row
            if i == n_variables - 1:
                ax.xaxis.set_tick_params(labelbottom=True)
                if not scale_j:
                    ax.ticklabel_format(axis="x", useMathText=True, scilimits=[-3, 3])
            else:
                ax.xaxis.set_tick_params(labelbottom=False)

            # Set axis labels on the edges
            if i == n_variables - 1:
                ax.set_xlabel(variable_names[j])
            if j == 0:
                ax.set_ylabel(variable_names[i])

    fig.tight_layout()

    return fig, axs
