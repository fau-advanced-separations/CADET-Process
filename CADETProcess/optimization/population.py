from __future__ import annotations

import uuid
from typing import Any, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from addict import Dict

from CADETProcess import CADETProcessError, plotting
from CADETProcess.optimization.individual import Individual, hash_array

__all__ = ["Population", "ParetoFront"]


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
        masked_f = np.ma.masked_invalid(self.f)
        return np.mean(masked_f, axis=0)

    @property
    def f_minimized(self) -> np.ndarray:
        """np.ndarray: All evaluated objective function values as if minimized."""
        return np.array([ind.f_minimized for ind in self.individuals])

    @property
    def f_best(self) -> np.ndarray:
        """np.ndarray: Best objective values."""
        f_best = np.min(self.f_minimized, axis=0)
        return np.multiply(self.objectives_minimization_factors, f_best)

    @property
    def f_best_indices(self) -> np.ndarray:
        """np.ndarray: Indices of the best objective values."""
        return np.argmin(self.f_minimized, axis=0)

    @property
    def g(self) -> np.ndarray | None:
        """np.ndarray: All evaluated nonlinear constraint function values."""
        if self.n_g > 0:
            return np.array([ind.g for ind in self.individuals])

    @property
    def g_min(self) -> np.ndarray | None:
        """np.ndarray: Minimum nonlinear constraint values."""
        if self.n_g > 0:
            return np.min(self.g, axis=0)

    @property
    def g_max(self) -> np.ndarray | None:
        """np.ndarray: Maximum nonlinear constraint values."""
        if self.n_g > 0:
            return np.max(self.g, axis=0)

    @property
    def g_avg(self) -> np.ndarray | None:
        """np.ndarray: Average nonlinear constraint values."""
        if self.n_g > 0:
            masked_g = np.ma.masked_invalid(self.g)
            return np.mean(masked_g, axis=0)

    @property
    def g_best(self) -> np.ndarray | None:
        """np.ndarray: Best nonlinear constraint values."""
        indices = np.argmin(self.cv_nonlincon, axis=0)
        return [self.g[ind, i] for i, ind in enumerate(indices)]

    @property
    def cv_nonlincon(self) -> np.ndarray | None:
        """np.ndarray: All evaluated nonlinear constraint violation values."""
        if self.n_g > 0:
            return np.array([ind.cv_nonlincon for ind in self.individuals])

    @property
    def cv_nonlincon_min(self) -> np.ndarray | None:
        """np.ndarray: Minimum nonlinear constraint violation values."""
        if self.n_g > 0:
            return np.min(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_max(self) -> np.ndarray | None:
        """np.ndarray: Maximum nonlinearconstraint violation values."""
        if self.n_g > 0:
            return np.max(self.cv_nonlincon, axis=0)

    @property
    def cv_nonlincon_avg(self) -> np.ndarray | None:
        """np.ndarray: Average nonlinear constraint violation values."""
        if self.n_g > 0:
            masked_cv_nonlincon = np.ma.masked_invalid(self.cv_nonlincon)
            return np.mean(masked_cv_nonlincon, axis=0)

    @property
    def m(self) -> np.ndarray | None:
        """np.ndarray: All evaluated meta scores."""
        if self.n_m > 0:
            return np.array([ind.m for ind in self.individuals])

    @property
    def m_min(self) -> np.ndarray | None:
        """np.ndarray: Minimum meta scores."""
        if self.n_m > 0:
            return np.min(self.m, axis=0)

    @property
    def m_max(self) -> np.ndarray | None:
        """np.ndarray: Maximum meta scores."""
        if self.n_m > 0:
            return np.max(self.m, axis=0)

    @property
    def m_avg(self) -> np.ndarray | None:
        """np.ndarray: Average meta scores."""
        if self.n_m > 0:
            masked_m = np.ma.masked_invalid(self.m)
            return np.mean(masked_m, axis=0)

    @property
    def m_minimized(self) -> np.ndarray | None:
        """np.ndarray: All evaluated meta scores, transformed to be minimized."""
        if self.n_m > 0:
            return np.array([ind.m_minimized for ind in self.individuals])

    @property
    def m_best(self) -> np.ndarray | None:
        """np.ndarray: Best meta scores."""
        if self.n_m > 0:
            m_best = np.min(self.m_minimized, axis=0)
            return np.multiply(self.meta_scores_minimization_factors, m_best)

    @property
    def m_best_indices(self) -> np.ndarray | None:
        """np.ndarray: Indices of the best meta scores."""
        if self.n_m > 0:
            return np.argmin(self.m_minimized, axis=0)

    @property
    def is_feasilbe(self) -> bool:
        """np.ndarray: False if any constraint is not met. True otherwise."""
        return np.array([ind.is_feasible for ind in self.individuals])

    @plotting.figure_utils
    def plot_objectives(
        self,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        autoscale: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        ax: npt.NDArray[plt.Axes] | None = None,
        setup_figure_kwargs: Optional[dict] = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """
        Plot the objective function values for each design variable.

        Parameters
        ----------
        include_meta : bool, default=True
            If True, include meta scores in the plot.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        autoscale : bool, default=True
            If True, automatically adjust the scaling of the axes.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        if self.n_x == 0:
            raise CADETProcessError("Cannot plot without individuals.")

        m = self.n_f
        if include_meta and self.m is not None:
            m += self.n_m

        if ax is None:
            fig, axs = plotting.setup_figure(
                **setup_figure_kwargs,
                nrows=m,
                ncols=self.n_x,
                aspect=1,
                squeeze=False,
            )
        else:
            axs = ax
            fig = axs[0, 0].get_figure()

        variables = self.variable_names
        feasible = self.feasible
        infeasible = self.infeasible
        x_feas = feasible.x
        x_infeas = infeasible.x

        if include_meta and self.m is not None:
            if len(feasible) > 0:
                values_feas = np.hstack((feasible.f, feasible.m))
            else:
                values_feas = np.empty((0, self.n_f + self.n_m))
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
                ax_ij = axs[i_metric, i_var]

                # Plot feasible/infeasible points
                if len(feasible) > 0:
                    v_metric_feas = values_feas[:, i_metric]
                    ax_ij.scatter(x_var_feas, v_metric_feas, alpha=0.5, color=color_feas)
                if len(infeasible) > 0 and plot_infeasible:
                    v_metric_infeas = values_infeas[:, i_metric]
                    ax_ij.scatter(x_var_infeas, v_metric_infeas, alpha=0.5, color=color_infeas)

                # Set axis labels and limits
                points = np.vstack([col.get_offsets() for col in ax_ij.collections])
                x_all = points[:, 0]
                v_all = points[:, 1]

                ax_ij.set_xlabel(var)
                ax_ij.set_ylabel(label)
                ax_ij.set_xlim(np.nanmin(x_all), np.nanmax(x_all))

                if autoscale and np.min(x_all) > 0:
                    if np.max(x_all) / np.min(x_all[x_all > 0]) > 100.0:
                        ax_ij.set_xscale("log")

                # Replace inf with nan
                mask = np.isfinite(v_all)
                v_all = v_all[mask]

                # Scale axis
                y_min = np.nanmin(v_all)
                y_max = np.nanmax(v_all)
                if y_min != y_max:
                    if autoscale and np.min(v_all) > 0:
                        if np.max(v_all) / np.min(v_all[v_all > 0]) > 100.0:
                            ax_ij.set_yscale("log")

                ax_ij.autoscale()

        return fig, axs

    @plotting.figure_utils
    def plot_pareto(
        self,
        include_meta: bool = True,
        plot_infeasible: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        *args: Any,
        ax: np.ndarray[plt.Axes] | None = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,

    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """
        Plot pairwise Pareto fronts for each generation in the optimization.

        Parameters
        ----------
        include_meta : bool, default=True
            If True, include meta scores in the plot.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        *args : Any
            Additional positional arguments passed to `plot_pairwise`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `plot_pairwise`.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        if include_meta:
            labels = self.objective_labels + self.meta_score_labels
        else:
            labels = self.objective_labels

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
            fig, ax = plot_pairwise(
                values_feas,
                labels,
                color=color_feas,
                *args,
                ax=ax,
                setup_figure_kwargs=setup_figure_kwargs,
                tight_layout=False,
                **kwargs,
            )
        if plot_infeasible and len(infeasible) > 0:
            fig, ax = plot_pairwise(
                values_infeas,
                labels,
                color=color_infeas,
                *args,
                ax=ax,
                tight_layout=False,
                **({"update_layout": False, **kwargs})
            )

        return fig, ax

    @plotting.figure_utils
    def plot_pairwise(
        self,
        use_transformed: bool = False,
        plot_infeasible: bool = True,
        color_feas: str = "blue",
        color_infeas: str = "red",
        *args: Any,
        ax: Optional[npt.NDArray[plt.Axes]] = None,
        setup_figure_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """
        Create a pairplot using Matplotlib.

        Parameters
        ----------
        use_transformed : bool, optional
            If True, use the transformed independent variables.
            The default is False.
        plot_infeasible : bool, default=True
            If True, plot infeasible points.
        color_feas : str, default='blue'
            Color for feasible points.
        color_infeas : str, default='red'
            Color for infeasible points.
        *args : Any
            Additional positional arguments passed to `plot_pairwise`.
        ax : np.ndarray[plt.Axes] | None, default=None
            Optional array of Matplotlib Axes.
            If not provided, a new figure is created.
        setup_figure_kwargs : dict | None, default=None
            Additional options to setup the figure.
        **kwargs : Any
            Additional keyword arguments passed to `plot_pairwise`.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and axes objects.
        """
        feasible = self.feasible
        infeasible = self.infeasible
        x_feas = feasible.x
        x_infeas = infeasible.x

        if use_transformed:
            x_feas = feasible.x_transformed
            x_infeas = infeasible.x_transformed
            labels = self.independent_variable_names
        else:
            x_feas = feasible.x
            x_infeas = infeasible.x
            labels = self.variable_names

        fig, ax = plot_pairwise(
            x_feas,
            labels,
            color=color_feas,
            *args,
            ax=ax,
            tight_layout=False,
            setup_figure_kwargs=setup_figure_kwargs,
            **kwargs,
        )

        if plot_infeasible and len(infeasible) > 0:
            fig, ax = plot_pairwise(
                x_infeas,
                labels,
                color=x_infeas,
                *args,
                ax=ax,
                tight_layout=False,
                **{"update_layout": False, **kwargs}
            )

        return fig, ax

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
        elif isinstance(other, (np.ndarray, list)):
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


def _determine_scaling(
    population: npt.ArrayLike,
    threshold: float = 100.0
) -> list[bool]:
    """
    Determine whether to use log scaling for each variable in a population.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    threshold : float, default=100.0
        Threshold for the data range to trigger log scaling.

    Returns
    -------
    list[bool]
        List of flags indicating whether to use log scaling for each variable.
    """
    n_variables = population.shape[1]
    scaling = []
    for i in range(n_variables):
        min_i, max_i = population[:, i].min(), population[:, i].max()
        scaling.append(min_i > 0 and (max_i / min_i) > threshold)
    return scaling


def _setup_pairwise_axes(
    population: npt.ArrayLike,
    variable_names: list[str] | None,
    autoscale: bool = True,
    update_layout: bool = True,
    ax: npt.NDArray[plt.Axes] | None = None,
    setup_figure_kwargs: dict | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """
    Set up a figure and axes for pairwise plots.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array-like structure with shape (n_samples, n_variables).
    variable_names : list[str], optional
        List of variable names. If None, default names are assigned.
    autoscale : bool, default=True
        If True, automatically determine log scaling for each variable.
    update_layout : bool, default=True
        If True, update layout (labels, ticks, etc.).
    ax : npt.NDArray[plt.Axes] | None, default=None
        Optional array of Matplotlib axes.
    setup_figure_kwargs : dict | None, default=None
        Additional figure setup options.

    Returns
    -------
    tuple
        A tuple containing:
        - plt.Figure: The Matplotlib Figure object.
        - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplot grid.
        - list[bool] : A list of flags indicating whether to use log scaling for
          each variable.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    n_variables = population.shape[1]

    # Determine scaling
    scaling = _determine_scaling(population) if autoscale else [False] * n_variables

    # Create or reuse axes
    if ax is None:
        fig, axs = plotting.setup_figure(
            nrows=n_variables,
            ncols=n_variables,
            sharex="col",
            sharey="row",
            squeeze=False,
            **{"aspect": 1.0, **(setup_figure_kwargs or {})},
        )
    else:
        axs = ax
        fig = axs[0, 0].get_figure()

    if axs.shape != (n_variables, n_variables):
        raise ValueError(
            "Inconsistent shape for provided axs. "
            f"Expected {(n_variables, n_variables)}, got {axs.shape}."
        )

    if update_layout:
        _update_layout(axs, population, variable_names, scaling)

    return fig, axs, scaling


def _update_layout(
    axs: npt.NDArray[plt.Axes],
    population: npt.ArrayLike,
    variable_names: list[str] | None,
    scaling: list[bool],
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """
    Set up a figure and axes for pairwise plots.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes] | None, default=None
        Array of Matplotlib axes.
    population : npt.ArrayLike
        2D array-like structure with shape (n_samples, n_variables).
    variable_names : list[str] | None
        List of variable names. If None, default names are assigned.
    scaling: list[bool]
        List of flags indicating whether to use log scaling for each variable.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    n_variables = population.shape[1]
    variable_names = variable_names or [f"$x_{{{i}}}$" for i in range(n_variables)]

    # Rows i
    for i in range(n_variables):
        scale_i = scaling[i]
        # Columns j
        for j in range(n_variables):
            scale_j = scaling[j]
            ax_ij = axs[i, j]

            # Apply log scale if needed
            if scale_j:
                if ax_ij.get_xscale() != "log":
                    ax_ij.set_xscale("log")
            else:
                ax_ij.ticklabel_format(axis="x", useMathText=True, scilimits=(-3, 3))

            if scale_i:
                if ax_ij.get_yscale() != "log":
                    ax_ij.set_yscale("log")
            else:
                ax_ij.ticklabel_format(axis="y", useMathText=True, scilimits=(-3, 3))

            # Ticks should only be visible on the first column ...
            if j == 0:
                ax_ij.yaxis.set_tick_params(labelleft=True)
            else:
                ax_ij.yaxis.set_tick_params(labelleft=False)
            # ... and last row
            if i == n_variables - 1:
                ax_ij.xaxis.set_tick_params(labelbottom=True)
            else:
                ax_ij.xaxis.set_tick_params(labelbottom=False)

            # Set axis labels on the edges
            if i == n_variables - 1:
                ax_ij.set_xlabel(variable_names[j])
            if j == 0:
                ax_ij.set_ylabel(variable_names[i])


def _plot_pairwise_histogram(
    axs: npt.NDArray[plt.Axes],
    data: npt.ArrayLike,
    color: str,
    n_bins: int = 20,
) -> None:
    """
    Plot histograms on the diagonal of a pairwise plot.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes]
        2D array of Matplotlib axes.
    data : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    color : str
        Color for the histograms.
    n_bins : int, default=20
        Number of bins for the histograms.
    """
    n_variables = axs.shape[0]

    for i in range(n_variables):
        ax = axs[i, i]

        x = data[:, i][np.isfinite(data[:, i])]

        if not hasattr(ax, "_pairwise_bins"):
            ax_hist = ax.twinx()
            ax_hist.set_yticks([])

            lo, hi = x.min(), x.max()

            if ax.get_xscale() == "log":
                if lo <= 0:
                    raise ValueError("Log-scaled histogram requires positive data.")
                bins = np.geomspace(lo, hi, n_bins + 1)
            else:
                bins = np.linspace(lo, hi, n_bins + 1)

            ax._pairwise_bins = bins
            ax._pairwise_hist_ax = ax_hist
        else:
            bins = ax._pairwise_bins
            ax_hist = ax._pairwise_hist_ax

        ax_hist.hist(
            x,
            bins=bins,
            alpha=0.7,
            color=color,
            edgecolor="black",
            align="mid",
        )


def _plot_pairwise_scatter(
    axs: npt.NDArray[plt.Axes],
    data: npt.ArrayLike,
    color: str,
) -> None:
    """
    Plot scatter plots for non-diagonal elements of a pairwise plot.

    Parameters
    ----------
    axs : npt.NDArray[plt.Axes]
        2D array of Matplotlib axes.
    data : npt.ArrayLike
        2D array with shape (n_samples, n_variables).
    color : str
        Color for the scatter points.
    """
    n_variables = axs.shape[0]
    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:
                continue  # Skip diagonal

            ax = axs[i, j]
            ax.scatter(
                data[:, j], data[:, i],
                alpha=0.5, color=color
            )


@plotting.figure_utils
def plot_pairwise(
    population: npt.ArrayLike,
    variable_names: list[str] | None = None,
    color: str = "blue",
    n_bins: int = 20,
    autoscale: bool = True,
    plot_scatter: bool = True,
    plot_histogram: bool = True,
    update_layout: bool = True,
    ax: npt.NDArray[plt.Axes] | None = None,
    setup_figure_kwargs: dict | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Create a pairwise scatter plot for all variables of a population.

    Parameters
    ----------
    population : npt.ArrayLike
        2D array-like structure containing numerical variables with shape
        (n_samples, n_variables)
    variable_names : list of str, optional
        list of variable names corresponding to columns in the data.
        If None, default names will be assigned.
    color : str
        Color for markers. Default is "tab10".
    n_bins : int, default=20
        Number of bins for histogram plots.
    autoscale : bool, default=True
        If True, automatically adjust the scaling of the axes.
    plot_scatter : bool, optional, default=True
        If True, add scatter plots.
    plot_histogram : bool, optional, default=True
        If True, add histogram plots.
    update_layout : bool, optional, default=True
        If True, update layout.
    ax : np.ndarray[plt.Axes] | None, default=None
        Optional array of Matplotlib axs.
        If not provided, a new figure is created.
    setup_figure_kwargs : dict | None, default=None
        Additional options to setup the figure.

    Returns
    -------
    tuple
        A tuple containing:
        - plt.Figure: The Matplotlib Figure object.
        - npt.NDArray[plt.Axes]: An array of Axes objects representing the subplot grid.

    Raises
    ------
    ValueError
        If data does not contain 2D data.
        If the provided axes array does not have the correct shape.
    """
    population = np.array(population, ndmin=2)

    if population.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with ndim={population.ndim}")

    fig, ax, scaling = _setup_pairwise_axes(
        population,
        variable_names,
        autoscale,
        update_layout,
        ax,
        setup_figure_kwargs
    )

    # Plot histograms and scatter plots
    if plot_histogram:
        _plot_pairwise_histogram(
            ax,
            population,
            color=color,
        )
    if plot_scatter:
        _plot_pairwise_scatter(
            ax,
            population,
            color=color,
        )
    if update_layout:
        _update_layout(
            ax,
            population,
            variable_names,
            scaling,
        )

    return fig, ax
