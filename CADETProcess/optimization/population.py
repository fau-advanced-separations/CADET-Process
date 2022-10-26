from pathlib import Path

import corner
import numpy as np
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.optimization.individual import Individual


class Population():
    """Collection of Individuals evaluated during Optimization.

    Attributes
    ----------
    individuals : list
        Individuals evaluated during optimization.

    See Also
    --------
    CADETProcess.optimization.Individual
    ParetoFront

    """

    def __init__(self):
        self._individuals = {}

    @property
    def dimensions(self):
        """tuple: Individual dimensions (n_x, n_f, n_g, n_m)"""
        if self.n_individuals == 0:
            return None

        return self.individuals[0].dimensions

    @property
    def variable_names(self):
        return self.individuals[0].variable_names

    @property
    def independent_variable_names(self):
        return self.individuals[0].independent_variable_names

    @property
    def objective_labels(self):
        return self.individuals[0].objective_labels

    @property
    def contraint_labels(self):
        return self.individuals[0].contraint_labels

    @property
    def meta_score_labels(self):
        return self.individuals[0].meta_score_labels

    def add_individual(self, individual, ignore_duplicate=True):
        """Add individual to population.

        Parameters
        ----------
        individual: Individual
            Individual to be added.
        ignore_duplicate : bool, optional
            If False, an Exception is thrown if individual already exists.

        Raises
        ------
        TypeError
            If Individual is not an instance of Individual.
        CADETProcessError
            If Individual does not match dimensions.
            If Individual already exists.
        """
        if not isinstance(individual, Individual):
            raise TypeError("Expected Individual")

        if self.dimensions is not None \
                and individual.dimensions != self.dimensions:
            raise CADETProcessError("Individual does not match dimensions.")

        if individual in self:
            if ignore_duplicate:
                return
            else:
                raise CADETProcessError("Individual already exists.")
        self._individuals[tuple(individual.x)] = individual

    def update(self, other):
        """Update Population with individuals of other Population.

        Parameters
        ----------
        other : Population
            Other population.

        Raises
        ------
        TypeError
            If other is not an instance of Population.
        CADETProcessError
            If dimensions do not match.
        """
        if not isinstance(other, Population):
            raise TypeError("Expected Population")

        if self.dimensions is not None and self.dimensions != other.dimensions:
            raise CADETProcessError("Dimensions do not match")

        self._individuals.update(other._individuals)

    def remove_individual(self, individual):
        """Remove individual from population.

        Parameters
        ----------
        individual: Individual
            Individual to be removed.

        Raises
        ------
        TypeError
            If individual is not an instance of Individual.
        CADETProcessError
            If individual is not in population.
        """
        if not isinstance(individual, Individual):
            raise TypeError("Expected Individual")

        if individual not in self:
            raise CADETProcessError("Individual is not in population.")
        self._individuals.pop(tuple(individual.x))

    def remove_similar(self):
        """Remove similar individuals from population."""
        for ind in self.individuals.copy():
            to_remove = []

            for ind_other in self.individuals.copy():
                if ind is ind_other:
                    continue

                if ind_other.is_similar(ind):
                    to_remove.append(ind_other)

            for i in reversed(to_remove):
                try:
                    self.remove_individual(i)
                except CADETProcessError:
                    pass

    @property
    def individuals(self):
        """list: All individuals."""
        return list(self._individuals.values())

    @property
    def n_individuals(self):
        """int: Number of indivuals."""
        return len(self.individuals)

    @property
    def x(self):
        """np.array: All evaluated points."""
        return np.array([ind.x for ind in self.individuals])

    @property
    def x_untransformed(self):
        """np.array: All evaluated points."""
        return np.array([ind.x_untransformed for ind in self.individuals])

    @property
    def f(self):
        """np.array: All evaluated objective function values."""
        return np.array([ind.f for ind in self.individuals])

    @property
    def f_min(self):
        """np.array: Minimum objective values."""
        return np.min(self.f, axis=0)

    @property
    def f_max(self):
        """np.array: Maximum objective values."""
        return np.max(self.f, axis=0)

    @property
    def g(self):
        """np.array: All evaluated nonlinear constraint function values."""
        if self.dimensions[2] > 0:
            return np.array([ind.g for ind in self.individuals])

    @property
    def g_min(self):
        """np.array: Minimum constraint values."""
        if self.dimensions[2] > 0:
            return np.min(self.g, axis=0)

    @property
    def g_max(self):
        """np.array: Maximum constraint values."""
        if self.dimensions[2] > 0:
            return np.max(self.g, axis=0)

    @property
    def m(self):
        """np.array: All evaluated metas core values."""
        if self.dimensions[3] > 0:
            return np.array([ind.m for ind in self.individuals])

    @property
    def m_min(self):
        """np.array: Minimum meta score values."""
        if self.dimensions[3] > 0:
            return np.min(self.m, axis=0)

    @property
    def m_max(self):
        """np.array: Maximum meta score values."""
        if self.dimensions[3] > 0:
            return np.max(self.m, axis=0)

    def setup_space_figure(self, include_meta=True, plot_individual=False):
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
            figsize=(n*8 + 2, m*8 + 2),
            squeeze=False,
        )
        plt.close(space_fig_all)

        space_figs_ind = []
        space_axs_ind = []
        for i in range(m*n):
            fig, ax = plt.subplots()
            space_figs_ind.append(fig)
            space_axs_ind.append(ax)
            plt.close(fig)

        space_axs_ind = np.array(space_axs_ind).reshape(space_axs_all.shape)

        if plot_individual:
            return space_figs_ind, space_axs_ind
        else:
            return space_fig_all, space_axs_all

    def plot_space(
            self,
            figs=None, axs=None,
            include_meta=True,
            plot_individual=False,
            autoscale=True, color=None,
            show=True,
            plot_directory=None):

        if axs is None:
            figs, axs = self.setup_space_figure(include_meta, plot_individual)

        if not isinstance(figs, list):
            figs = [figs]

        layout = plotting.Layout()
        layout.y_label = '$f~/~-$'

        variables = self.variable_names
        x = self.x_untransformed

        if include_meta and self.m is not None:
            values = np.hstack((self.f, self.m))
            labels = self.objective_labels + self.meta_score_labels
        else:
            values = self.f
            labels = self.objective_labels

        for i_var, var in enumerate(variables):
            x_var = x[:, i_var]

            for i_metric, label in enumerate(labels):
                v_metric = values[:, i_metric]

                ax = axs[i_metric][i_var]

                ax.scatter(x_var, v_metric, color=color)

                points = np.vstack([col.get_offsets() for col in ax.collections])

                x_all = points[:, 0]
                v_all = points[:, 1]

                layout.x_lim = (np.nanmin(x_all), np.nanmax(x_all))
                layout.x_label = var
                if autoscale and np.min(x_all) > 0:
                    if np.max(x_all) / np.min(x_all[x_all > 0]) > 100.0:
                        ax.set_xscale('log')
                        layout.x_label = f"$log_{{10}}$({var})"

                y_min = np.nanmin(v_all)
                y_max = np.nanmax(v_all)
                y_lim = (
                    min(0.9*y_min, y_min - 0.01*(y_max-y_min)),
                    1.1*y_max
                )
                layout.y_label = label
                if autoscale and np.min(v_all) > 0:
                    if np.max(v_all) / np.min(v_all[v_all > 0]) > 100.0:
                        ax.set_yscale('log')
                        layout.y_label = f"$log_{{10}}$({label})"
                        y_lim = (y_min/2, y_max*2)
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
                    fig.savefig(
                        f'{plot_directory / "parameter_space"}_{i}.png'
                    )
            else:
                figs[0].savefig(
                    f'{plot_directory / "parameter_space"}.png'
                )

        return figs, axs

    def setup_pareto(self):
        n = self.dimensions[1]
        plot = Scatter(
            figsize=(6 * n, 5 * n),
            tight_layout=True,
            plot_3d=False
        )
        return plot

    def plot_pareto(
            self, plot=None, color=None, show=True, plot_directory=None):
        if plot is None:
            plot = self.setup_pareto()
        plot.add(self.f, s=10, color=color)

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            plot.save(f'{plot_directory / "pareto.png"}')

        if not show:
            plt.close(plot.fig)
        else:
            plot.show()

        return plot

    def plot_corner(self, untransformed=True, show=True, plot_directory=None):
        if untransformed:
            x = self.x
            labels = self.independent_variable_names
        else:
            x = self.x_untransformed
            labels = self.variable_names

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

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            fig.savefig(f'{plot_directory / "corner.png"}')

        if not show:
            plt.close(fig)

    def __contains__(self, other):
        if isinstance(other, Individual):
            key = tuple(other.x)
        elif isinstance(other, list):
            key = tuple(other)
        else:
            key = None

        if key in self._individuals:
            return True
        else:
            return False

    def __getitem__(self, x):
        x = tuple(x)
        return self._individuals[x]

    def __len__(self):
        return self.n_individuals

    def __iter__(self):
        return iter(self.individuals)


class ParetoFront(Population):
    def __init__(self, similarity_tol=1e-1, cv_tol=1e-6, *args, **kwargs):
        self.similarity_tol = similarity_tol
        self.cv_tol = cv_tol
        super().__init__(*args, **kwargs)

    def update_individual(self, individual):
        """Update the Pareto front with new individual.

        If any individual in the pareto front is dominated, it is removed.

        Parameters
        ----------
        individual : Individual
            Individual to update the pareto front with.

        Returns
        -------
        significant_improvement : bool
            True if pareto front has improved significantly. False otherwise.
        """
        significant = []

        is_dominated = False
        dominates_one = False
        has_twin = False
        to_remove = []

        try:
            if np.any(np.array(individual.g) > self.cv_tol):
                return False
        except TypeError:
            pass

        for i, ind_pareto in enumerate(self):
            if not dominates_one and ind_pareto.dominates(individual):
                is_dominated = True
                break
            elif individual.dominates(ind_pareto):
                dominates_one = True
                to_remove.append(ind_pareto)
                significant.append(
                    not individual.is_similar(ind_pareto, self.similarity_tol)
                )
            elif individual.is_similar(ind_pareto, self.similarity_tol):
                has_twin = True
                break

        for i in reversed(to_remove):
            self.remove_individual(i)

        if not is_dominated and not has_twin:
            if len(self) == 0:
                significant.append(True)
            elif sum(self.dimensions[1:]) > 1:
                if len(significant) == 0 \
                        or (len(significant) and any(significant)):
                    significant.append(True)

            self.add_individual(individual)

        if len(self) == 0:
            self.add_individual(individual)

        return any(significant)

    def update_population(self, population):
        """Update the Pareto front with new population.

        If any individual in the pareto front is dominated, it is removed.

        Parameters
        ----------
        population : list
            Individuals to update the pareto front with.

        Returns
        -------
        new_members : list
            New members added to the pareto front.
        significant_improvement : bool
            True if pareto front has improved significantly. False otherwise.
        """
        new_members = []
        significant = []

        for ind_new in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []

            try:
                if np.any(np.array(ind_new.g) > self.cv_tol):
                    break
            except TypeError:
                pass

            for i, ind_pareto in enumerate(self):
                if not dominates_one and ind_pareto.dominates(ind_new):
                    is_dominated = True
                    break
                elif ind_new.dominates(ind_pareto):
                    dominates_one = True
                    to_remove.append(ind_pareto)
                    significant.append(not ind_new.is_similar(ind_pareto))
                elif ind_new.is_similar(ind_pareto):
                    has_twin = True
                    break

            for i in reversed(to_remove):
                self.remove_individual(i)

            if not is_dominated and not has_twin:
                if len(self) == 0:
                    significant.append(True)
                elif sum(self.dimensions[1:]) > 1:
                    if len(significant) == 0 \
                            or (len(significant) and any(significant)):
                        significant.append(True)

                self.add_individual(ind_new)
                new_members.append(ind_new)

        if len(self) == 0:
            indices = np.argmin(population.g, axis=0)
            for index in indices:
                ind_new = population.individuals[index]
                self.add_individual(ind_new)

        return new_members, any(significant)

    def remove_infeasible(self):
        """Remove infeasible individuals from pareto front."""
        for ind in self.individuals.copy():
            try:
                if np.any(np.array(ind.g) > self.cv_tol):
                    self.remove_individual(ind)
            except TypeError:
                pass

    def remove_dominated(self):
        """Remove dominated individuals from pareto front."""
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
