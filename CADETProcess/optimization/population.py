from pathlib import Path
import uuid

from addict import Dict
import corner
import numpy as np
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

from CADETProcess import CADETProcessError
from CADETProcess import plotting
from CADETProcess.optimization.individual import hash_array, Individual


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

    def __init__(self, id=None):
        self._individuals = {}
        if id is None:
            self.id = uuid.uuid4()
        else:
            if isinstance(id, bytes):
                id = id.decode(encoding='utf=8')
            self.id = uuid.UUID(id)

    @property
    def feasible(self):
        """Population: Population containing only feasible individuals."""
        pop = Population()
        pop._individuals = {ind.id: ind for ind in self.individuals if ind.is_feasible}

        return pop

    @property
    def infeasible(self):
        """Population: Population containing only infeasible individuals."""
        pop = Population()
        pop._individuals = {ind.id: ind for ind in self.individuals if not ind.is_feasible}

        return pop

    @property
    def n_x(self):
        return self.individuals[0].n_x

    @property
    def n_f(self):
        return self.individuals[0].n_f

    @property
    def n_g(self):
        return self.individuals[0].n_g

    @property
    def n_m(self):
        return self.individuals[0].n_m

    @property
    def dimensions(self):
        """tuple: Individual dimensions (n_x, n_f, n_g, n_m)"""
        if self.n_individuals == 0:
            return None

        return self.individuals[0].dimensions

    @property
    def variable_names(self):
        if self.individuals[0].variable_names is None:
            return [f'x_{i}' for i in range(self.n_x)]
        else:
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

        self._individuals[individual.id] = individual

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
        self._individuals.pop(individual.id)

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

    def remove_similar(self):
        """Remove similar individuals from population."""
        for ind in self.individuals.copy():
            to_remove = []

            for ind_other in self.individuals.copy():
                if ind is ind_other:
                    continue

                if ind_other.is_similar(ind, self.similarity_tol):
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
    def f_avg(self):
        """np.array: Average objective values."""
        return np.mean(self.f, axis=0)

    @property
    def g(self):
        """np.array: All evaluated nonlinear constraint function values."""
        if self.dimensions[2] > 0:
            return np.array([ind.g for ind in self.individuals])

    @property
    def g_min(self):
        """np.array: Minimum nonlinear constraint values."""
        if self.dimensions[2] > 0:
            return np.min(self.g, axis=0)

    @property
    def g_max(self):
        """np.array: Maximum nonlinear constraint values."""
        if self.dimensions[2] > 0:
            return np.max(self.g, axis=0)

    @property
    def g_avg(self):
        """np.array: Average nonlinear constraint values."""
        return np.mean(self.g, axis=0)

    @property
    def cv(self):
        """np.array: All evaluated nonlinear constraint function values."""
        if self.dimensions[2] > 0:
            return np.array([ind.cv for ind in self.individuals])

    @property
    def cv_min(self):
        """np.array: Minimum nonlinear constraint violation values."""
        if self.dimensions[2] > 0:
            return np.min(self.cv, axis=0)

    @property
    def cv_max(self):
        """np.array: Maximum nonlinearconstraint violation values."""
        if self.dimensions[2] > 0:
            return np.max(self.cv, axis=0)

    @property
    def cv_avg(self):
        """np.array: Average nonlinear constraint violation values."""
        return np.mean(self.cv, axis=0)

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

    @property
    def cv_avg(self):
        """np.array: Average meta score values."""
        return np.mean(self.m, axis=0)

    def setup_objectives_figure(self, include_meta=True, plot_individual=False):
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

    def plot_objectives(
            self,
            figs=None, axs=None,
            include_meta=True,
            plot_individual=False,
            autoscale=True,
            color_feas='blue',
            color_infeas='red',
            show=True,
            plot_directory=None):
        """Plot the objective function values for each design variable.

        Parameters
        ----------
        figs : plt.Figure or list of plt.Figure, optional
            Figure(s) to plot the objectives on.
        axs : plt.Axes or list of plt.Axes, optional
            Axes to plot the objectives on.
            If None, new figures and axes will be created.
        include_meta : bool, optional
            If True, meta scores will be included in the plot. The default is True.
        plot_individual : bool, optional
            If True, create separate figures for each objective. Otherwise, all
            objectives are plotted in one figure.
            The default is False.
        autoscale : bool, optional
            If True, automatically adjust the scaling of the axes. The default is True.
        color_feas : str, optional
            The color for the feasible points. The default is 'blue'.
        color_infeas : str, optional
            The color for the infeasible points. The default is 'red'.
        show : bool, optional
            If True, display the plot. The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved.
            The default is None.

        Returns
        -------
        tuple
            Tuple with (lists of) figure and axes objects.

        """
        if axs is None:
            figs, axs = self.setup_objectives_figure(include_meta, plot_individual)

        if not isinstance(figs, list):
            figs = [figs]

        layout = plotting.Layout()
        layout.y_label = '$f~/~-$'

        variables = self.variable_names
        feasible = self.feasible
        infeasible = self.infeasible
        x_feas = feasible.x
        x_infeas = infeasible.x

        if include_meta and self.m is not None:
            values_feas = np.hstack((feasible.f, feasible.m))
            values_infeas = np.hstack((infeasible.f, infeasible.m))
            labels = self.objective_labels + self.meta_score_labels
        else:
            values_feas = feasible.f
            values_infeas = infeasible.f
            labels = self.objective_labels

        for i_var, var in enumerate(variables):
            x_var_feas = x_feas[:, i_var]
            if len(x_infeas) > 0:
                x_var_infeas = x_infeas[:, i_var]

            for i_metric, label in enumerate(labels):
                ax = axs[i_metric][i_var]

                if len(x_infeas) > 0:
                    v_metric_infeas = values_infeas[:, i_metric]
                    ax.scatter(x_var_infeas, v_metric_infeas, alpha=0.5, color=color_infeas)

                v_metric_feas = values_feas[:, i_metric]
                ax.scatter(x_var_feas, v_metric_feas, alpha=0.5, color=color_feas)

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
                        f'{plot_directory / "objectives"}_{i}.png'
                    )
            else:
                figs[0].savefig(
                    f'{plot_directory / "objectives"}.png'
                )

        return figs, axs

    def setup_pareto(self):
        n = self.dimensions[1]
        plot = Scatter(
            figsize=(6 * n, 5 * n),
            tight_layout=True,
            plot_3d=False,
            labels=self.objective_labels,
        )
        return plot

    def plot_pareto(
            self, plot=None, color=None, show=True, plot_directory=None):
        """Plot pairwise Pareto fronts of for each generation in the optimization.

        The Pareto front represents the optimal solutions that cannot be improved in one
        objective without sacrificing another.
        The method shows a pairwise Pareto plot, where each objective is plotted against
        every other objective in a scatter plot, allowing for a visualization of the
        trade-offs between the objectives.
        To highlight the progress, a colormap is used where later generations are
        plotted with darker blueish colors.

        Parameters
        ----------
        plot : pymoo.visualization.scatter.Scatter, optional
            Base figure. If None is provided, a new one will be setup.
        color: str
            Color for scatter points.
        show : bool, optional
            If True, display the plot.
            The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved.
            The default is None.

        See Also
        --------
        setup_pareto
        CADETProcess.optimization.OptimizationResults.plot_pareto
        pymoo.visualization.scatter.Scatter

        """
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
        """Create a corner plot of the independent variables.

        Parameters
        ----------
        untransformed : bool, optional
            If True, use the untransformed independent variables.
            The default is True.
        show : bool, optional
            If True, display the plot.
            The default is True.
        plot_directory : str, optional
            The directory where the plot should be saved.
            The default is None.

        See Also
        --------
        CADETProcess.results.plot_corner
        corner.corner
        """
        if untransformed:
            x = self.x_untransformed
            labels = self.variable_names
        else:
            x = self.x
            labels = self.independent_variable_names

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
            key = other.id
        elif isinstance(other, (np.array, list)):
            key = hash_array(other)
        else:
            key = None

        if key in self._individuals:
            return True
        else:
            return False

    def __getitem__(self, x):
        key = hash_array(x)

        return self._individuals[key]

    def __len__(self):
        return self.n_individuals

    def __iter__(self):
        return iter(self.individuals)

    def to_dict(self):
        """Convert Population to a dictionary.

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
    def from_dict(cls, data):
        """Create Population from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing population data.

        Returns
        -------
        Population
            Population created from data.
        """
        id = data['id']
        if isinstance(id, bytes):
            id = id.decode(encoding='utf=8')
        population = cls(id)
        for individual_data in data['individuals'].values():
            individual = Individual.from_dict(individual_data)
            population.add_individual(individual)
        return population


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

        if self.similarity_tol != 0:
            self.remove_similar()

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
                # Do not add if invalid
                if np.any(np.array(ind_new.cv) > self.cv_tol):
                    break
            except TypeError:
                pass

            for i, ind_pareto in enumerate(self):
                # Do not add if is dominated
                if not dominates_one and ind_pareto.dominates(ind_new):
                    is_dominated = True
                    break
                elif ind_new.dominates(ind_pareto):
                    # Remove existing if new dominates
                    dominates_one = True
                    to_remove.append(ind_pareto)
                    if not ind_new.is_similar(ind_pareto, self.similarity_tol):
                        significant.append(True)
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
            indices = np.argmin(population.cv, axis=0)
            for index in indices:
                ind_new = population.individuals[index]
                self.add_individual(ind_new)

        if self.similarity_tol is not None:
            self.remove_similar()

        return new_members, any(significant)

    def remove_infeasible(self):
        """Remove infeasible individuals from pareto front."""
        for ind in self.individuals.copy():
            try:
                if np.any(np.array(ind.cv) > self.cv_tol):
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

    def to_dict(self):
        """Convert ParetoFront to a dictionary.

        Returns
        -------
        dict
            ParetoFront as a dictionary with individuals stored as list of dictionaries.
        """
        front = super().to_dict()
        if self.similarity_tol is not None:
            front['similarity_tol'] = self.similarity_tol
        front['cv_tol'] = self.cv_tol

        return front

    @classmethod
    def from_dict(cls, data):
        """Create ParetoFront from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing population data.

        Returns
        -------
        ParetoFront
            ParetoFront created from data.
        """
        front = cls(data['similarity_tol'], data['cv_tol'], data['id'])
        for individual_data in data['individuals'].values():
            individual = Individual.from_dict(individual_data)
            front.add_individual(individual)

        return front
