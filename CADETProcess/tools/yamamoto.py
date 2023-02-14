from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from CADETProcess import CADETProcessError
from CADETProcess.processModel import TubularReactorBase, StericMassAction


__all__ = ['GradientExperiment', 'plot_experiments', 'YamamotoResults', 'fit_parameters']


class GradientExperiment():
    def __init__(
            self, time, c_salt, c_protein, gradient_volume,
            c_salt_start=None, c_salt_end=None):
        self.time = time
        self.c_salt = c_salt
        self.c_protein = c_protein
        self.gradient_volume = gradient_volume

        if c_salt_start is None:
            c_salt_start = np.min(c_salt)
        self.c_salt_start = c_salt_start

        if c_salt_end is None:
            c_salt_end = np.max(c_salt)
        self.c_salt_end = c_salt_end

    @property
    def n_proteins(self):
        return self.c_protein.shape[1]

    @property
    def c_salt_at_max(self):
        """np.array: Salt concentration at protein peak maximum in mM."""
        max_c_protein = np.argmax(self.c_protein, axis=0)

        return self.c_salt[max_c_protein]

    @property
    def t_at_max(self):
        max_c_protein = np.argmax(self.c_protein, axis=0)

        return self.time[max_c_protein]

    def calculate_normalized_gradient_slope(self, column_volume, total_porosity):
        """Calculate normalized concentration gradient slope.

        Parameters
        ----------
        column_volume : float
            Void volume of column in m³.
        total_porosity : float
            Total porosity of column.

        Returns
        -------
        normalized_slope : float
            Gradient slope in mM.

        """
        slope = (self.c_salt_end - self.c_salt_start) / self.gradient_volume
        vol_factor = column_volume - total_porosity*column_volume
        return slope * vol_factor

    def plot(self, fig=None, ax=None, sec_ax=None):
        if ax is None:
            fig, ax = plt.subplots()

            ax.set_xlabel('$Time / s$')
            ax.set_ylabel('$c_{Protein} / mM$')

            sec_ax = ax.twinx()
            sec_ax.set_ylabel('$c_{Salt} / mM$')

        ax.plot(self.time, self.c_protein, label='Protein')

        c_p_max = np.max(self.c_protein, axis=0)
        ax.vlines(self.t_at_max, ymin=0, ymax=c_p_max, color='k', linestyle='--')
        sec_ax.plot(self.time, self.c_salt, 'k', label='Salt')
        sec_ax.plot(self.t_at_max, self.c_salt_at_max, 'ro')

        fig.tight_layout()

        return fig, ax, sec_ax


def plot_experiments(experiments):
    fig = ax = sec_ax = None

    for exp in experiments:
        fig, ax, sec_ax = exp.plot(fig, ax, sec_ax)


def yamamoto_equation(log_c_salt_at_max_M, lambda_, nu, k_eq):
    """

    Parameters
    ----------
    log_c_salt_at_max_M : float
        log10 of salt concentrations in M at protein peak maximum.
    lambda_ : float
        Resin capacity in mM.
    nu : float
        Characteristic charge.
    k_eq : float, optional
        Equilibrium constant.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    lambda_M = lambda_/1000

    return np.multiply((nu + 1), log_c_salt_at_max_M) - np.log10(k_eq * lambda_M**nu * (nu + 1))


class YamamotoResults:
    """Parameter values determined using Yamamoto's method."""

    def __init__(self, column, experiments, log_gradient_slope, log_c_salt_at_max_M):
        self.column = column
        self.experiments = experiments
        self.log_gradient_slope = log_gradient_slope
        self.log_c_salt_at_max_M = log_c_salt_at_max_M

    @property
    def characteristic_charge(self):
        return self.column.binding_model.characteristic_charge[1:]

    @property
    def k_eq(self):
        return self.column.binding_model.adsorption_rate[1:]

    def plot(self, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

            ax.set_ylabel('Normalized Gradient Slope $GH$ / $M$')
            ax.set_xlabel('Peak Salt Concentration $I_R$ / $M$')

        n_proteins = self.experiments[0].n_proteins

        for i_p in range(n_proteins):
            k_eq = self.k_eq[i_p]
            nu = self.characteristic_charge[i_p]

            x = [
                min(self.log_c_salt_at_max_M[:, i_p])*1.05,
                max(self.log_c_salt_at_max_M[:, i_p])*0.95
            ]

            y = yamamoto_equation(
                log_c_salt_at_max_M=x,
                lambda_=self.column.binding_model.capacity,
                nu=nu,
                k_eq=k_eq,
            )
            ax.plot(x, y, 'k')

            ax.plot(self.log_c_salt_at_max_M, self.log_gradient_slope, 'ro')

        fig.tight_layout()

        return fig, ax


def fit_parameters(experiments, column):
    """Fit parameters using Yamamoto's method.

    Parameters
    ----------
    experiments : list
        List of GradientExperiments.
    column : Column
        Column object with binding model

    Returns
    -------
    yamamoto_results : YamamotoResults
        Parameter values.

    """
    if not isinstance(column, TubularReactorBase):
        raise TypeError("Expected Column Model.")

    if not isinstance(column.binding_model, StericMassAction):
        raise TypeError("Yamamoto's method only works for Steric Mass Action model.")

    n_proteins = column.n_comp - 1
    for exp in experiments:
        if exp.n_proteins != n_proteins:
            raise CADETProcessError("Experiments must have same number of proteins.")

    normalized_gradient_slope = [
        exp.calculate_normalized_gradient_slope(column.volume, column.total_porosity)
        for exp in experiments
    ]
    log_gradient_slope = np.log10(np.divide(normalized_gradient_slope, 1000))

    nu = np.zeros((experiments[0].n_proteins,))
    k_eq = np.zeros((experiments[0].n_proteins,))
    log_c_salt_at_max_M = np.zeros((len(experiments), experiments[0].n_proteins,))

    for i_p in range(experiments[0].n_proteins):
        c_salt_at_max = [exp.c_salt_at_max[i_p] for exp in experiments]
        log_c_salt_at_max_M[:, i_p] = np.log10(np.array(c_salt_at_max)/1000)

        nu[i_p], k_eq[i_p] = _fit_yamamoto(
            log_c_salt_at_max_M[:, i_p], log_gradient_slope, column.binding_model.capacity
        )

    column.binding_model.characteristic_charge = [0, *nu.tolist()]
    column.binding_model.adsorption_rate = [0, *k_eq.tolist()]

    yamamoto_results = YamamotoResults(
        column, experiments,
        log_gradient_slope, log_c_salt_at_max_M
        )

    return yamamoto_results


def _fit_yamamoto(log_c_salt_at_max_M, log_gradient_slope, lambda_):
    """

    Parameters
    ----------
    log_c_salt_at_max_M : list
        log10 of salt concentrations at protein peak maximum in mM.
    log_gradient_slope : TYPE
        DESCRIPTION.
    lambda_ : TYPE
        Resin capacity in mM.

    Returns
    -------
    nu : float
        Characteristic charge.
    k_eq : TYPE, optional
        Equilibrium constant.

    """
    bounds = ((0, 1e-10), (1000, 1000))

    def yamamoto_wrapper(c_s, nu, k_eq):
        return yamamoto_equation(c_s, lambda_, nu, k_eq)

    results, pcov = curve_fit(
        yamamoto_wrapper, log_c_salt_at_max_M, log_gradient_slope, bounds=bounds
    )

    return results
