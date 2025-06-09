from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from CADETProcess import CADETProcessError
from CADETProcess.processModel import (
    ChromatographicColumnBase,
    StericMassAction,
    TubularReactorBase,
)

__all__ = [
    "GradientExperiment",
    "plot_experiments",
    "YamamotoResults",
    "fit_parameters",
]


class GradientExperiment:
    """Gradient Experiment Class."""

    def __init__(
        self,
        time: npt.ArrayLike,
        c_salt: npt.ArrayLike,
        c_protein: npt.ArrayLike,
        gradient_volume: float,
        c_salt_start: Optional[float] = None,
        c_salt_end: Optional[float] = None,
    ) -> None:
        """
        Initialize a GradientExperiment instance.

        Parameters
        ----------
        time : ArrayLike
            Time points in seconds.
        c_salt : ArrayLike
            Salt concentration in mM.
        c_protein : ArrayLike
            Protein concentration(s) in mM.
        gradient_volume : float
            Gradient volume in m³.
        c_salt_start : Optional[float], default=None
            Starting salt concentration in mM.
        c_salt_end : Optional[float], default=None
            Ending salt concentration in mM.
        """
        self.time = np.array(time)
        self.c_salt = np.array(c_salt)

        c_protein = np.array(c_protein)
        if len(c_protein.shape) == 1:
            c_protein = np.expand_dims(c_protein, axis=1)
        self.c_protein = c_protein
        self.gradient_volume = gradient_volume

        if c_salt_start is None:
            c_salt_start = np.min(c_salt)
        self.c_salt_start = c_salt_start

        if c_salt_end is None:
            c_salt_end = np.max(c_salt)
        self.c_salt_end = c_salt_end

    @property
    def n_proteins(self) -> int:
        """int: Number of proteins."""
        return self.c_protein.shape[1]

    @property
    def c_salt_at_max(self) -> float:
        """float: Salt concentration at protein peak maximum in mM."""
        max_c_protein = np.argmax(self.c_protein, axis=0)

        return self.c_salt[max_c_protein]

    @property
    def t_at_max(self) -> float:
        """int: Number of proteins."""
        max_c_protein = np.argmax(self.c_protein, axis=0)

        return self.time[max_c_protein]

    def calculate_normalized_gradient_slope(
        self,
        column_volume: float,
        total_porosity: float,
    ) -> float:
        """
        Calculate normalized concentration gradient slope.

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
        vol_factor = column_volume - total_porosity * column_volume
        return slope * vol_factor

    def plot(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        sec_ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Plot the gradient experiment data.

        Parameters
        ----------
        fig : Optional[plt.Figure], default=None
            Existing matplotlib Figure object.
        ax : Optional[plt.Axes], default=None
            Existing matplotlib Axes object for primary plot.
        sec_ax : Optional[plt.Axes], default=None
            Existing secondary Axes for salt concentration.

        Returns
        -------
        fig : plt.Figure
            The figure object containing the plot.
        ax : plt.Axes
            The axes object for the protein concentration.
        sec_ax : plt.Axes
            The secondary axes object for the salt concentration.
        """
        if ax is None:
            fig, ax = plt.subplots()

            ax.set_xlabel("$Time / s$")
            ax.set_ylabel("$c_{Protein} / mM$")

            sec_ax = ax.twinx()
            sec_ax.set_ylabel("$c_{Salt} / mM$")

        ax.plot(self.time, self.c_protein, label="Protein")

        c_p_max = np.max(self.c_protein, axis=0)
        ax.vlines(self.t_at_max, ymin=0, ymax=c_p_max, color="k", linestyle="--")
        sec_ax.plot(self.time, self.c_salt, "k", label="Salt")
        sec_ax.plot(self.t_at_max, self.c_salt_at_max, "ro")

        fig.tight_layout()

        return fig, ax, sec_ax


def plot_experiments(experiments: list[GradientExperiment]) -> None:
    """
    Plot multiple gradient experiments in a single figure.

    Parameters
    ----------
    experiments : list of GradientExperiment
        List of gradient experiment instances to plot.
    """
    fig = ax = sec_ax = None

    for exp in experiments:
        fig, ax, sec_ax = exp.plot(fig, ax, sec_ax)


def yamamoto_equation(
    log_c_salt_at_max_M: float,
    lambda_: float,
    nu: float,
    k_eq: float,
) -> np.ndarray:
    r"""
    Calculate the theoretical normalized gradient slope using Yamamoto's method.

    Yamamoto's method is used in ion-exchange chromatography to model the relationship
    between the normalized gradient slope (GH) and the peak salt concentration (I_R)
    during a linear salt gradient elution. This method is based on the steric mass
    action (SMA) model and allows for the determination of key chromatographic
    parameters such as the characteristic charge (ν) and the equilibrium constant
    (k_eq).

    The underlying equation is:

    .. math::
        \log(GH) = (ν + 1) \log(I_R) - \log(k_{eq} \cdot \lambda^ν \cdot (ν + 1))

    where:
        - GH: Normalized gradient slope (logarithmic scale)
        - I_R: Salt concentration at the protein peak maximum (in M)
        - ν (nu): Characteristic charge of the protein
        - k_eq: Equilibrium constant (binding affinity)
        - λ (lambda_): Resin capacity in mM

    Parameters
    ----------
    log_c_salt_at_max_M : np.ndarray
        Log10 of salt concentrations at protein peak maximum in M.
    lambda_ : float
        Resin capacity in mM.
    nu : float
        Characteristic charge of the molecule.
    k_eq : float
        Equilibrium constant of the binding model.

    Returns
    -------
    np.ndarray
        Calculated normalized gradient slope (GH) values in logarithmic scale.
    """
    lambda_M = lambda_ / 1000

    return np.multiply((nu + 1), log_c_salt_at_max_M) - np.log10(
        k_eq * lambda_M**nu * (nu + 1)
    )


class YamamotoResults:
    """Parameter values determined using Yamamoto's method."""

    def __init__(
        self,
        column: ChromatographicColumnBase,
        experiments: list[GradientExperiment],
        log_gradient_slope: npt.ArrayLike,
        log_c_salt_at_max_M: npt.ArrayLike,
    ) -> None:
        """
        Initialize YamamotoResults with column, experiments, and log-transformed data.

        Parameters
        ----------
        column : Column
            Column object containing the binding model.
        experiments : list of GradientExperiment
            List of gradient experiment instances.
        log_gradient_slope : np.ndarray
            Normalized gradient slopes in logarithmic scale.
        log_c_salt_at_max_M : np.ndarray
            Log10 of salt concentrations at protein peak maximum in M.
        """
        self.column = column
        self.experiments = experiments
        self.log_gradient_slope = log_gradient_slope
        self.log_c_salt_at_max_M = log_c_salt_at_max_M

    @property
    def characteristic_charge(self) -> np.ndarray:
        """np.ndarray: Characteristic charges of the binding model."""
        return np.array(self.column.binding_model.characteristic_charge[1:])

    @property
    def k_eq(self) -> np.ndarray:
        """np.ndarray: Equilibrium constants of the binding model."""
        return np.array(self.column.binding_model.adsorption_rate[1:])

    def plot(
        self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the normalized gradient slope against the peak salt concentration.

        Parameters
        ----------
        fig : Optional[plt.Figure], default=None
            Existing matplotlib Figure object.
        ax : Optional[plt.Axes], default=None
            Existing matplotlib Axes object.

        Returns
        -------
        fig : plt.Figure
            The figure object containing the plot.
        ax : plt.Axes
            The axes object for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_ylabel("Normalized Gradient Slope $GH$ / $M$")
            ax.set_xlabel("Peak Salt Concentration $I_R$ / $M$")

        n_proteins = self.experiments[0].n_proteins

        for i_p in range(n_proteins):
            k_eq = self.k_eq[i_p]
            nu = self.characteristic_charge[i_p]

            x = [
                min(self.log_c_salt_at_max_M[:, i_p]) * 1.05,
                max(self.log_c_salt_at_max_M[:, i_p]) * 0.95,
            ]

            y = yamamoto_equation(
                log_c_salt_at_max_M=x,
                lambda_=self.column.binding_model.capacity,
                nu=nu,
                k_eq=k_eq,
            )
            ax.plot(x, y, "k")
            ax.plot(self.log_c_salt_at_max_M[:, i_p], self.log_gradient_slope, "ro")

        fig.tight_layout()
        return fig, ax


def fit_parameters(
    experiments: list,
    column: ChromatographicColumnBase,
) -> YamamotoResults:
    """
    Fit parameters using Yamamoto's method.

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
    log_c_salt_at_max_M = np.zeros(
        (
            len(experiments),
            experiments[0].n_proteins,
        )
    )

    for i_p in range(experiments[0].n_proteins):
        c_salt_at_max = [exp.c_salt_at_max[i_p] for exp in experiments]
        log_c_salt_at_max_M[:, i_p] = np.log10(np.array(c_salt_at_max) / 1000)

        nu[i_p], k_eq[i_p] = _fit_yamamoto(
            log_c_salt_at_max_M[:, i_p],
            log_gradient_slope,
            column.binding_model.capacity,
        )

    column.binding_model.characteristic_charge = [0, *nu.tolist()]
    column.binding_model.adsorption_rate = [0, *k_eq.tolist()]

    yamamoto_results = YamamotoResults(
        column, experiments, log_gradient_slope, log_c_salt_at_max_M
    )

    return yamamoto_results


def _fit_yamamoto(
    log_c_salt_at_max_M: np.ndarray, log_gradient_slope: np.ndarray, lambda_: float
) -> tuple[float, float]:
    """
    Fit the Yamamoto model to experimental data using non-linear curve fitting.

    Parameters
    ----------
    log_c_salt_at_max_M : np.ndarray
        Log10 of salt concentrations at protein peak maximum in mM.
    log_gradient_slope : np.ndarray
        Normalized gradient slopes in logarithmic scale.
    lambda_ : float
        Resin capacity in mM.

    Returns
    -------
    nu : float
        Fitted characteristic charge.
    k_eq : float
        Fitted equilibrium constant.
    """
    bounds = ((0, 1e-10), (1000, 1000))

    def yamamoto_wrapper(c_s: float, nu: float, k_eq: float) -> Callable:
        return yamamoto_equation(c_s, lambda_, nu, k_eq)

    results, pcov = curve_fit(
        yamamoto_wrapper,
        log_c_salt_at_max_M,
        log_gradient_slope,
        bounds=bounds,
        p0=(1, 1),
    )

    return results
