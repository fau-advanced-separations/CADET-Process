from typing import Any

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure, UnsignedFloat, UnsignedInteger, Vector

__all__ = ["Fraction", "FractionPool"]


class Fraction(Structure):
    """
    A class representing a fraction of a mixture.

    A fraction is defined by its mass and volume, and can be used to calculate
    properties such as cumulative mass, purity, and concentration.

    Attributes
    ----------
    mass : np.ndarray
        Mass of each component in the fraction.
    volume : float
        Volume of the fraction.

    Properties
    ----------
    n_comp : int
        Number of components in the fraction.
    fraction_mass : np.ndarray
        Cumulative mass of all species in the fraction.
    purity : np.ndarray
        Purity of the fraction, with invalid values set to zero.
    concentration : np.ndarray
        Component concentrations of the fraction, with invalid values set to zero.

    See Also
    --------
    CADETProcess.fractionation.FractionPool
    CADETProcess.fractionation.Fractionator
    """

    mass = Vector()
    volume = UnsignedFloat()
    start = UnsignedFloat()
    end = UnsignedFloat()

    _parameters = ["mass", "volume", "start", "end"]

    @property
    def n_comp(self) -> int:
        """int: Number of components in the fraction."""
        return self.mass.size

    @property
    def fraction_mass(self) -> np.ndarray:
        """
        np.ndarray: Cumulative mass all species in the fraction.

        See Also
        --------
        mass
        purity
        concentration
        """
        return sum(self.mass)

    @property
    def purity(self) -> np.ndarray:
        """
        np.ndarray: Purity of the fraction.

        Invalid values are replaced by zero.

        See Also
        --------
        mass
        fraction_mass
        concentration
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            purity = self.mass / self.fraction_mass

        return np.nan_to_num(purity)

    @property
    def concentration(self) -> np.ndarray:
        """
        np.ndarray: Component concentrations of the fraction.

        Invalid values are replaced by zero.

        See Also
        --------
        mass
        volume
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            concentration = self.mass / self.volume

        return np.nan_to_num(concentration)

    def __repr__(self) -> str:
        """str: String representation of the fraction."""
        return f"{self.__class__.__name__}(mass={self.mass},volume={self.volume})"


class FractionPool(Structure):
    """
    Collection of pooled fractions.

    This class manages multiple fractions of a mixture, facilitating the
    calculation of cumulative properties of the pool, such as total volume,
    total mass, average purity, and average concentration.

    Attributes
    ----------
    n_comp : int
        The number of components each fraction in the pool should have.

    Properties
    ----------
    fractions : list
        List of fractions in the pool.
    n_fractions : int
        Number of fractions in the pool.
    volume : float
        Total volume of all fractions in the pool.
    mass : np.ndarray
        Cumulative mass of each component in the pool.
    pool_mass : float
        Total mass of all components in the pool.
    purity : np.ndarray
        Overall purity of the pool, with invalid values set to zero.
    concentration : np.ndarray
        Average concentration of the pool, with invalid values set to zero.

    See Also
    --------
    CADETProcess.fractionation.Fraction
    CADETProcess.fractionation.Fractionator
    """

    n_comp = UnsignedInteger()

    _parameters = ["n_comp"]

    def __init__(self, n_comp: int, *args: Any, **kwargs: Any) -> None:
        """
        Initialize a FractionPool instance.

        Parameters
        ----------
        n_comp : int
            The number of components each fraction in the pool should have.
        *args : Optional
            Optional Parameters for Structure class.
        **kwargs : Optional
            Additional Parameters for Structure class.
        """
        self._fractions = []
        self.n_comp = n_comp

        super().__init__(*args, **kwargs)

    def add_fraction(self, fraction: Fraction) -> None:
        """
        Add a fraction to the fraction pool.

        Parameters
        ----------
        fraction : Fraction
            The fraction to be added to the pool.

        Raises
        ------
        CADETProcessError
            If the fraction is not an instance of the Fraction class, or if
            the number of components in the fraction does not match the number
            of components in the pool.
        """
        if not isinstance(fraction, Fraction):
            raise CADETProcessError("Expected Fraction")

        if fraction.n_comp != self.n_comp:
            raise CADETProcessError("Number of components does not match.")

        self._fractions.append(fraction)

    @property
    def fractions(self) -> list[Fraction]:
        """list: List of fractions in the pool."""
        if len(self._fractions) == 0:
            return [Fraction(np.zeros((self.n_comp,)), 0)]
        return self._fractions

    @property
    def n_fractions(self) -> int:
        """int: Number of fractions in the pool."""
        return len(self._fractions)

    @property
    def volume(self) -> float:
        """float: Sum of all fraction volumes in the fraction pool."""
        return sum(frac.volume for frac in self.fractions)

    @property
    def mass(self) -> np.ndarray:
        """np.ndarray: Cumulative component mass in the fraction pool."""
        return np.sum([frac.mass for frac in self.fractions], axis=0)

    @property
    def pool_mass(self) -> float:
        """float: Sum of cumulative component mass in the fraction pool."""
        return sum(frac.fraction_mass for frac in self.fractions)

    @property
    def purity(self) -> np.ndarray:
        """
        Total purity of components in the fraction pool.

        Invalid values are replaced by zero.

        Returns
        -------
        purity : np.ndarray
            Purity of each component in the fraction pool.

        See Also
        --------
        mass
        pool_mass
        concentration
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            purity = self.mass / self.pool_mass

        return np.nan_to_num(purity)

    @property
    def concentration(self) -> np.ndarray:
        """
        Total concentration of components in the fraction pool.

        Invalid values are replaced by zero.

        Returns
        -------
        concentration : np.ndarray
            Average concentration of the fraction pool.

        See Also
        --------
        mass
        volume
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            concentration = self.mass / self.volume

        return np.nan_to_num(concentration)

    def __repr__(self) -> str:
        """str: String representation of the fraction pool."""
        return f"{self.__class__.__name__}(n_comp={self.n_comp})"
