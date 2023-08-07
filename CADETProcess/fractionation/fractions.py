import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    UnsignedInteger, UnsignedFloat, Vector
)


class Fraction(Structure):
    """A class representing a fraction of a mixture.

    A fraction is defined by its mass and volume, and can be used to calculate
    properties such as cumulative mass, purity, and concentration.

    Attributes
    ----------
    mass : np.ndarray
        The mass of each component in the fraction. The array is 1-dimensional.
    volume : float
        The volume of the fraction.
    """

    mass = Vector()
    volume = UnsignedFloat()

    def __init__(self, mass, volume):
        """Initialize a Fraction instance.

        Parameters
        ----------
        mass : numpy.ndarray
            The mass of each component in the fraction.
            The array should be 1-dimensional.
        volume : float
            The volume of the fraction.
        """
        self.mass = mass
        self.volume = volume

    @property
    def n_comp(self):
        """int: Number of components in the fraction."""
        return self.mass.size

    @property
    def fraction_mass(self):
        """np.ndarray: Cumulative mass all species in the fraction.

        See Also
        --------
        mass
        purity
        concentration
        """
        return sum(self.mass)

    @property
    def purity(self):
        """np.ndarray: Purity of the fraction.

        Invalid values are replaced by zero.

        See Also
        --------
        mass
        fraction_mass
        concentration
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = self.mass / self.fraction_mass

        return np.nan_to_num(purity)

    @property
    def concentration(self):
        """np.ndarray: Component concentrations of the fraction.

        Invalid values are replaced by zero.

        See Also
        --------
        mass
        volume
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration = self.mass / self.volume

        return np.nan_to_num(concentration)

    def __repr__(self):
        return \
            f"{self.__class__.__name__}" \
            f"(mass={self.mass},volume={self.volume})"


class FractionPool(Structure):
    """Collection of pooled fractions.

    This class manages multiple fractions of a mixture, facilitating the
    calculation of cumulative properties of the pool, such as total volume,
    total mass, average purity, and average concentration.

    Attributes
    ----------
    n_comp : int
        The number of components each fraction in the pool should have.

    See Also
    --------
    CADETProcess.fractionation.Fraction
    CADETProcess.fractionation.Fractionator
    """

    n_comp = UnsignedInteger()

    def __init__(self, n_comp):
        """Initialize a FractionPool instance.

        Parameters
        ----------
        n_comp : int
            The number of components each fraction in the pool should have.
        """
        self._fractions = []
        self.n_comp = n_comp

    def add_fraction(self, fraction):
        """Add a fraction to the fraction pool.

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
            raise CADETProcessError('Expected Fraction')

        if fraction.n_comp != self.n_comp:
            raise CADETProcessError('Number of components does not match.')

        self._fractions.append(fraction)

    @property
    def fractions(self):
        """list: List of fractions in the pool."""
        if len(self._fractions) == 0:
            return [Fraction(np.zeros((self.n_comp,)), 0)]
        return self._fractions

    @property
    def n_fractions(self):
        """int: Number of fractions in the pool."""
        return len(self._fractions)

    @property
    def volume(self):
        """float: Sum of all fraction volumes in the fraction pool."""
        return sum(frac.volume for frac in self.fractions)

    @property
    def mass(self):
        """np.ndarray: Cumulative component mass in the fraction pool."""
        return np.sum([frac.mass for frac in self.fractions], axis=0)

    @property
    def pool_mass(self):
        """float: Sum of cumulative component mass in the fraction pool."""
        return sum(frac.fraction_mass for frac in self.fractions)

    @property
    def purity(self):
        """Total purity of components in the fraction pool.

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
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = self.mass / self.pool_mass

        return np.nan_to_num(purity)

    @property
    def concentration(self):
        """Total concentration of components in the fraction pool.

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
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration = self.mass / self.volume

        return np.nan_to_num(concentration)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_comp={self.n_comp})"
