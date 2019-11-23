import numpy as np

from CADETProcess import CADETProcessError

class Fraction:
    def __init__(self, mass, volume):
        self.mass = mass
        self.volume = volume

    @property
    def n_comp(self):
        return self.mass.size

    @property
    def fraction_mass(self):
        """Returns the sum of all collected species in the fraction.

        Returns
        -------
        fraction_mass : float
            Cumulative mass of all species in the fraction.

        See also
        --------
        mass
        purity
        concentration
        """
        return sum(self.mass)

    @property
    def purity(self):
        """Returns the purity of the fraction.

        Invalid values are replaced by zero.

        Returns
        -------
        purity : ndarray
            Purity of the fraction.

        See also
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
        """Returns the concentration of the fraction.

        Invalid values are replaced by zero.

        Returns
        -------
        concentration : ndarray
            Concentration of the fraction.

        See also
        --------
        mass
        volume
        """
        return self.mass / self.volume

    def __repr__(self):
       return "%s(mass=np.%r,volume=%r)" % (self.__class__.__name__,
                 self.mass, self.volume)


class FractionPool:
    """
    """
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self._fractions = []


    def add_fraction(self, fraction):
        if not isinstance(fraction, Fraction):
            raise CADETProcessError('Expected Fraction')

        if fraction.n_comp != self.n_comp:
            raise CADETProcessError('Number of components not matching')

        self._fractions.append(fraction)

    @property
    def fractions(self):
        if len(self._fractions) == 0:
            return [Fraction(np.zeros((self.n_comp,)),0)]
        return self._fractions

    @property
    def volume(self):
        """Returns the sum of all fraction volumes in the fraction pool

        Returns
        -------
        volume : float
            Cumulative volume of all fractions in the pool.
        """
        return sum(frac.volume for frac in self.fractions)

    @property
    def mass(self):
        """Returns the cumulative sum of the fraction masses of the pool.

        Returns
        -------
        mass : float
            Cumulative mass of all fractions in the pool.
        """
        return sum(frac.mass for frac in self.fractions)

    @property
    def pool_mass(self):
        """Returns the sum of all component masses of all fractions of the pool.

        Returns
        -------
        pool_mass : float
            Cumulative mass of all fractions in the pool.
        """
        return sum(frac.fraction_mass for frac in self.fractions)

    @property
    def purity(self):
        """Returns the average purity of the fraction pool.

        Invalid values are replaced by zero.

        Returns
        -------
        purity : ndarray
            Average purity of the fraction.

        See also
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
        """Returns the average concentration of the fraction pool.

        Invalid values are replaced by zero.

        Returns
        -------
        concentration : ndarray
            Average concentration of the fraction pool.

        See also
        --------
        mass
        volume
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration = self.mass / self.volume

        return np.nan_to_num(concentration)

    def __repr__(self):
       return "%s(n_comp=%r)" % (self.__class__.__name__, self.n_comp)
