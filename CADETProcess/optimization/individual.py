import hashlib

import numpy as np

from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Vector


def hash_array(array):
    """Compute a hash value for an array of floats using the sha256 hash function.

    Parameters
    ----------
    array : numpy.ndarray
        An array of floats.

    Returns
    -------
    str
        A hash value as a string of hexadecimal characters.

    Examples
    --------
    >>> import numpy as np
    >>> hash_array(np.array([1, 2.0]))
    '3dfc9d56e04dcd01590f48b1b57c9ed9fecb1e94e11d3c3f13cf0fd97b7a9f0f'
    """
    array = np.asarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


class Individual(metaclass=StructMeta):
    """Set of variables evaluated during Optimization.

    Attributes
    ----------
    x : list
        Variable values.
    f : list
        Objective values.
    g : list
        Nonlinear constraint values.
    m : list
        Meta score values.

    See Also
    --------
    CADETProcess.optimization.Population
    """
    x = Vector()
    f = Vector()
    g = Vector()
    m = Vector()

    def __init__(
            self,
            x,
            f=None,
            g=None,
            m=None,
            x_untransformed=None,
            independent_variable_names=None,
            objective_labels=None,
            contraint_labels=None,
            meta_score_labels=None,
            variable_names=None):
        self.x = x
        self.f = f
        self.g = g
        self.m = m

        if x_untransformed is None:
            x_untransformed = x
            variable_names = independent_variable_names

        self.x_untransformed = x_untransformed

        self.variable_names = variable_names
        self.independent_variable_names = independent_variable_names
        self.objective_labels = objective_labels
        self.contraint_labels = contraint_labels
        self.meta_score_labels = meta_score_labels

        self.id = hash_array(self.x)

    @property
    def is_evaluated(self):
        """bool: Return True if individual has been evaluated. False otherwise."""
        if self.f is None:
            return False
        else:
            return True

    @property
    def is_feasible(self):
        """bool: Return False if any constraint is not met. True otherwise."""
        if self.g is not None and np.any(np.array(self.g) > 0):
            return False
        else:
            return True

    @property
    def dimensions(self):
        """tuple: Individual dimensions (n_x, n_f, n_g)"""
        if self.g is None:
            g = 0
        else:
            g = len(self.g)
        if self.m is None:
            m = 0
        else:
            m = len(self.m)

        return (len(self.x), len(self.f), g, m)

    def dominates(self, other):
        """Determine if individual dominates other.

        Parameters
        ----------
        other : Individual
            Other individual

        Returns
        -------
        dominates : bool
            True if objectives of "self" are not strictly worse than the
            corresponding objectives of "other" and at least one objective is
            strictly better. False otherwise
        """
        if self.is_feasible and not other.is_feasible:
            return True

        if not self.is_feasible and not other.is_feasible:
            if np.any(self.g < other.g):
                return True

        if self.m is not None:
            self_values = self.m
            other_values = other.m
        else:
            self_values = self.f
            other_values = other.f

        if np.any(self_values > other_values):
            return False

        if np.any(self_values < other_values):
            return True

        return False

    def is_similar(self, other, tol=1e-1):
        """Determine if individual is similar to other.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if individuals are close to each other. False otherwise
        """
        similar_x = self.is_similar_x(other, tol)
        similar_f = self.is_similar_f(other, tol)

        if self.g is not None:
            similar_g = self.is_similar_g(other, tol)
        else:
            similar_g = True

        if self.m is not None:
            similar_m = self.is_similar_m(other, tol)
        else:
            similar_m = True

        return similar_x and similar_f and similar_g and similar_m

    def is_similar_x(self, other, tol=1e-1):
        """Determine if individual is similar to other based on parameter values.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if parameters are close to each other. False otherwise
        """
        similar_x = np.allclose(self.x, other.x, rtol=tol)

        return similar_x

    def is_similar_f(self, other, tol=1e-1):
        """Determine if individual is similar to other based on objective values.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if parameters are close to each other. False otherwise
        """
        similar_f = np.allclose(self.f, other.f, rtol=tol)

        return similar_f

    def is_similar_g(self, other, tol=1e-1):
        """Determine if individual is similar to other based on constraint values.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if parameters are close to each other. False otherwise
        """
        similar_g = np.allclose(self.g, other.g, rtol=tol)

        return similar_g

    def is_similar_m(self, other, tol=1e-1):
        """Determine if individual is similar to other based on meta score values.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if parameters are close to each other. False otherwise
        """
        similar_m = np.allclose(self.m, other.m, rtol=tol)

        return similar_m

    def __str__(self):
        return str(list(self.x))

    def __repr__(self):
        if self.g is None:
            return f'{self.__class__.__name__}({self.x}, {self.f})'
        else:
            return f'{self.__class__.__name__}({self.x}, {self.f}, {self.g})'

    def to_dict(self):
        """Converts individual to a dictionary.

        Returns
        -------
        dict: A dictionary representation of the individual's attributes.
        """
        return {
            "x": self.x,
            "f": self.f,
            "g": self.g,
            "m": self.m,
            "x_untransformed": self.x_untransformed,
            "variable_names": self.variable_names,
            "independent_variable_names": self.independent_variable_names,
            "objective_labels": self.objective_labels,
            "contraint_labels": self.contraint_labels,
            "meta_score_labels": self.meta_score_labels,
        }

    @classmethod
    def from_dict(cls, data):
        """Create Individual from dictionary representation of its attributes.

        Parameters
        ----------
        data : dict
            A dictionary representation of the individual's attributes.

        Returns
        -------
        individual
            Individual idual created from the dictionary.
        """

        return cls(**data)
