import hashlib
from typing import Optional

import numpy as np
import numpy.typing as npt
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Bool, Float, Structure, Vector


def hash_array(array: np.ndarray) -> str:
    """
    Compute a hash value for an array of floats using the sha256 hash function.

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


class Individual(Structure):
    """
    Set of variables evaluated during Optimization.

    Attributes
    ----------
    id : str
        UUID for individual.
    x : np.ndarray
        Variable values in untransformed space.
    x_transformed : np.ndarray
        Independent variable values in transformed space.
    cv_bounds : np.ndarray
        Vound constraint violations.
    cv_lincon : np.ndarray
        Linear constraint violations.
    cv_lineqcon : np.ndarray
        Linear equality constraint violations.
    f : np.ndarray
        Objective values.
    f_min : np.ndarray
        Minimized objective values.
    g : np.ndarray
        Nonlinear constraint values.
    cv_nonlincon : np.ndarray
        Nonlinear constraints violation.
    m : np.ndarray
        Meta score values.
    m_min : np.ndarray
        Minimized meta score values.
    is_feasible : bool
        True, if individual fulfills all constraints.

    See Also
    --------
    CADETProcess.optimization.Population
    """

    x = Vector()
    x_transformed = Vector()
    cv_bounds = Vector()
    cv_lincon = Vector()
    cv_lineqcon = Vector()
    f = Vector()
    f_min = Vector()
    g = Vector()
    cv_nonlincon = Vector()
    cv_nonlincon_tol = Float()
    m = Vector()
    m_min = Vector()
    is_feasible = Bool()

    def __init__(
        self,
        x: npt.ArrayLike,
        f: Optional[npt.ArrayLike] = None,
        g: Optional[npt.ArrayLike] = None,
        f_min: Optional[npt.ArrayLike] = None,
        x_transformed: Optional[npt.ArrayLike] = None,
        cv_bounds: Optional[npt.ArrayLike] = None,
        cv_lincon: Optional[npt.ArrayLike] = None,
        cv_lineqcon: Optional[npt.ArrayLike] = None,
        cv_nonlincon: Optional[npt.ArrayLike] = None,
        m: Optional[npt.ArrayLike] = None,
        m_min: Optional[npt.ArrayLike] = None,
        independent_variable_names: list[str] = None,
        objective_labels: list[str] = None,
        nonlinear_constraint_labels: list[str] = None,
        meta_score_labels: list[str] = None,
        variable_names: list[str] = None,
        is_feasible: bool = True,
    ) -> None:
        """Initialize Individual Object."""
        self.x = x
        if x_transformed is None:
            x_transformed = x
            independent_variable_names = variable_names
        self.x_transformed = x_transformed

        self.cv_bounds = cv_bounds
        self.cv_lincon = cv_lincon
        self.cv_lineqcon = cv_lineqcon

        self.f = f
        if f_min is None:
            f_min = f
        self.f_min = f_min

        self.g = g
        if g is not None and cv_nonlincon is None:
            cv_nonlincon = g
        self.cv_nonlincon = cv_nonlincon

        self.m = m
        if m_min is None:
            m_min = m
        self.m_min = m_min

        if isinstance(variable_names, np.ndarray):
            variable_names = [s.decode() for s in variable_names]
        self.variable_names = variable_names
        if isinstance(independent_variable_names, np.ndarray):
            independent_variable_names = [
                s.decode() for s in independent_variable_names
            ]
        self.independent_variable_names = independent_variable_names
        if isinstance(objective_labels, np.ndarray):
            objective_labels = [s.decode() for s in objective_labels]
        self.objective_labels = objective_labels
        if isinstance(nonlinear_constraint_labels, np.ndarray):
            nonlinear_constraint_labels = [
                s.decode() for s in nonlinear_constraint_labels
            ]
        self.nonlinear_constraint_labels = nonlinear_constraint_labels
        if isinstance(meta_score_labels, np.ndarray):
            meta_score_labels = [s.decode() for s in meta_score_labels]
        self.meta_score_labels = meta_score_labels

        self.id = hash_array(self.x)
        self.is_feasible = is_feasible

    @property
    def id_short(self) -> str:
        """str: Id shortened to the first seven digits."""
        return self.id[0:7]

    @property
    def is_evaluated(self) -> bool:
        """bool: Return True if individual has been evaluated. False otherwise."""
        if self.f is None:
            return False
        else:
            return True

    @property
    def n_x(self) -> int:
        """int: Number of variables."""
        return len(self.x)

    @property
    def n_f(self) -> int:
        """int: Number of objectives."""
        if self.f is None:
            return 0
        return len(self.f)

    @property
    def n_g(self) -> int:
        """int: Number of nonlinear constraints."""
        if self.g is None:
            return 0
        else:
            return len(self.g)

    @property
    def n_m(self) -> int:
        """int: Number of meta scores."""
        if self.m is None:
            return 0
        else:
            return len(self.m)

    @property
    def cv(self) -> np.ndarray:
        """
        All constraint violations combined.

        (cv_bounds, cv_lincon, cv_lineqcon, cv_nonlincon)

        Returns
        -------
        np.ndarray
            All constraint violations combined.
        """
        cvs = (self.cv_bounds, self.cv_lincon, self.cv_lineqcon, self.cv_nonlincon)
        return np.concatenate([cv for cv in cvs if cv is not None])

    @property
    def dimensions(self) -> tuple[int]:
        """tuple: Individual dimensions (n_x, n_f, n_g, n_m)."""
        return (self.n_x, self.n_f, self.n_g, self.n_m)

    @property
    def objectives_minimization_factors(self) -> np.ndarray:
        """np.ndarray: Array indicating objectives transformed to minimization."""
        return self.f_min / self.f

    @property
    def meta_scores_minimization_factors(self) -> np.ndarray:
        """np.ndarray: Array indicating meta sorces transformed to minimization."""
        return self.m_min / self.m

    def dominates(self, other: "Individual") -> bool:
        """
        Determine if individual dominates other.

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
        if not self.is_evaluated:
            raise CADETProcessError("Individual needs to be evaluated first.")
        if not other.is_evaluated:
            raise CADETProcessError("Other individual needs to be evaluated first.")

        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False

        if not self.is_feasible and not other.is_feasible:
            if np.any(self.cv < other.cv):
                better_in_all = np.all(self.cv <= other.cv)
                strictly_better_in_one = np.any(self.cv < other.cv)
                return better_in_all and strictly_better_in_one

        if self.m is not None:
            self_values = self.m_min
            other_values = other.m_min
        else:
            self_values = self.f_min
            other_values = other.f_min

        if np.any(self_values > other_values):
            return False

        if np.any(self_values < other_values):
            return True

        return False

    def is_similar(self, other: "Individual", tol: float = 1e-1) -> bool:
        """
        Determine if individual is similar to other.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float, optional
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.

        Returns
        -------
        is_similar : bool
            True if individuals are close to each other. False otherwise
        """
        if not tol:
            return False

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

    def is_similar_x(
        self,
        other: "Individual",
        tol: float = 1e-1,
        use_transformed: bool = False,
    ) -> bool:
        """
        Determine if individual is similar to other based on parameter values.

        Parameters
        ----------
        other : Individual
            Other individual
        tol : float
            Relative tolerance parameter.
            To reduce number of entries, a rather high rtol is chosen.
        use_transformed : bool
            If True, use independent transformed space.
            The default is False.

        Returns
        -------
        is_similar : bool
            True if parameters are close to each other. False otherwise
        """
        similar_x = np.allclose(self.x, other.x, rtol=tol)

        return similar_x

    def is_similar_f(self, other: "Individual", tol: float = 1e-1) -> bool:
        """
        Determine if individual is similar to other based on objective values.

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

    def is_similar_g(self, other: "Individual", tol: float | None = 1e-1) -> bool:
        """
        Determine if individual is similar to other based on constraint values.

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

    def is_similar_m(self, other: "Individual", tol: float | None = 1e-1) -> bool:
        """
        Determine if individual is similar to other based on meta score values.

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

    def __str__(self) -> str:
        """str: String representation of the individual."""
        return str(list(self.x))

    def __repr__(self) -> str:
        """str: String representation of the individual."""
        if self.g is None:
            return f"{self.__class__.__name__}({self.x}, {self.f})"
        else:
            return f"{self.__class__.__name__}({self.x}, {self.f}, {self.g})"

    def to_dict(self) -> dict:
        """
        Convert individual to a dictionary.

        Returns
        -------
        dict: A dictionary representation of the individual's attributes.
        """
        data = Dict()

        data.x = self.x
        data.x_transformed = self.x_transformed

        data.cv_bounds = self.cv_bounds
        data.cv_lincon = self.cv_lincon
        data.cv_lineqcon = self.cv_lineqcon

        data.f = self.f
        data.f_min = self.f_min

        if self.g is not None:
            data.g = self.g
            data.cv_nonlincon = self.cv_nonlincon

        if self.m is not None:
            data.m = self.m
            data.m_min = self.m_min

        data.variable_names = self.variable_names
        data.independent_variable_names = self.independent_variable_names
        if self.objective_labels is not None:
            data.objective_labels = self.objective_labels
        if self.nonlinear_constraint_labels is not None:
            data.nonlinear_constraint_labels = self.nonlinear_constraint_labels
        if self.meta_score_labels is not None:
            data.meta_score_labels = self.meta_score_labels

        data.is_feasible = self.is_feasible

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        """
        Create Individual from dictionary representation of its attributes.

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
