from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy
from matplotlib.axes import Axes
from numpy.exceptions import VisibleDeprecationWarning

from CADETProcess import CADETProcessError, plotting
from CADETProcess.dataStructure import NdPolynomial, Structure

__all__ = ["Section", "TimeLine", "MultiTimeLine"]


class Section(Structure):
    """
    Helper class to store parameter states between events.

    Attributes
    ----------
    start : float
        Start time of section
    end : float
        End time of section.
    coeffs : int or float or array_like
        Polynomial coefficients of state in order of increasing degree.
    n_entries : int
        Number of entries (e.g. components, output_states)
    degree : int
        Degree of polynomial to represent state.

    Notes
    -----
        if coeffs is int: Set constant value for for all entries
        if coeffs is list: Set value per component (check length!)
        if coeffs is ndarray (or list of lists): set polynomial coefficients
    """

    coeffs = NdPolynomial(size=("n_entries", "n_poly_coeffs"))

    def __init__(
        self,
        start: float,
        end: float,
        coeffs: int | float | npt.ArrayLike,
        is_polynomial: bool = False,
    ) -> None:
        """Construct section object."""
        if start > end:
            raise ValueError("End time must be greater than start time")

        self.start = start
        self.end = end
        diff = end - start

        coeffs = np.array(coeffs, ndmin=1, dtype=np.float64)
        self.parameter_shape = coeffs.shape

        if is_polynomial:
            coeffs = np.array(coeffs, ndmin=2, dtype=np.float64)
            self.degree = coeffs.shape[-1] - 1
            self.n_entries = coeffs.shape[0]

            self.coeffs = coeffs
        else:
            self.degree = 0
            self.n_entries = coeffs.size
            self.coeffs = coeffs.reshape((coeffs.size, 1))

        self._poly = []
        for i in range(self.n_entries):
            poly = np.polynomial.polynomial.Polynomial(
                self.coeffs[i], domain=(start, end), window=(0, diff)
            )
            self._poly.append(poly)

        self._poly_der = []
        for iEntry in range(self.n_entries):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                poly_der = self._poly[iEntry].deriv(1)
                self._poly_der.append(poly_der)

    @property
    def is_polynomial(self) -> bool:
        """bool: True if Section represents polynomial parameter. False otherwise."""
        if self.degree > 0:
            return True
        return False

    @property
    def n_poly_coeffs(self) -> int:
        """int: Number of polynomial coefficients."""
        return self.degree + 1

    @property
    def is_single_entry(self) -> bool:
        """bool: True if Section contains single entry. False otherwise."""
        if self.n_entries > 1:
            return True

        if self.is_polynomial and self.parameter_shape.ndim == 1:
            return True

        return False

    def value(self, t: float) -> float:
        """
        Return value of parameter section at time t.

        Parameters
        ----------
        t : float
            Time at which function is evaluated.

        Returns
        -------
        y : float
            Value of parameter state at time t.

        Raises
        ------
        ValueError
            If t is lower than start or larger than end of section time.
        """
        if np.any(t < self.start) or np.any(self.end < t):
            raise ValueError("Time exceeds section times")

        value = np.array([p(t) for p in self._poly])

        return value

    __call__ = value

    def coefficients(self, offset: float = 0.0) -> np.ndarray:
        """
        Get coefficients at (time) offset.

        Parameters
        ----------
        offset : float
            (Time) offset to be evaluated.

        Returns
        -------
        coeffs : np.ndarray
            Coefficients at offset.
        """
        coeffs = []
        for i in range(self.n_entries):
            c = self.coeffs[i].copy()
            c[0] = self._poly[i](offset)
            if self.degree > 0:
                c[1] = self._poly_der[i](offset)
            coeffs.append(c)

        return np.array(coeffs).reshape(self.parameter_shape)

    def derivative(self, t: float, order: Optional[int] = 1) -> np.ndarray:
        """
        Return derivative of parameter section at time t.

        Parameters
        ----------
        t : float
            Time at which function is evaluated.
        order : int, default=1
            Order of deriviation. @TODO: Not yet implemented.

        Returns
        -------
        y_dot : float
            Derivative of parameter state at time t.

        Raises
        ------
        ValueError
            If t is lower than start or larger than end of section time.
        ValueError
            If order is larger than polynomial degree
        """
        if np.any(t < self.start) or np.any(self.end < t):
            raise ValueError("Time exceeds section times")

        deriv = np.array([p.deriv(t).coef for p in self._poly_der])

        return deriv

    def integral(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> np.ndarray:
        """
        Return integral of function in interval [start, end].

        Parameters
        ----------
        start : float, optional
            Lower integration bound.
        end : float, optional
            Upper integration bound.

        Returns
        -------
        Y : np.ndarray
            Value of definite integral between start and end.

        Raises
        ------
        ValueError
            If integration bounds exceed section times.
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if not ((self.start <= start) & (start <= end) & (end <= self.end)):
            raise ValueError("Integration bounds exceed section times")

        integ_methods = [p.integ(lbnd=start) for p in self._poly]
        return np.array([i(end) for i in integ_methods])

    def __repr__(self) -> str:
        """str: String representation of the Section."""
        args = f"start={self.start}, end={self.end}, coeffs={self.coeffs}"
        if self.degree > 0:
            args += f", degree={self.degree}"

        return f"Section({args})"


class TimeLine:
    """
    Class representing a timeline of time-varying data.

    The timeline is made up of Sections, which are continuous time intervals.
    Each Section represents a piecewise polynomial function that defines the
    variation of a given parameter over time.

    Attributes
    ----------
    sections : List[Section]
        List of Sections that make up the timeline.
    """

    def __init__(self) -> None:
        """Initialize TimeLine object."""
        self._sections = []

    @property
    def sections(self) -> list:
        """list: Sections of the TimeLine."""
        return self._sections

    @property
    def degree(self) -> int:
        """int: Degree of the polynomial functions used to represent each Section."""
        if len(self.sections) > 0:
            return self.sections[0].degree

    @property
    def n_entries(self) -> int:
        """int: Number of entries in the parameter vector for each Section."""
        if len(self.sections) > 0:
            return self.sections[0].n_entries

    def add_section(self, section: Section) -> None:
        """
        Add a Section to the timeline.

        Parameters
        ----------
        section : Section
            The Section to be added to the timeline.

        Raises
        ------
        TypeError
            If section is not an instance of the Section class.
        CADETProcessError
            If the polynomial degree of the Section does not match the degree of
            the other Sections in the timeline.
        CADETProcessError
            If the Section introduces a gap in the timeline.
        """
        if not isinstance(section, Section):
            raise TypeError("Expected Section")
        if len(self.sections) > 0:
            if section.degree != self.degree:
                raise CADETProcessError("Polynomial degree does not match")

            if not (section.start == self.end or section.end == self.start):
                raise CADETProcessError("Sections times must be without gaps")

        self._sections.append(section)
        self._sections = sorted(self._sections, key=lambda sec: sec.start)

        self.update_piecewise_poly()

    def update_piecewise_poly(self) -> None:
        """Update the piecewise polynomial representation of the timeline."""
        x = []
        coeffs = []
        for sec in self.sections:
            coeffs.append(np.array(sec.coeffs))
            x.append(sec.start)
        x.append(sec.end)

        piecewise_poly = []
        for iEntry in range(self.n_entries):
            c = np.array([iCoeff[iEntry, :] for iCoeff in coeffs])
            c_decreasing = np.fliplr(c)
            p = scipy.interpolate.PPoly(c_decreasing.T, x)
            piecewise_poly.append(p)

        self._piecewise_poly = piecewise_poly

    @property
    def piecewise_poly(self) -> list:
        """list: scipy.interpolate.PPoly for each dimension."""
        return self._piecewise_poly

    def value(self, time: float) -> np.ndarray:
        """
        np.ndarray: Value of parameter at given time.

        Parameters
        ----------
        time : np.float or array_like
            time points at which to evaluate.
        """
        return np.array([p(time) for p in self.piecewise_poly]).T

    def coefficients(self, time: float) -> np.ndarray:
        """
        Return coefficient of polynomial at given time.

        Parameters
        ----------
        time : float
            Time at which polynomial coefficients are queried.

        Returns
        -------
        coefficients : np.ndarray
            !!! Array of coefficients in ORDER !!!
        """
        section_index = self.section_index(time)
        c = self.sections[section_index].coefficients(time)

        return c

    def integral(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate integral of sections in interval [start, end].

        Parameters
        ----------
        start : float, optional
            Lower integration bound.
        end : float, optional
            Upper integration bound.

        Returns
        -------
        Y : np.ndarray
            Value of definite integral between start and end.

        Raises
        ------
        ValueError
            If integration bounds exceed section times.
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if not ((self.start <= start) & (start <= end) & (end <= self.end)):
            raise ValueError("Integration bounds exceed section times")

        return np.array([p.integrate(start, end) for p in self.piecewise_poly]).T

    def section_index(self, time: float) -> int:
        """
        Return the index of the section that contains the specified time.

        Parameters
        ----------
        time : float
            The time to check.

        Returns
        -------
        int
            The index of the section that contains the specified time.
        """
        section_times = np.array(self.section_times)

        return np.argmin(time >= section_times) - 1

    @property
    def section_times(self) -> list[float]:
        """List of float: The start and end times of all sections in the timeline."""
        if len(self.sections) == 0:
            return []

        return [self.sections[0].start] + [sec.end for sec in self.sections]

    @property
    def start(self) -> float:
        """float: The start time of the timeline."""
        return self.section_times[0]

    @property
    def end(self) -> float:
        """float: The end time of the timeline."""
        return self.section_times[-1]

    @plotting.create_and_save_figure
    def plot(self, ax: Axes, x_axis_in_minutes: bool = True) -> Axes:
        """
        Plot the state of the timeline over time.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.
        x_axis_in_minutes: bool, optional
            If True, the x-axis will be plotted using minutes. The default is True.

        Returns
        -------
        ax : Axes
            The axes with the plot of the timeline state.
        """
        start = self.sections[0].start
        end = self.sections[-1].end
        time = np.linspace(start, end, 1001)
        y = self.value(time)

        if x_axis_in_minutes:
            time = time / 60
            start = start / 60
            end = end / 60

        ax.plot(time, y)

        layout = plotting.Layout()
        layout.x_label = "$time~/~s$"
        if x_axis_in_minutes:
            layout.x_label = "$time~/~min$"
        layout.y_label = "$state$"
        layout.x_lim = (start, end)
        layout.y_lim = (np.min(y), 1.1 * np.max(y))

        plotting.set_layout(ax, layout)

        return ax

    @classmethod
    def from_constant(cls, start: float, end: float, value: float) -> TimeLine:
        """
        Create a timeline with a constant value for a given time range.

        Parameters
        ----------
        start : float
            The start time of the time range.
        end : float
            The end time of the time range.
        value : float
            The value of the timeline during the time range.

        Returns
        -------
        TimeLine
            A TimeLine instance with a single section with the specified constant value.
        """
        tl = cls()
        tl.add_section(Section(start, end, value))

        return tl

    @classmethod
    def from_profile(
        cls,
        time: npt.ArrayLike,
        profile: npt.ArrayLike,
        s: float = 1e-6,
    ) -> TimeLine:
        """
        Create a timeline from a profile.

        Parameters
        ----------
        time : array_like
            The time values of the profile.
        profile : array_like
            The profile values.
        smoothing : float, optional
            The smoothing factor for the spline interpolation. The default is 1e-6.

        Returns
        -------
        TimeLine
            A TimeLine instance with polynomial sections created from the profile.
        """
        from scipy import interpolate

        tl = cls()

        tck = interpolate.splrep(time, profile, s=s)
        ppoly = interpolate.PPoly.from_spline(tck)

        for i, (start, sec) in enumerate(zip(ppoly.x, ppoly.c.T)):
            if i < 3:
                continue
            elif i > len(ppoly.x) - 5:
                continue
            end = ppoly.x[i + 1]
            tl.add_section(Section(start, end, np.flip(sec), is_polynomial=True))

        return tl


class MultiTimeLine:
    """
    Class for a collection of TimeLines with the same number of entries.

    Attributes
    ----------
    base_state : np.ndarray
        The base state that each TimeLine represents.
    n_entries : int
        The number of entries in each TimeLine.
    time_lines : list of TimeLine
        The collection of TimeLines.
    degree : int
        The degree of the polynomials in each section.
    """

    def __init__(self, base_state: list, is_polynomial: bool = False) -> None:
        """
        Initialize a MultiTimeLine instance.

        Parameters
        ----------
        base_state : list
            The base state that each TimeLine represents.
        is_polynomial : bool, optional
            Option wheter MultiTimeLine is polynomial. The default is False

        """
        base_state = np.array(base_state, ndmin=1, dtype=np.float64)
        self.is_polynomial = is_polynomial

        if self.is_polynomial:
            if base_state.ndim == 1:
                self.is_single_entry = True
            else:
                self.is_single_entry = False
            self.base_state = np.array(base_state, ndmin=2, dtype=np.float64)
            self.degree = self.base_state.shape[-1] - 1
        else:
            self.degree = 0
            self.base_state = base_state

        self.time_lines = [TimeLine() for _ in range(self.size)]

    @property
    def degree(self) -> int:
        """int: The degree of the polynomials in each section."""
        return self._degree

    @degree.setter
    def degree(self, degree: int) -> None:
        self._degree = degree

    @property
    def n_entries(self) -> int:
        """int: Number of entries handled by MultiTimeline."""
        if self.degree > 0:
            return len(self.base_state)
        return self.base_state.size

    @property
    def size(self) -> int:
        """int: Total number of internal TimeLines handled Number by MultiTimeline."""
        return self.base_state.size

    @property
    def section_times(self) -> list:
        """list: Combined section times of all TimeLines."""
        time_line_sections = [tl.section_times for tl in self.time_lines]

        section_times = set(itertools.chain.from_iterable(time_line_sections))

        return sorted(list(section_times))

    def add_section(self, section: Section, entry_index: tuple) -> None:
        """
        Add section to TimeLine with specific entry index.

        Parameters
        ----------
        section : Section
            Section to be added.
        entry_index : tuple
            Index of the entry in the base_state for which the section will be added.

        Raises
        ------
        ValueError
            If entry index is out of bounds for base_state.
        """
        index = flatten_index(self.base_state.shape, entry_index)[0]
        if index > self.size:
            raise CADETProcessError("Entry index is out of bounds.")
        self.time_lines[index].add_section(section)

    @property
    def combined_time_line(self) -> TimeLine:
        """TimeLine: Object representing combination of all timelines in the MultiTimeLine."""
        tl = TimeLine()

        n_poly_coeffs = self.degree + 1

        coeffs = self.base_state

        section_times = self.section_times
        for iSec in range(len(section_times) - 1):
            start = self.section_times[iSec]
            end = self.section_times[iSec + 1]

            if not self.is_polynomial:
                for i, entry in enumerate(self.time_lines):
                    if len(entry.sections) > 0:
                        index = unflatten_index(self.base_state.shape, i)[0]
                        coeff = entry.coefficients(start)[0]
                        coeffs[index] = coeff
            else:
                for i_entry in range(self.n_entries):
                    tl_indices = slice(
                        i_entry * n_poly_coeffs, (i_entry + 1) * n_poly_coeffs
                    )
                    i_entry_tl = self.time_lines[tl_indices]
                    for i_poly, i_poly_tl in enumerate(i_entry_tl):
                        if self.is_single_entry:
                            index = (0, i_poly)
                        else:
                            index = (i_entry, i_poly)

                        if (
                            len(i_poly_tl.sections) > 0
                            and start in i_poly_tl.section_times
                        ):
                            coeffs[index] = i_poly_tl.coefficients(start)[0]

            section = Section(start, end, coeffs, self.is_polynomial)
            tl.add_section(section)

            coeffs = tl.coefficients(end)

        return tl


def generate_indices(
    shape: tuple[int, ...],
    indices: Optional[list[list[int]]] = None,
) -> list[tuple]:
    """
    Generate tuples representing indices for an array with a given shape.

    This method allows specifying a list of indices where each entry can also contain
    slices.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array to be indexed.
    indices : list of list of int, optional
        A list where each sub-list contains indices for one dimension of the array.
        'None' indicates a full slice (':') for that dimension.

    Raises
    ------
    ValueError
        If 'parameter' is a scalar or if an index in 'indices' is out of bounds.

    Returns
    -------
    list
        A list of tuples, where each tuple represents indices into the 'parameter'
        array. If the 'parameter' array was 1D, this will be a list of integers instead.

    Examples
    --------
    >>> parameter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> indices = [[0, 1], [1, 2]]
    >>> generate_indices(parameter, indices)
    [(0, 1), (1, 2)]
    """
    if not shape:
        raise ValueError(
            "Shape must not be empty, scalar parameters are not supported."
        )

    if indices is None:
        indices = np.s_[:]

    if not isinstance(indices, list):
        indices = [
            indices,
        ]

    indices_array = np.array(indices, ndmin=1)

    indices = []
    for ind in indices_array:
        ind = tuple(np.array(ind, ndmin=1))
        indices.append(ind)

    _validate_indices(shape, indices)

    return indices


def _validate_indices(shape: tuple[int, ...], indices: list[list[int]]) -> None:
    """Validate that all indices can be set in an array with shape `shape`."""
    param_ref = np.arange(np.prod(shape)).reshape(shape)

    for ind in indices:
        param_ref[ind]


def unravel(shape: tuple[int, ...], indices: list[int] | tuple) -> list[tuple]:
    """
    Unravel indices of a multi-dimensional array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the original array.
    indices : list of int or tuple
        The indices in the flattened array to be unraveled.

    Returns
    -------
    list of tuple
        The unraveled indices in the multi-dimensional array.
    """
    if len(shape) == 0:
        return []

    indices_flat_ref = np.arange(np.prod(shape)).reshape(shape)

    indices_unraveled = []
    for ind in indices:
        indices_flat = indices_flat_ref[ind].flatten()
        indices_unravel = np.unravel_index(indices_flat, shape)
        if not isinstance(indices_flat, (int, np.int64)):
            indices_unravel = list(zip(*indices_unravel))
        else:
            indices_unravel = [indices_unravel]
        indices_unraveled += indices_unravel

    return indices_unraveled


def flatten_index(shape: tuple[int], indices: tuple | list[tuple]) -> list[int]:
    """
    Flatten indices to access array.

    Parameters
    ----------
    shape: tuple
        Shape of the array to be indexed
    indices : tuple or list of tuples
        Indices in tuple notation

    Returns
    -------
    indices_flat : list
        List of indices in flat notation.
    """
    if not isinstance(indices, list):
        indices = [indices]

    indices_flat_ref = np.arange(np.prod(shape)).reshape(shape)

    return [indices_flat_ref[i] for i in indices]


def unflatten_index(shape: tuple[int, ...], indices_flat: int | list[int]) -> list[int]:
    """
    Unflatten indices to access array.

    Parameters
    ----------
    shape : tuple
        Shape of the array to be indexed
    indices_flat : int or list of ints
        Flat indices

    Returns
    -------
    indices : list
        List of unflattened indices.
    """
    if not isinstance(indices_flat, list):
        indices_flat = [indices_flat]

    indices_unravel = np.unravel_index(indices_flat, shape)
    indices = list(zip(*indices_unravel))

    return indices


def get_inhomogeneous_shape(value: np.ndarray) -> list[tuple[int, ...]]:
    """If array is inhomogeneous, return list with shape of every element."""
    with warnings.catch_warnings():  # Catch warnings for compatibility with numpy<1.24
        warnings.simplefilter("error")
        try:
            return np.array(value).shape
        except (ValueError, VisibleDeprecationWarning):
            pass

    shape = []

    for i in value:
        i_shape = get_inhomogeneous_shape(i)
        shape.append(i_shape)

    return shape


def get_full_shape(inhomogeneous_shape: list) -> tuple[int]:
    """Create full shape from inhomogeneous shape to be used with numpy arrays."""
    first_dimension = len(inhomogeneous_shape)

    sub_dims = ()
    for sub_dim in inhomogeneous_shape:
        if not isinstance(sub_dim, tuple):
            sub_dim = get_full_shape(sub_dim)

        sub_dims += (sub_dim,)

    max_dims = {}
    for el in sub_dims:
        for i, dim in enumerate(el):
            try:
                max_dims[i] = max(max_dims[i], dim)
            except KeyError:
                max_dims[i] = dim

    dims = (first_dimension,) + tuple(max_dims.values())

    return dims


def extract_inhomogeneous_array(
    full_array: np.ndarray, inhomogeneous_shape: tuple[int, ...]
) -> None:
    """Get inhomogeneous array from full_array."""
    array = []
    for i in inhomogeneous_shape:
        array.append(full_array[i, :])
