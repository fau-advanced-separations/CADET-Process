from collections import defaultdict
import itertools

import numpy as np
import scipy
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import NdPolynomial
from CADETProcess import plotting

class Section(metaclass=StructMeta):
    """Helper class to store parameter states between events.

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
    coeffs = NdPolynomial(dep=('n_entries', 'n_poly_coeffs'), default=0)
    
    def __init__(self, start, end, coeffs, n_entries=None, degree=0):
        if n_entries is None:
            if isinstance(coeffs, (int, bool, float)):
                n_entries = 1
            elif isinstance(coeffs, (list, tuple, np.ndarray)) and degree == 0:
                n_entries = len(coeffs)
            else:
                raise ValueError("Ambiguous entries for n_entries and degree")
        self.n_entries = n_entries
        
        if n_entries == 1:
            coeffs = [coeffs]
            
        self.degree = degree
        self.coeffs = coeffs

        self.start = start
        self.end = end
        diff = end-start

        self._poly = []
        for i in range(self.n_entries):
            poly = np.polynomial.polynomial.Polynomial(
                self.coeffs[i], domain=(start, end), window=(0,diff)
            )
            self._poly.append(poly)

        self._poly_der = []
        for iEntry in range(self.n_entries):
            poly_der = self._poly[iEntry].deriv(1)
            self._poly_der.append(poly_der)


    @property
    def n_poly_coeffs(self):
        return self.degree + 1

    def value(self, t):
        """Return value of parameter section at time t.

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
            raise ValueError('Time exceeds section times')
            
        value = np.array([p(t) for p in self._poly])

        return value
    
    def coefficients(self, offset=0):
        coeffs = []
        for i in range(self.n_entries):
            c = self.coeffs[i].copy()
            c[0] = self._poly[i](offset)
            if self.degree > 0:
                c[1] = self._poly_der[i](offset)
            coeffs.append(c)
        
        return np.array(coeffs)
    
    def derivative(self, t, order=1):
        """Return derivative of parameter section at time t.
        
        Parameters
        ----------
        t : float
            Time at which function is evaluated.
        
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
            raise ValueError('Time exceeds section times')
            
        deriv = np.array([p.deriv(t).coef for p in self._poly_der])
        
        return deriv        

    def integral(self, start=None, end=None):
        """Return integral of function in interval [start, end].

        Parameters
        ----------
        start : float, optional
            Lower integration bound.
        end : float, optional
            Upper integration bound.

        Returns
        -------
        Y : float
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
            raise ValueError('Integration bounds exceed section times')

        integ_methods = [p.integ(lbnd=start) for p in self._poly]
        return np.array([i(end) for i in integ_methods])


class TimeLine():
    def __init__(self, n_entries=None):
        if n_entries:
            pass
        self._sections = [] 
    
    @property
    def sections(self):
        return self._sections

    @property
    def degree(self):
        if len(self.sections) > 0:
            return self.sections[0].degree
    
    @property
    def n_entries(self):
        if len(self.sections) > 0:
            return self.sections[0].n_entries
    
    def add_section(self, section, entry_index=None):
        """Add section to TimeLine.

        Parameters
        ----------
        section : Section
            Section to be added.

        Raises
        ------
        TypeError
            If section is not instance of Section.
        CADETProcessError
            If polynomial degree does not match.
        CADETProcessError
            If section introduces a gap.
        """
        if not isinstance(section, Section):
            raise TypeError('Expected Section')
        if len(self.sections) > 0:
            if section.degree != self.degree:
                raise CADETProcessError('Polynomial degree does not match')

            if not (section.start == self.end or section.end == self.start):
                raise CADETProcessError('Sections times must be without gaps')

        self._sections.append(section)
        self._sections = sorted(self._sections, key=lambda sec: sec.start)
    
        self.update_piecewise_poly()
        
    def update_piecewise_poly(self):
        x = []
        coeffs = []
        for sec in self.sections:
            coeffs.append(np.array(sec.coeffs))
            x.append(sec.start)
        x.append(sec.end)
            
        piecewise_poly = []
        for iEntry in range(self.n_entries):
            c = np.array([iCoeff[iEntry,:] for iCoeff in coeffs])
            c_decreasing = np.fliplr(c)
            p = scipy.interpolate.PPoly(c_decreasing.T, x)
            piecewise_poly.append(p)
        
        self._piecewise_poly = piecewise_poly

    @property
    def piecewise_poly(self):
        """list: scipy.interpolate.PPoly for each dimension.
        """
        return self._piecewise_poly

    def value(self, time):
        """np.array: Value of parameter at given time

        Parameters
        ----------
        time : np.float or array_like
            time points at which to evaluate.

        """
        return np.array([p(time) for p in self.piecewise_poly]).T
    
    def coefficients(self, time):
        """Return coefficient of polynomial at given time.

        Parameters
        ----------
        time : float
            Time at which polynomial coefficients are queried.

        Returns
        -------
        coefficients : np.array
            !!! Array of coefficients in ORDER !!!
        """
        section_index = self.section_index(time)
        c = self.sections[section_index].coefficients(time)

        return c
    
    def integral(self, start=None, end=None):
        """Calculate integral of sections in interval [start, end].

        Parameters
        ----------
        start : float, optional
            Lower integration bound.
        end : float, optional
            Upper integration bound.

        Returns
        -------
        Y : float
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
            raise ValueError('Integration bounds exceed section times')

        return np.array([p.integrate(start, end) for p in self.piecewise_poly]).T
        
    def section_index(self, time):
        section_times = np.array(self.section_times)
        
        return np.argmin(time >= section_times) - 1
    
    @property
    def section_times(self):
        if len(self.sections) == 0:
            return []
        
        return [self.sections[0].start] + [sec.end for sec in self.sections]

    @property
    def start(self):
        return self.section_times[0]

    @property
    def end(self):
        return self.section_times[-1]
        
    @plotting.save_fig
    def plot(self):
        """Plot section state over time.
        """
        start = self.sections[0].start
        end = self.sections[-1].end
        time = np.linspace(start, end, 1001)/60
        y = self.value(time)
        
        fig, ax = plotting.setup_figure()
        ax.plot(time,y)
        
        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$state$'
        layout.xlim = (start/60, end/60)
        layout.ylim = (0, 1.1*np.max(y))
        
        plotting.set_layout(fig, ax, layout)        

        return ax


class MultiTimeLine():
    def __init__(self, base_state):
        self.base_state = base_state
        self.n_entries = len(base_state)
        self._degree = None
        self.time_lines = [TimeLine() for _ in range(self.n_entries)]
        
    @property
    def degree(self):
        return self._degree
    
    @degree.setter
    def degree(self, degree):
        self._degree = degree
        
    @property
    def section_times(self):
        time_line_sections = [tl.section_times for tl in self.time_lines]
        
        section_times = set(itertools.chain.from_iterable(time_line_sections))
        
        return sorted(list(section_times))
    
    def add_section(self, section, entry_index=None):
        if self.degree is None:
            self.degree = section.degree
        
        if section.degree != self.degree:
            raise CADETProcessError('Polynomial degree does not match')

        if entry_index is None:
            for tl in self.time_lines:
                tl.add_section(section)
        else:
            if entry_index > self.n_entries:
                raise ValueError("Index exceeds entries.")
        
            self.time_lines[entry_index].add_section(section)

    def combined_time_line(self):
        tl = TimeLine()
        
        section_times = self.section_times
        for iSec in range(len(section_times) - 1):
            start = self.section_times[iSec]
            end = self.section_times[iSec+1]
            
            coeffs = []
            for i, entry in enumerate(self.time_lines):
                if len(entry.sections) == 0:
                    coeffs.append(self.base_state[i])
                else:
                    coeffs.append(entry.coefficients(start)[0])
                    
            section = Section(start, end, coeffs, self.n_entries, self.degree)
            tl.add_section(section)
        
        return tl
    