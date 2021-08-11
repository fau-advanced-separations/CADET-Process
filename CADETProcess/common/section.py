import numpy as np
import scipy
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta, NdPolynomial

class Section(metaclass=StructMeta):
    """Helper class to store parameter states between events.

    Attributes
    ----------
    start : float
        Start time of section
    end : float
        End time of section.
    state : int or float or array_like
        Polynomial coefficients of state in order of increasing degree.
    n_entries : int
        Number of entries (e.g. components, output_states)
    degree : int
        Degree of polynomial to represent state.
        
    Notes
    -----
    if state is int: Set constant value for for all entries
    if state is list: Set value per component (check length!)
    if state is ndarray (or list of lists): set polynomial coefficients
    """
    state = NdPolynomial(dep=('n_entries', 'n_poly_coeffs'), default=0)
    
    def __init__(self, start, end, state, n_entries=None, degree=0):
        if n_entries is None:
            if isinstance(state, (int, bool, float)):
                n_entries = 1
            elif isinstance(state, (list, tuple, np.ndarray)) and degree == 0:
                n_entries = len(state)
            else:
                raise ValueError("Ambiguous entries for n_entries and degree")
        self.n_entries = n_entries
        
        if n_entries == 1:
            state = [state]
            
        self.degree = degree
        self.state = state

        self._poly = []
        for i in range(self.n_entries):
            poly = np.polynomial.Polynomial(
                self.state[i], domain=(start, end), window=(0,1)
            )
            self._poly.append(poly)

        self.start = start
        self.end = end

    @property
    def n_poly_coeffs(self):
        return self.degree + 1

    def value(self, t):
        """Return value of function at time t.

        Parameters
        ----------
        t : float
            Time at which function is evaluated.

        Returns
        -------
        y : float
            Value of attribute at time t.

        Raises
        ------
        ValueError
            If t is lower than start or larger than end of section time.
        """
        if np.any(t < self.start) or np.any(self.end < t):
            raise ValueError('Time exceeds section times')
            
        value = np.array([p(t) for p in self._poly])

        return value

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
    def __init__(self):
        self._sections = [] 
    
    @property
    def sections(self):
        return sorted(self._sections, key=lambda sec: sec.start)

    @property
    def degree(self):
        if len(self.sections) > 0:
            return self.sections[0].degree
    
    @property
    def n_entries(self):
        if len(self.sections) > 0:
            return self.sections[0].n_entries
    
    def add_section(self, section):
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
    
        self.update_piecewise_poly()

    def update_piecewise_poly(self):
        x = []
        state = []
        for sec in self.sections:
            state.append(np.array(sec.state))
            x.append(sec.start)
        x.append(sec.end)
            
        piecewise_poly = []
        for i in range(self.n_entries):
            c = np.array([s[i,:] for s in state])
            c_decreasing = np.fliplr(c)
            p = scipy.interpolate.PPoly(c_decreasing.T, x)
            piecewise_poly.append(p)
        c_decreasing = np.fliplr(state)
        
        self._piecewise_poly = piecewise_poly
        

    @property
    def piecewise_poly(self):
        """list: scipy.interpolate.PPoly for each dimension.

        Returns
        -------
        piecewise_poly : list
            DESCRIPTION.

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
        c = self.sections[section_index].state.copy()
        y = self.value(time)
        c[:,0] = y
        
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
        return [self.sections[0].start] + [sec.end for sec in self.sections]

    @property
    def start(self):
        return self.section_times[0]
    
    @property
    def end(self):
        return self.section_times[-1]
        
    def plot(self, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()        
        start = self.sections[0].start
        end = self.sections[-1].end
        time = np.linspace(start, end, 1001)
        
        ax.plot(time, self.value(time))
        
        ax.set_xlabel("time / s")
        ax.set_ylabel("state")
        
        if show:
            plt.show()
            
        return ax

