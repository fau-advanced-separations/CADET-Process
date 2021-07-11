import numpy as np
import scipy
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError

class Section():
    """Helper class to store parameter state between to sections.

    Attributes
    ----------
    start : float
        Start time of section
    end : float
        End time of section.
    state : float or array_like
        Parameter state during section
    """
    def __init__(self, start, end, state):
        if isinstance(state, bool):
            state = int(state)

        if isinstance(state, (int, float, tuple, list)):
            state = np.array((state), ndmin=2)
            
        self._state = state

        self.start = start
        self.end = end

    @property
    def state(self):
        """
        Each row represents the polynomial coefficients of one component
        in increasing order.
        """
        return self._state
    
    @property
    def n_dim(self):
        return len(self.state)    
    
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
            
        return self.state
    
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
            
        if not ((self.start <= start) & (start < end) & (end <= self.end)):
            raise ValueError('Integration bounds exceed section times')

        return (end - start) * self.state

class PolynomialSection(Section):
    """Helper class to store parameter state between to sections.

    Attributes
    ----------
    start : float
        Start time of section
    end : float
        End time of section.
    state : float or array_like
        Polynomial coefficients of state in order of increasing degree.
    degree : int
        Degree of polynomial to represent state.
    n_dim : int, optional
        Number of state dimensions.
        If not given, it will be inferred from the coeffs dimensions.
    """
    def __init__(self, start, end, state):
        if isinstance(state, bool):
            state = int(state)

        if isinstance(state, (int, float)):
            state = np.array((state), ndmin=2)
                
        if isinstance(state, (tuple, list)):
            for s in state:
                if isinstance(s, (tuple, list)):
                    missing = 4 - len(s)
                    s += missing*(0,)
            state = np.array((state), ndmin=2)

        if state.shape[1] > 4:
            raise CADETProcessError('Only cubic polynomials are supported')
        
        _state = np.zeros((state.shape[0], 4))
        _state[:,0:state.shape[1]] = state

        self._poly = []
        for s in _state:
            poly = np.polynomial.Polynomial(
                s, domain=(start, end), window=(0,1)
            )   
            self._poly.append(poly)

        self.start = start
        self.end = end

    @property
    def state(self):
        """
        Each row represents the polynomial coefficients of one component
        in increasing order.
        """
        return np.array([p.coef for p in self._poly])
    
    @property
    def n_dim(self):
        return len(self._poly)
    
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
            
        if not ((self.start <= start) & (start < end) & (end <= self.end)):
            raise ValueError('Integration bounds exceed section times')

        integ_methods = [p.integ(lbnd=start) for p in self._poly]
        return np.array([i(end) for i in integ_methods])

class TimeLine():
    def __init__(self):
        self._sections = []

    @property
    def sections(self):
        return sorted(self._sections, key=lambda sec: sec.start)
    
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
            If section dimensions do not match Timeline dimensions.
        CADETProcessError
            If section introduces a gap.
        """
        if not isinstance(section, Section):
            raise TypeError('Expected Section')
        if len(self.sections) > 0:
            if section.n_dim != self.n_dim:
                raise CADETProcessError('Number of dimensions not matching')
            if not (section.start == self.end or section.end == self.start):
                raise CADETProcessError('Sections times must be without gaps')

        self._sections.append(section)
    
    @property
    def piecewise_poly(self):
        """list: scipy.interpolate.PPoly for each dimension.

        Returns
        -------
        piecewise_poly : list
            DESCRIPTION.

        """
        x = []
        state = []
        for sec in self.sections:
            state.append(np.array((sec.state),ndmin=2))
            x.append(sec.start)
        x.append(sec.end)
            
        piecewise_poly = []
        for i in range(self.n_dim):
            c = np.array([s[i,:] for s in state])
            c_decreasing = np.fliplr(c)
            p = scipy.interpolate.PPoly(c_decreasing.T, x)
            piecewise_poly.append(p)
        
        return piecewise_poly
    
    @property
    def n_dim(self):
        if len(self.sections) > 0:
            return self.sections[0].n_dim

    def value(self, time):
        """np.array: Value of parameter at given time

        Parameters
        ----------
        time : np.float or array_like
            time points at which to evaluate.

        """
        return np.array([p(time) for p in self.piecewise_poly], ndmin=2).T
    
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
        c = self.sections[section_index].state
        y = self.value(time)
        c[:,0] = y[:,0]
        
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
                
        if not ((self.start <= start) & (start < end) & (end <= self.end)):
            raise ValueError('Integration bounds exceed section times')
        
        integral = [p.integrate(start, end) for p in self.piecewise_poly]
        
        return np.array(integral)
    
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
        
    def plot(self):
        start = self.sections[0].start
        end = self.sections[-1].end
        time = np.linspace(start, end, 1001)
        plt.figure()
        plt.plot(time, self.value(time))

