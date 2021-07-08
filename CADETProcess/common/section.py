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
        End time of section
    state : float or list of floats
        StateTuple of polynomial coefficients in order of increasing degree.
    """
    def __init__(self, start, end, state):
        if isinstance(state, bool):
            state = int(state)
            
        if isinstance(state, (int, float)):
            state = (state,)
            
        if not isinstance(state, tuple):
            raise TypeError('Expected tuple')
        
        if len(state) <= 4:
            missing = 4 - len(state)
            state = state + missing*(0,)
        else:
            raise CADETProcessError('Only cubic polynomials are supported')

        self._poly = np.polynomial.Polynomial(
            state, domain=(start, end), window=(0,1)
        )

        self.start = start
        self.end = end

    @property
    def state(self):
        return self._poly.coef
    
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
            If t is lower than start or higher than end of section times.
        """
        if np.any(t < self.start) or np.any(self.end < t):
            raise ValueError('Time exceeds section times')

        return self._poly(t)


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
            
        if not (self.start <= start) & (start < end) & (end < self.end):
            raise ValueError('Integration bounds exceed section times')

        integ = self._poly.integ(lbnd=start)
        return integ(end)

class TimeLine():
    def __init__(self):
        self._sections = []

    @property
    def sections(self):
        return self._sections
    
    def add_section(self, section):
        if not isinstance(section, Section):
            raise CADETProcessError('Expected Section')

        self._sections.append(section)

    def value(self, time):
        x = [0.0]
        coeffs = []
        
        for sec in self.sections:
            coeffs.append(np.flip(sec.state))
            x.append(sec.end)
        
        piecewise_poly = scipy.interpolate.PPoly(np.array(coeffs).T, x)
        
        return piecewise_poly(time)

    def integral(self, start, end):
        """Return integral of sections in interval [start, end].

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
                
        if not (self.start <= start) & (start < end) & (end < self.end):
            raise ValueError('Integration bounds exceed section times')
    

    @property
    def section_times(self):
            return [self.sections[0].start] + [sec.end for sec in self.sections]
        
    def plot(self):
        plt.plot(self.time, self.value(self.time))

