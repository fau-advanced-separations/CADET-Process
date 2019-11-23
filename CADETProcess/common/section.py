from numpy.polynomial import Polynomial

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
    poly : bool
        If True, the states are assumed to be polynomial coefficients.
    """
    def __init__(self, start, end, state, poly=False):
        if poly:
            self._poly = Polynomial(state, domain=(start, end), window=(0,1))
        else:
            self._poly = None

        self.state = state

        self.start = start
        self.end = end

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
        if not (self.start <= t <= self.end):
            raise ValueError('Exceeds bounds')

        if self._poly is not None:
            return self._poly(t)

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

        if not (self.start <= start < end <= self.end):
            raise ValueError('Integration bounds exceed section times')

        if self._poly is not None:
            integ = self._poly.integ(lbnd=start)
            return integ(end)
        else:
            return (end-start) * self.state

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

    def value(self, t):
        for index, section in enumerate(self.sections):

            if index < len(self.sections)-1:
                cond = section.start <= t < section.end
            if index == len(self.sections)-1:
                cond = section.start <= t <= section.end
            if cond:
                return section.value(t)

        raise CADETProcessError('Could not find section')

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

        if not (self.start <= start < end <= self.end):
            raise ValueError('Integration bounds exceed section times')

    @property
    def section_times(self):
            return [self.sections[0].start] + [sec.end for sec in self.sections]
