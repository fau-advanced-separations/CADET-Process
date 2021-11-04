import numpy as np
from scipy.integrate import simps

from CADETProcess import log
from CADETProcess.dataStructure import StructMeta, Bool, UnsignedFloat
from CADETProcess.common import TimeSignal


class StationarityEvaluator(metaclass=StructMeta):
    """Class for checking two succeding chromatograms for stationarity

    Attributes
    ----------

    Notes
    -----
    !!! Implement check_skewness and width deviation
    """
    check_concentration = Bool(default=True)
    max_concentration_deviation = UnsignedFloat(default=0.1)

    check_area = Bool(default=True)
    max_area_deviation = UnsignedFloat(default=1)

    check_height = Bool(default=True)
    max_height_deviation = UnsignedFloat(default=0.1)

    def __init__(self):
        self.logger = log.get_logger('StationarityEvaluator')

    def assert_stationarity(self, conc_old, conc_new):
        """Check Wrapper function for checking stationarity of two succeeding cycles.

        First the module 'stationarity' is imported, then the concentration
        profiles for the current and the last cycles are defined. After this
        all checking function from module 'stafs = FlowSheet(n_comp=2, name=flow_sheet_name)
        tionarity' are called.

        Parameters
        ----------
        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle
        """
        if not isinstance(conc_old, TimeSignal):
            raise TypeError('Expcected TimeSignal')
        if not isinstance(conc_new, TimeSignal):
            raise TypeError('Expcected TimeSignal')

        criteria = {}
        if self.check_concentration:
            criterion, value = self.check_concentration_deviation(conc_old, conc_new)
            criteria['concentration_deviation'] = {
                    'status': criterion,
                    'value': value}
        if self.check_area:
            criterion, value = self.check_area_deviation(conc_old, conc_new)
            criteria['area_deviation'] = {
                    'status': criterion,
                    'value': value}
        if self.check_height:
            criterion, value = self.check_height_deviation(conc_old, conc_new)
            criteria['height_deviation'] = {
                    'status': criterion,
                    'value': value}

        self.logger.debug('Stationrity criteria: {}'.format(criteria))

        if all([crit['status'] for crit in criteria.values()]):
            return True

        return False

    def concentration_deviation(self, conc_old, conc_new):
        """Calculate the concentration profile deviation of two succeeding cycles.

        Parameters
        ----------
        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        concentration_deviation : np.array
            Concentration difference of two succeding cycles.
        """
        return np.max(abs(conc_new.signal - conc_old.signal), axis=0)

    def check_concentration_deviation(self, conc_old, conc_new):
        """Check if deviation in concentration profiles is smaller than eps

        Parameters
        ----------
        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        bool
            True, if concentration deviation smaller than eps, False otherwise.
            False
        """
        conc_dev = self.concentration_deviation(conc_old, conc_new)

        if np.any(conc_dev > self.max_concentration_deviation):
            criterion = False
        else:
            criterion = True

        return criterion, np.max(conc_dev)


    def area_deviation(self, conc_old, conc_new):
        """Calculate the area deviation of two succeeding cycles.

        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        area_deviation : np.array
            Area deviation of two succeding cycles.
        """
        area_old = simps(conc_old.signal, conc_old.time, axis=0)
        area_new = simps(conc_new.signal, conc_new.time, axis=0)

        return abs(area_old - area_new)

    def check_area_deviation(self, conc_old, conc_new, eps=1):
        """Check if deviation in concentration profiles is smaller than eps

        Parameters
        ----------
        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        bool
            True, if area deviation is smaller than eps, False otherwise.
        """
        area_dev = self.area_deviation(conc_old, conc_new)

        if np.any(area_dev > self.max_area_deviation):
            criterion = False
        else:
            criterion = True

        return criterion, area_dev

    def height_deviation(self, conc_old, conc_new):
        """Calculate the height deviation of two succeeding cycles.

        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        height_deviation : np.array
            Height deviation of two succeding cycles.
        """
        height_old = np.amax(conc_old.signal, 0)
        height_new = np.amax(conc_new.signal, 0)

        return abs(height_old - height_new)

    def check_height_deviation(self, conc_old, conc_new):
        """Check if deviation in peak heigth is smaller than eps.

        Parameters
        ----------
        conc_old : TimeSignal
            Concentration profile of previous cycle
        conc_new : TimeSignal
            Concentration profile of current cycle

        Returns
        -------
        bool
            True, if height deviation is smaller than eps, False otherwise.
        """
        abs_height_deviation = self.height_deviation(conc_old, conc_new)

        if np.any(abs_height_deviation > self.max_height_deviation):
            criterion = False
        else:
            criterion = True

        return criterion, abs_height_deviation
