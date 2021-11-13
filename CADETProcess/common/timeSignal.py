import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure  import StructMeta
from CADETProcess.dataStructure import Float, List, NdArray, String
from CADETProcess import plotting

class TimeSignal(metaclass=StructMeta):
    """Class for storing concentration profiles after simulation.

    Attributes
    ----------
    time : NdArray
        NdArray object for the time of a chromatogram.
    signal : NdArray
        NdArray of the concentration of a chromatogram.
    cycle_time : float
            Maximum value of time vector.
    """
    time = NdArray()
    signal = NdArray()
    cycle_time = Float()
    name = String()

    def __init__(self, time, signal, name=''):
        self.time = time
        self.signal = signal
        self.name = name
        self.cycle_time = float(max(self.time))

    @plotting.save_fig
    def plot(self, start=0, end=None):
        """Plots the whole time_signal for each component.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            end time for plotting

        See also
        --------
        plotlib
        plot_purity
        """
        x = self.time / 60
        y = self.signal

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mol \cdot L^{-1}$'
        layout.xlim = (start, end)
        layout.ylim = (0, 1.1*np.max(y))
        
        plotting.set_layout(fig, ax, layout)        

        return ax

    @property
    def local_purity(self):
        """Returns the local purity of the signal.

        Creates an array with the size of the signal with zero values. An array
        with the sum of the signal is created. Every value for signal_sum under
        a defined value is set to NaN. The other values are set for each
        component. Errorstatehandling of floating point error by the division:
        ignores the divide and invalid. Returns a NaN, for zero and infinity
        with large finite numbers.

        Returns
        -------
        local_purity : NdArray
            Returns the local purity for each component as an array.
        """
        purity = np.zeros(self.signal.shape)
        signal_sum = self.signal.sum(1)
        signal_sum[signal_sum < 1e-6] = np.nan
        for comp in range(self.n_comp):
            signal = self.signal[:,comp]
            with np.errstate(divide='ignore', invalid='ignore'):
                purity[:,comp] = np.divide(signal, signal_sum)
        purity = np.nan_to_num(purity)

        return np.nan_to_num(purity)
    
    @plotting.save_fig
    def plot_purity(self, start=0, end=None):
        """Plots the local purity for each component of the concentration
        profile.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            ent time for plotting

        See also
        --------
        plotlib
        plot
        """
        x = self.time / 60
        y = self.local_purity * 100

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mol \cdot L^{-1}$'
        layout.xlim = (start, end)
        layout.ylim = (0, 1.1*np.max(y))
        
        plotting.set_layout(fig, ax, layout)        
        
        return ax

    @property
    def n_comp(self):
        """Number of components of the signal

        Returns
        --------
        n_comp : int
            Returns the number of the components for the signal.
        """
        return self.signal.shape[1]

    def __str__(self):
        return self.__class__.__name__


class Chromatogram(TimeSignal):
    """Class for storing the concentration profile after simulation.

    Defines the time, the signal, the mass of feed, the volume of eluent and
    the solid phase and the flow_rate. Also a List object for the
    fractionation_state is initialized. If no mass of feed is set, it is
    calculated by integration of the whole chromatogram. For each component, a
    InterpolatedUnivariateSpline object is created which is later used for
    interpolation and integration

    Attributes
    ----------
    process_meta : ProcessMeta
        Additional information required for calculating performance

    See also
    --------
    TimeSignal
    EventHandler
    ProcessMeta
    Performance
    """
    _fractionation_state = List()

    def __init__(self, time, signal, Q, *args, **kwargs):
        super().__init__(time, signal, *args, **kwargs)

        self.Q = Q

        vec_q_value = np.vectorize(Q.value)
        q_vector = vec_q_value(time)
        dm_dt = signal * q_vector[:, None]

        self.interpolated_dm_dt = InterpolatedSignal(self.time, dm_dt)
        self.interpolated_Q = InterpolatedUnivariateSpline(time, q_vector)

    def fraction_mass(self, start, end):
        """Component mass in a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_mass : np.array
            Mass of all components in the fraction
        """
        return self.interpolated_dm_dt.integral(start, end)

    def fraction_volume(self, start, end):
        """Volume of a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_volume : np.array
            Volume of the fraction
        """
        return self.interpolated_Q.integral(start, end)

    @property
    def fractionation_state(self):
        """Returns the current fractionation position


        First it sets the state_length to the length of number of component +1.
        For a zero length the fractionation_state list is empty. If the state
        is >= the state_length and an integer a CADETProcessError is raised. The
        entry of the list is set to the state_length and the
        fractionation_state is set to 1. Else it sets the fractionation_state
        to state and raises a CADETProcessError if the length of the sate unequals
        the state_length or the sum of the states unequals 1.

        Parameters
        ----------
        state : dict
            Dict with the flow_rates and the values of them.

        Raises
        ------
        CADETProcessError
            If state is integer and the state >= the state_length.
            If the length of the states is unequal the state_length
            If the sum of the states is unequal 1

        Returns
        -------
        fractionation_state : List
            Object from class List, contains the fractionation_states.

        See also
        --------
        unitOperation.output_state

        Notes
        -----
        Maybe some mistakes in the Error description
        """
        if self._fractionation_state is None:
            self.fractionation_state = 0

        return self._fractionation_state

    @fractionation_state.setter
    def fractionation_state(self, state):
        state_length = self.n_comp + 1

        if state_length == 0:
            fractionation_state = []

        if type(state) is int:
            if state >= state_length:
                raise CADETProcessError('Index exceeds fractionation states')

            fractionation_state = [0] * state_length
            fractionation_state[state] = 1
        else:
            if len(state) != state_length:
                raise CADETProcessError('Expected length {}.'.format(state_length))

            elif sum(state) != 1:
                raise CADETProcessError('Sum of fractions must be 1')

            fractionation_state = state

        self._fractionation_state = fractionation_state


class InterpolatedSignal():
    def __init__(self, time, signal):
        self._signal = [
                InterpolatedUnivariateSpline(time, signal[:,comp])
                for comp in range(signal.shape[1])
                ]

    def integral(self, start, end):
        return np.array([
            self._signal[comp].integral(start, end)
            for comp in range(len(self._signal))
        ])

    def __call__(self, t):
        return np.array([
            self._signal[comp](t) for comp in range(len(self._signal))
        ])
