import logging
import os

from addict import Dict
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.common import settings
from CADETProcess.common import String
from CADETProcess.common import EventHandler
from CADETProcess.common import plotlib, PlotParameters
from CADETProcess.common import Performance
from CADETProcess.common import Chromatogram
from CADETProcess.common import ProcessMeta

from CADETProcess.fractionation.fractions import Fraction, FractionPool

class Fractionator(EventHandler):
    """Class for Chromatogram Fractionation

    To set Events for starting and ending a fractionation it inherits from the
    EventHandler class. It defines a ranking list for components as a
    DependentlySizedUnsignedList with the number of components as dependent
    size. The time_signal to fractionate is defined. If no ranking is set,
    every component is equivalently.

    Attributes
    -----------
    chromatogram : Chromatogram
        Object of the class TimeSignal, array with the concentration over time
        for a simulated process.

    """
    name = String(default='Fractionator')
    performance_keys = ['mass', 'concentration', 'purity', 'recovery',
        'productivity', 'eluent_consumption']


    def __init__(self, process_meta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.process_meta = process_meta
        self._chromatograms = []
        self.reset()

    @property
    def process_meta(self):
        return self._process_meta

    @process_meta.setter
    def process_meta(self, process_meta):
        if not isinstance(process_meta, ProcessMeta):
            raise TypeError('Expected ProcessMeta')
        self._process_meta = process_meta

    def add_chromatogram(self, chromatogram):
        """Add Chromatogram to list of chromatograms to be fractionized
        """
        if not isinstance(chromatogram, Chromatogram):
            raise CADETProcessError('Expected Chromatogram')
        if len(self._chromatograms) > 0:
            if chromatogram.n_comp != self._chromatograms[0].n_comp:
                raise CADETProcessError('Number of components don\'t match')
            if chromatogram.cycle_time != self._chromatograms[0].cycle_time:
                raise CADETProcessError('Cycle_time does not match')
        self._chromatograms.append(chromatogram)

        self.reset()

    @property
    def chromatograms(self):
        """Chromatogram to be fractionized.

        After setting, the cycle time is set and the results are reset.

        Parameters
        ----------
        chrmatogram : Chromatogram
            Chromatogram to be fractionized.

        Raises
        ------
        CADETProcessError
            If the chromatogram is not a Chromatogram instance

        Returns
        -------
        chromatogram : Chromatogram
            Chromatogram to be fractionized

        See also
        ---------
        Chromatogram
        reset
        cycle_time
        """
        return self._chromatograms

    @property
    def n_comp(self):
        """Shortcut property to Chromatogram.n_comp.

        Returns
        --------
        n_comp : int
            Number of components to be fractionized
        """
        return self.chromatograms[0].n_comp

    @property
    def cycle_time(self):
        """Shortcut property to Chromatogram.cycle_time.

        Returns
        --------
        cycle_time : ndarray
            Cycle time of process.
        """
        return self.chromatograms[0].cycle_time


    @property
    def time(self):
        """Shortcut property to Chromatogram time vector.

        Returns
        --------
        time : NdArray
            Sets the time from TimeSignal object.
        """
        return self.chromatograms[0].time

    def plot_fraction_signal(self, start=0, end=None, index=0,
                             show=False, save_path=None):
        """Plot the signal without the waste fractions.

        See also
        --------
        plotlib
        plot_purity
        """
        chrom = self.chromatograms[index]
        time_line = self.event_parameter_time_lines[str(index)]
        fill_regions = []
        x = chrom.time/60
        y = chrom.signal

        for sec in time_line.sections:
            comp_index = int(np.where(sec.state)[0])
            if comp_index == self.n_comp:
                color_index=-1
                text = 'W'
            else:
                color_index = comp_index
                text = str(comp_index + 1)

            if sec.start != sec.end:
                fill_regions.append({
                        'start': sec.start/60,
                        'end': sec.end/60,
                        'y_max': 1.1*np.max(chrom.signal),
                        'color_index': color_index,
                        'text': text
                        })
        if end is None:
            end = np.max(x)

        if len(time_line.sections) == 0:
            fill_regions.append({
                'start': start,
                'end': end,
                'y_max': 1.1*np.max(chrom.signal),
                'color_index': -1,
                'text': 'W'
                })

        plot_parameters = PlotParameters()
        plot_parameters.x_label = '$time~/~min$'
        plot_parameters.y_label = '$c~/~mol \cdot L^{-1}$'
        plot_parameters.fill_regions = fill_regions
        plot_parameters.xlim = (start, end)
        plot_parameters.ylim = (0, 1.1*np.max(chrom.signal))
        plotlib.plot(x, y, plot_parameters, show=show, save_path=save_path)


    @property
    def fractionation_state(self):
        """Returns a state matrix of all fractions over time

        Returns
        -------
        fractionation_state : ndarray
            Array of fractionation state over time
        """
        if self._fractionation_state is None:
            self._fractionation_state = self.state_vector[self.chromatogram]

        return self._fractionation_state

    @property
    def fraction_pools(self):
        """Returns a list of the component and waste fraction pools.

        For every event, the end time is determined and a Fraction object is
        created which holds information about start and end time, as well as
        the mass and the volume of the fraction. The fractions are pooled
        depending on the event state.

        Returns
        -------
        fraction_pools : list
            List with fraction pools.
        """
        if self._fraction_pools is None:
            self._fraction_pools = [FractionPool(self.n_comp)
                                    for _ in range(self.n_comp + 1)]

            for chrom_index, chrom in enumerate(self.chromatograms):
                chrom_events = self.performer_event_lists[str(chrom_index)]
                for evt_index, evt in enumerate(chrom_events):
                    target = int(np.nonzero(evt.state)[0])

                    frac_start = evt.time

                    if evt_index < len(chrom_events) - 1:
                        frac_end = chrom_events[evt_index + 1].time
                        fraction = self._create_fraction(
                                chrom_index, frac_start, frac_end)
                        self.add_fraction(fraction, target)
                    else:
                        frac_end = self.cycle_time
                        fraction = self._create_fraction(
                                chrom_index, frac_start, frac_end)
                        self.add_fraction(fraction, target)

                        frac_start = 0
                        frac_end = chrom_events[0].time
                        fraction = self._create_fraction(
                                chrom_index, frac_start, frac_end)
                        self.add_fraction(fraction, target)

        return self._fraction_pools

    def _create_fraction(self, chrom_index, start, end):
        """Helper function to create Fraction object calculate mass

        Parameters
        ----------
        start : float
            start time of the fraction
        start : float
            start time of the fraction

        Returns
        -------
        fraction : Fraction
            Chromatogram fraction
        """
        mass = self.chromatograms[chrom_index].fraction_mass(start, end)
        volume = self.chromatograms[chrom_index].fraction_volume(start, end)
        return Fraction(mass, volume)

    def add_fraction(self, fraction, target):
        """
        Waste is the last fraction
        """
        if not isinstance(fraction, Fraction):
            raise TypeError('Expected Fraction')

        if target not in range(self.n_comp + 1):
            raise CADETProcessError('Not a valid target')
        self._fraction_pools[target].add_fraction(fraction)


    @property
    def mass(self):
        """ndarray: Collected component mass in corresponding fraction pools.
        """
        if self._mass is None:
            self._mass = np.array([pool.mass[comp]
                         for comp, pool in enumerate(self.fraction_pools[:-1])])
        return self._mass

    @property
    def concentration(self):
        """ndarray: Component concentration in corresponding fraction pool.
        """
        return np.array([pool.concentration[comp]
                         for comp, pool in enumerate(self.fraction_pools[:-1])])

    @property
    def purity(self):
        """ndarray: Component purity in corresponding fraction pool.
        """
        return np.array([pool.purity[comp]
                         for comp, pool in enumerate(self.fraction_pools[:-1])])

    @property
    def recovery(self):
        """ndarray: Component recovery yield in corresponding fraction pool.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            recovery = self.mass / self.process_meta.m_feed

        return np.nan_to_num(recovery)

    @property
    def productivity(self):
        """ndarray: Specific productivity for components in corresponding
        fraction pool.
        """
        return self.mass / (self.process_meta.cycle_time *
                            self.process_meta.V_solid)

    @property
    def eluent_consumption(self):
        """ndarray: Component mass per unit volume of eluent in corresponding
        fraction pool.

        Note
        ----
        This is the inverse of the regularly used specific eluent consumption
        It is preferred in order to avoid numeric issues if collected mass is 0.
        """
        return self.mass / self.process_meta.V_eluent

    @property
    def performance(self):
        self.reset()
        return Performance(self.mass, self.concentration, self.purity,
                           self.recovery, self.productivity,
                           self.eluent_consumption)

    def reset(self):
        """Resets the results when fractionation times are changed.
        """
        self._fractionation_state = None
        self._fraction_pools = None
        self._mass = None


    def initial_values(self, purity_required=0.95):
        """Create events from chromatogram with minimum purity.

        Function creates fractions for areas in the chromatogram, where the
        local purity profile is higher than the purity required.

        Parameters
        -----------
        purity_required : float or list of floats
            Minimum purity required for the components in the fractionation

        Raises
        ------
        ValueError
            If size of purity parameter does not math number of components
        """
        if isinstance(purity_required, float):
            purity_required = [purity_required]*self.n_comp
        elif len(purity_required) != self.n_comp:
            raise ValueError('Expected array with size {}'.format(
                    self.chromatogram.n_comp))

        self._events = []
        self.reset()

        for chrom_index, chrom in enumerate(self.chromatograms):
            purity_min = np.zeros(chrom.signal.shape)
            purity_min[chrom.local_purity > purity_required] = 1
            diff = np.vstack((purity_min[0,:] - purity_min[-1,:],
                              np.diff(purity_min, axis=0)))

            for comp in range(self.n_comp):
                if purity_required[comp] > 0:
                    on_indices = np.where(diff[:,comp] == 1)
                    on_indices = on_indices[0]
                    for index, on_evt in enumerate(on_indices):
                        time = chrom.time[int(on_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_start_' + str(index)
                        self.add_event(event_name,
                                       str(chrom_index), comp, time)

                    off_indices = np.where(diff[:,comp] == -1)
                    off_indices = off_indices[0]
                    for index, off_evt in enumerate(off_indices):
                        time = chrom.time[int(off_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_end_' + str(index)
                        self.add_event(event_name,
                                       str(chrom_index), self.n_comp, time)

    @property
    def parameters(self):
        parameters = super().parameters
        for index, chrom in enumerate(self.chromatograms):
            parameters[str(index)] = chrom.fractionation_state

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        for chrom_index, chrom in enumerate(self.chromatograms):
            try:
                frac_state = parameters.pop(str(chrom_index))
                self.chromatograms[chrom_index].fractionation_state = frac_state
            except KeyError:
                pass

        super(Fractionator, self.__class__).parameters.fset(self, parameters)

    def save(self, case_dir, start=0, end=None):
        path = os.path.join(settings.project_directory, case_dir)

        for index, chrom in enumerate(self.chromatograms):
            chrom.plot(save_path=path + '/chrom_{}.png'.format(index))
            chrom.plot_purity(start=start, end=end,
                              save_path=path + '/chrom_purity.png')

        for index, chrom in enumerate(self.chromatograms):
            self.plot_fraction_signal(
                    start=start, end=end,
                    save_path=path + '/fractionation_signal_{}.png'.format(index),
                    index=index)


import warnings

from CADETProcess.optimization import OptimizationProblem, COBYLA, TrustConstr
from CADETProcess.optimization import mass, ranked_objective_decorator
from CADETProcess.optimization import purity, nonlin_bounds_decorator

def optimize_fractionation(chromatograms, process_meta, purity_required,
                           obj_fun=None, return_results=False):
    """Optimizing the fraction times by variation of the fractionation events.

    Function creates a fractionation object and creates initial values with
    purity_required. An OptimizationProblem is instantiated and the
    fractionation events are set as optimization variables. The order of the
    fractionation times is enforced by linear constraints. If no objective
    function is provided, the mass of all components is maximized. COBYLA is
    used to solve the optimization problem.

    Parameters
    ----------
    chromatograms : Chromatogram or list of Chromatograms
        Chromatogram to be fractionated
    process_meta : ProcessMeta

    purity_required :  float or array_like
        Minimum required purity for components. If float, same value is assumed
        for all components.
    obj_fun : function, optional
        Objective function used for OptimizationProblem. If None, the mass of
        all components is maximized.
    return_results : Bool
        If True, also returns optimization results. For Debugging.

    Raises
    -------
    TypeError
        If chromatogram is not an instance of Chromatogram.
    Warning
        If purity requirements cannot be fulfilled.

    Returns
    -------
    performance : Performance
        FractionationPerformance
    solver : OptimizationSolver
        If return_solver is True

    See also
    --------
    Chromatogram
    Fractionation
    COBYLA
    """
    frac = Fractionator(process_meta)

    if not isinstance(chromatograms, list):
        chromatograms = [chromatograms]
    for chrom in chromatograms:
        frac.add_chromatogram(chrom)

    frac.initial_values(purity_required)

    if len(frac.events) == 0:
        warnings.warn("No areas found with sufficient purity. Returning")
        return frac

    opt = OptimizationProblem(frac)
    opt.logger.setLevel(logging.WARNING)

    if obj_fun is None:
        obj_fun = ranked_objective_decorator(1)(mass)
    opt.objective_fun = obj_fun

    if isinstance(purity_required, float):
        purity_required = [purity_required] * frac.n_comp
    opt.nonlinear_constraint_fun = \
        nonlin_bounds_decorator(purity_required)(purity)

    for evt in frac.events:
        opt.add_variable(evt.name + '.time', evt.name)

    for chrom_index, chrom in enumerate(frac.chromatograms):
        chrom_events = frac.performer_event_lists[str(chrom_index)]
        evt_names = [evt.name for evt in chrom_events]
        for evt_index, evt in enumerate(chrom_events):
            if evt_index < len(chrom_events) - 1:
                opt.add_linear_constraint(
                    [evt_names[evt_index], evt_names[evt_index+1]], [1,-1]
                    )
            else:
                opt.add_linear_constraint(
                    [evt_names[0], evt_names[-1]],[-1,1], frac.cycle_time)

    opt.x0 = [evt.time for evt in frac.events]

    if not opt.check_nonlinear_constraints(opt.x0):
        warnings.warn("No areas found with sufficient purity. Returning")
        return frac

    solver = COBYLA()
    solver.rhobeg = 1.0
    try:
        opt_results = solver.optimize(opt)
    except CADETProcessError:
        try:
            solver.rhobeg = 0.01
            opt_results = solver.optimize(opt)
            opt.logger.info('Optimization failed, re-trying with smaller rho')
        except CADETProcessError:
            frac.initial_values()
            warnings.warn('Optimization failed. Returning initial values')
            opt_results = None

    if return_results:
        return frac, opt_results, solver, opt

    return frac
