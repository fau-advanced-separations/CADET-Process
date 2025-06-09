import os
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
from addict import Dict
from matplotlib.axes import Axes

from CADETProcess import CADETProcessError, SimulationResults, plotting, settings
from CADETProcess.dataStructure import String
from CADETProcess.dynamicEvents import Event, EventHandler
from CADETProcess.fractionation.fractions import Fraction, FractionPool
from CADETProcess.performance import Performance
from CADETProcess.processModel import ComponentSystem, Process
from CADETProcess.solution import SolutionIO, slice_solution

__all__ = ["Fractionator"]


class Fractionator(EventHandler):
    """
    Class for Chromatogram Fractionation.

    This class is responsible for setting events for starting and ending fractionation,
    handling multiple chromatograms, and calculating various performance metrics.

    Attributes
    ----------
    name : String
        Name of the fractionator, defaulting to 'Fractionator'.
    performance_keys : list
        Keys for performance metrics including mass, concentration, purity, recovery,
        productivity, and eluent consumption.
    """

    name = String(default="Fractionator")
    performance_keys: list[str] = [
        "mass",
        "concentration",
        "purity",
        "recovery",
        "productivity",
        "eluent_consumption",
    ]

    def __init__(
        self,
        simulation_results: SimulationResults,
        components: Optional[list[str]] = None,
        use_total_concentration_components: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Fractionator.

        Parameters
        ----------
        simulation_results : SimulationResults
            Simulation results containing chromatograms.
        components : list, optional
            List of components to be fractionated. Default is None.
        use_total_concentration_components : bool, optional
            Use total concentration components. Default is True.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.components: Optional[list[str]] = components
        self.use_total_concentration_components: bool = use_total_concentration_components
        self.simulation_results = simulation_results

        super().__init__(*args, **kwargs)

    @property
    def simulation_results(self) -> SimulationResults:
        """SimulationResults: The simulation results containing the chromatograms."""
        return self._simulation_results

    @simulation_results.setter
    def simulation_results(self, simulation_results: SimulationResults) -> None:
        """
        Set the simulation results.

        Parameters
        ----------
        simulation_results : SimulationResults
            Simulation results containing chromatograms.

        Raises
        ------
        TypeError
            If simulation_results is not of type SimulationResults.
        CADETProcessError
            If the simulation results do not contain any chromatograms.
        """
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError("Expected SimulationResults")

        if len(simulation_results.chromatograms) == 0:
            raise CADETProcessError("Simulation results do not contain chromatogram")

        self._simulation_results = simulation_results

        self._chromatograms = [
            slice_solution(
                chrom,
                components=self.components,
                use_total_concentration_components=self.use_total_concentration_components,
            )
            for chrom in simulation_results.chromatograms
        ]

        m_feed = np.zeros((self.component_system.n_comp,))
        counter = 0
        for comp, indices in simulation_results.component_system.indices.items():
            if comp in self.component_system.names:
                m_feed_comp = simulation_results.process.m_feed[indices]
                if self.use_total_concentration_components:
                    m_feed[counter] = np.sum(m_feed_comp)
                    counter += 1
                else:
                    n_species = len(indices)
                    m_feed[counter : counter + n_species] = m_feed_comp
                    counter += n_species
        self.m_feed: np.ndarray = m_feed

        self._fractionation_states = Dict({chrom: [] for chrom in self.chromatograms})
        self._chromatogram_events = Dict({chrom: [] for chrom in self.chromatograms})

        self._cycle_time = self.process.cycle_time

        self.reset()

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: The component system of the chromatograms."""
        return self.chromatograms[0].component_system

        """Enable calling functions with chromatogram object or name."""

    def _call_by_chrom_name(func: Callable) -> Callable:
        @wraps(func)
        def wrapper_call_by_chrom_name(
            self: "Fractionator", chrom: str | SolutionIO, *args: Any, **kwargs: Any
        ) -> Any:
            """Enable calling functions with chromatogram object or name."""
            if isinstance(chrom, str):
                try:
                    chrom = self.chromatograms_dict[chrom]
                except KeyError:
                    raise CADETProcessError("Not a valid unit")
            return func(self, chrom, *args, **kwargs)

        return wrapper_call_by_chrom_name

    @property
    def chromatograms(self) -> list[SolutionIO]:
        """
        list[SolutionIO]: Chromatograms to be fractionized.

        See Also
        --------
        SolutionIO
        reset
        cycle_time
        """
        return self._chromatograms

    @property
    def chromatograms_dict(self) -> dict[str, SolutionIO]:
        """dict: Chromatogram names and objects."""
        return {chrom.name: chrom for chrom in self.chromatograms}

    @property
    def chromatogram_names(self) -> list[str]:
        """list[str]: Names of chromatogram."""
        return [chrom.name for chrom in self.chromatograms]

    @property
    def n_chromatograms(self) -> int:
        """int: Number of Chromatograms Fractionator."""
        return len(self.chromatograms)

    @property
    def chromatogram_events(self) -> dict[SolutionIO, list[Event]]:
        """dict[SolutionIO, list[Event]]: Events sorted by chromatogram."""
        chrom_events = {
            chrom: sorted(events, key=lambda evt: evt.time)
            for chrom, events in self._chromatogram_events.items()
        }

        return chrom_events

    @property
    def process(self) -> Process:
        """Process: The process from the simulation results."""
        return self.simulation_results.process

    @property
    def n_comp(self) -> int:
        """int: Number of components to be fractionized."""
        return self.chromatograms[0].n_comp

    @property
    def cycle_time(self) -> float:
        """
        The cycle time of the Fractionator.

        Note that in some situations, it might be desired to set a custom cycle time
        for calculating the performance indicators. For this purpose, overwrite the
        cycle time in the Process object after adding it to the Fractionator.

        Warning: This is not a robust feature! Side effects can ocurr in the Process!

        See Also
        --------
        productivity

        """
        return self._cycle_time

    @property
    def time(self) -> np.ndarray:
        """np.ndarray: solution times of Chromatogram."""
        return self.chromatograms[0].time

    @plotting.create_and_save_figure
    def plot_fraction_signal(
        self,
        chromatogram: Optional[SolutionIO | str] = None,
        x_axis_in_minutes: bool = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the signal without the waste fractions.

        Parameters
        ----------
        chromatogram : Optional[SolutionIO | str]
            Chromatogram to be plotted. If None, the first one is plotted.
        ax : Axes, optional
            Axes to plot on. If None, a new figure is created.
        x_axis_in_minutes: bool, optional
            Option to use x-aches (time) in minutes, default is set to True.
        *args : Any
            Optional Parameter passed down to plot function.
        **kwargs : Any
            Additional Parameter passed down to plot function.

        Returns
        -------
        ax : Axes
            Axes with plot of parameter state.

        See Also
        --------
        CADETProcess.plot
        plot_purity
        """
        if chromatogram is None:
            chromatogram = list(
                self.performer_timelines["fractionation_states"].keys()
            )[0]
        if isinstance(chromatogram, str):
            chromatogram = self.chromatograms_dict[chromatogram]

        time_line = self.performer_timelines["fractionation_states"][chromatogram.name]

        try:
            start: float = kwargs["start"]
            if x_axis_in_minutes:
                start = start / 60
        except KeyError:
            start = 0
        try:
            end: float = kwargs["end"]
            if x_axis_in_minutes:
                end = end / 60
        except KeyError:
            end = np.max(chromatogram.time)

        _, ax = chromatogram.plot(
            show=False, ax=ax, x_axis_in_minutes=x_axis_in_minutes, *args, **kwargs
        )

        y_max = 1.1 * np.max(chromatogram.solution)

        fill_regions = []
        for sec in time_line.sections:
            comp_index = int(np.where(sec.coeffs)[0].squeeze())
            if comp_index == self.n_comp:
                color_index = -1
                text = "W"
            else:
                color_index = comp_index
                text = self.component_system.names[comp_index]

            sec_start = sec.start
            sec_end = sec.end

            if x_axis_in_minutes:
                sec_start = sec_start / 60
                sec_end = sec_end / 60

            if sec_start != sec_end:
                fill_regions.append(
                    plotting.FillRegion(
                        start=sec_start,
                        end=sec_end,
                        y_max=y_max,
                        color_index=color_index,
                        text=text,
                    )
                )

        if len(time_line.sections) == 0:
            fill_regions.append(
                plotting.FillRegion(
                    start=sec_start, end=sec_end, y_max=y_max, color_index=-1, text="W"
                )
            )

        plotting.add_fill_regions(ax, fill_regions, (start, end))

        return ax

    @property
    def fractionation_states(self) -> Dict:
        """dict: Fractionation state of Chromatograms.

        Notes
        -----
        This is just a dummy variable to support interfacing with Events.

        """
        return self._fractionation_states

    @_call_by_chrom_name
    def set_fractionation_state(
        self, chrom: SolutionIO, state: int | list[float]
    ) -> None:
        """
        Set fractionation states of Chromatogram.

        Parameters
        ----------
        chrom : SolutionIO
            Chromatogram object which is to be fractionated.
        state : int or list of floats
            New fractionation state of the Chromatogram.

        Raises
        ------
        CADETProcessError
            If Chromatogram not in Fractionator
            If state is integer and the state >= the n_comp.
            If the length of the states is unequal the state_length.
            If the sum of the states is not equal to 1.
        """
        if chrom not in self.chromatograms:
            raise CADETProcessError("Chromatogram not in Fractionator")

        state_length = self.n_comp + 1

        if state_length == 0:
            fractionation_state = []

        if type(state) is int:
            if state >= state_length:
                raise CADETProcessError("Index exceeds fractionation states")

            fractionation_state = [0] * state_length
            fractionation_state[state] = 1
        else:
            if len(state) != state_length:
                raise CADETProcessError(f"Expected length {state_length}.")

            elif sum(state) != 1:
                raise CADETProcessError("Sum of fractions must be 1")

            fractionation_state = state

        self._fractionation_states[chrom] = fractionation_state

    @property
    def n_fractions_per_pool(self) -> list[int]:
        """list: number of fractions per pool."""
        return [pool.n_fractions for pool in self.fraction_pools]

    @property
    def fraction_pools(self) -> list[FractionPool]:
        """
        List of the component and waste fraction pools.

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
            self._fraction_pools = [
                FractionPool(self.n_comp) for _ in range(self.n_comp + 1)
            ]

            for chrom_index, chrom in enumerate(self.chromatograms):
                chrom_events = self.chromatogram_events[chrom]
                for evt_index, evt in enumerate(chrom_events):
                    target = np.nonzero(evt.full_state)[0].squeeze()

                    frac_start = evt.time

                    if evt_index < len(chrom_events) - 1:
                        frac_end = chrom_events[evt_index + 1].time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)
                    else:
                        frac_end = self.cycle_time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)

                        frac_start = 0
                        frac_end = chrom_events[0].time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)

        return self._fraction_pools

    def _create_fraction(self, chrom_index: int, start: float, end: float) -> Fraction:
        """
        Create Fraction object calculate mass.

        Parameters
        ----------
        chrom_index : int
            index of the chromatogram
        start : float
            start time of the fraction
        end : float
            end time of the fraction

        Returns
        -------
        fraction : Fraction
            Chromatogram fraction
        """
        fraction = self.chromatograms[chrom_index].create_fraction(start, end)

        return fraction

    def add_fraction(self, fraction: Fraction, target: int) -> None:
        """
        Add Fraction to the FractionPool of target component.

        Notes
        -----
        Waste is always the last fraction
        """
        if not isinstance(fraction, Fraction):
            raise TypeError("Expected Fraction")

        if target not in range(self.n_comp + 1):
            raise CADETProcessError("Not a valid target")
        self._fraction_pools[target].add_fraction(fraction)

    @property
    def mass(self) -> np.ndarray:
        """ndarray: Component mass in corresponding fraction pools."""
        if self._mass is None:
            self._mass = np.array([
                pool.mass[comp]
                for comp, pool in enumerate(self.fraction_pools[:-1])
            ])
        return self._mass

    @property
    def total_mass(self) -> np.ndarray:
        """ndarray: Total mass of each component in all fraction pools."""
        return np.sum([pool.mass for pool in self.fraction_pools], axis=0)

    @property
    def concentration(self) -> np.ndarray:
        """ndarray: Component concentration in corresponding fraction pool."""
        return np.array([
            pool.concentration[comp]
            for comp, pool in enumerate(self.fraction_pools[:-1])
        ])

    @property
    def purity(self) -> np.ndarray:
        """ndarray: Component purity in corresponding fraction pool."""
        return np.array([
            pool.purity[comp]
            for comp, pool in enumerate(self.fraction_pools[:-1])
        ])

    @property
    def recovery(self) -> np.ndarray:
        """ndarray: Component recovery yield in corresponding fraction pool."""
        with np.errstate(divide="ignore", invalid="ignore"):
            recovery = self.mass / self.m_feed

        return np.nan_to_num(recovery)

    @property
    def mass_balance_difference(self) -> np.ndarray:
        """
        ndarray: Difference in mass balance between m_feed and fraction pools.

        The mass balance is calculated as the difference between the feed mass (m_feed)
        and the mass in the fraction pools. It represents the discrepancy or change in
        mass during the fractionation process.

        Returns
        -------
        ndarray
            Difference in mass balance between m_feed and fraction pools for each
            component.

        Notes
        -----
        Positive values indicate a surplus of mass in the fraction pools compared to
        the feed, while negative values indicate a deficit. A value of zero indicates
        a mass balance where the mass in the fraction pools is equal to the feed mass.

        """
        return self.total_mass - self.m_feed

    @property
    def productivity(self) -> np.ndarray:
        """ndarray: Specific productivity in corresponding fraction pool."""
        return self.mass / (self.process.cycle_time * self.process.V_solid)

    @property
    def eluent_consumption(self) -> np.ndarray:
        """
        ndarray: Specific eluent consumption in corresponding fraction pool.

        Notes
        -----
        This is the inverse of the regularly used specific eluent
        consumption. It is preferred here in order to avoid numeric issues
        if the collected mass is 0.
        """
        return self.mass / self.process.V_eluent

    @property
    def performance(self) -> Performance:
        """Performance: The performance metrics of the fractionation."""
        self.reset()
        return Performance(
            self.mass,
            self.concentration,
            self.purity,
            self.recovery,
            self.productivity,
            self.eluent_consumption,
            self.mass_balance_difference,
            self.component_system,
        )

    def reset(self) -> None:
        """Reset the results when fractionation times are changed."""
        self._fractionation_state = None
        self._fraction_pools = None
        self._mass = None

    def add_fractionation_event(
        self,
        event_name: str,
        target: str | int,
        time: float,
        chromatogram: Optional[SolutionIO | str] = None,
    ) -> None:
        """
        Add a fractionation event.

        Parameters
        ----------
        event_name : str
            The name of the event.
        target : str | int
            The indice or name of target component in Component System.
        time : float
            The time of the event.
        chromatogram : Optional[SolutionIO | str]
            The chromatogram associated with the event.
            If None and there is only one chromatogram, it will be used.

        Raises
        ------
        CADETProcessError
            If the chromatogram is not found.
        """
        if chromatogram is None and self.n_chromatograms > 1:
            raise CADETProcessError(
                "Missing chromatogram for which the fractionation is added."
            )
        elif chromatogram is None and self.n_chromatograms == 1:
            chromatogram = self.chromatograms[0]

        if isinstance(chromatogram, str):
            try:
                chromatogram = self.chromatograms_dict[f"{chromatogram}"]
            except KeyError:
                raise CADETProcessError("Could not find chromatogram.")

        if chromatogram not in self.chromatograms:
            raise CADETProcessError("Could not find chromatogram.")

        param_path = f"fractionation_states.{chromatogram.name}"

        if isinstance(target, str):
            target = self.component_system.names.index(target)

        evt = self.add_event(event_name, param_path, target, time)
        self._chromatogram_events[chromatogram].append(evt)

        self.reset()

    def initial_values(self, purity_required: float | list[float] = 0.95) -> None:
        """
        Create events from chromatogram with minimum purity.

        Function creates fractions for areas in the chromatogram, where the
        local purity profile is higher than the purity required.

        Parameters
        ----------
        purity_required : float or list of floats
            Minimum purity required for the components in the fractionation

        Raises
        ------
        ValueError
            If the size of the purity parameter does not match the number of components
        """
        if isinstance(purity_required, float):
            purity_required = [purity_required] * self.n_comp
        elif len(purity_required) != self.n_comp:
            raise ValueError(f"Expected purity array with size {self.n_comp}")

        self._events = []
        self._chromatogram_events = Dict({chrom: [] for chrom in self.chromatograms})
        self.reset()

        for chrom_index, chrom in enumerate(self.chromatograms):
            purity_min = np.zeros(chrom.solution.shape)
            purity_min[chrom.local_purity_components > purity_required] = 1
            diff = np.vstack((
                purity_min[0, :] - purity_min[-1, :],
                np.diff(purity_min, axis=0)
            ))

            for comp in range(self.n_comp):
                if purity_required[comp] > 0:
                    on_indices = np.where(diff[:, comp] == 1)[0]
                    off_indices = np.where(diff[:, comp] == -1)[0] - 1

                    # Handle the case where the entire array is above the threshold
                    if (
                        len(on_indices) == 0
                        and len(off_indices) == 0
                        and purity_min[0, comp] == 1
                    ):
                        on_indices = np.array([0])
                        off_indices = np.array([purity_min.shape[0] - 1])

                    # Ensure regions with a single value are ignored
                    valid_regions = [
                        (on, off)
                        for on, off in zip(on_indices, off_indices)
                        if off != on
                    ]

                    for index, (on_evt, off_evt) in enumerate(valid_regions):
                        # Add start event
                        time = chrom.time[int(on_evt)]
                        event_name = f"chrom_{chrom_index}_comp_{comp}_start_{index}"
                        param_path = f"fractionation_states.{chrom.name}"
                        evt = self.add_event(event_name, param_path, comp, time)
                        self._chromatogram_events[chrom].append(evt)

                        # Add end event
                        time = chrom.time[int(off_evt)]
                        event_name = f"chrom_{chrom_index}_comp_{comp}_end_{index}"
                        param_path = f"fractionation_states.{chrom.name}"
                        evt = self.add_event(event_name, param_path, self.n_comp, time)
                        self._chromatogram_events[chrom].append(evt)

        if not self.check_duplicate_events():
            chrom_events = self._chromatogram_events.copy()
            for chrom, events in chrom_events.items():
                events_at_time = defaultdict(list)
                for event in events:
                    events_at_time[event.time].append(event)

                for events in events_at_time.values():
                    if len(events) == 1:
                        continue
                    for evt in events:
                        if evt.state == self.n_comp:
                            self.remove_event(evt.name)
                            self._chromatogram_events[chrom].remove(evt)

    @property
    def parameters(self) -> Dict:
        """dict: Parameters of the fractionator."""
        parameters = super().parameters
        parameters["fractionation_states"] = {
            chrom.name: self.fractionation_states[chrom] for chrom in self.chromatograms
        }

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters: Dict) -> None:
        try:
            fractionation_states = parameters.pop("fractionation_states")
            for chrom, state in fractionation_states.items():
                self.set_fractionation_state(chrom, state)
        except KeyError:
            pass

        super(Fractionator, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self) -> Dict:
        """dict: Section dependent parameters of the fractionator."""
        return self.parameters

    def save(
        self,
        case_dir: str,
        start: float = 0,
        end: Optional[float] = None,
    ) -> None:
        """
        Save chromatogram and purity plots to a specified directory.

        Parameters
        ----------
        case_dir : str
            Directory name within the working directory to save plots.
        start : float, optional
            Start time for plotting purity, default is 0.
        end : Optional[float]
            End time for plotting purity. If None, includes all data.
        """
        path = os.path.join(settings.working_directory, case_dir)

        for index, chrom in enumerate(self.chromatograms):
            chrom.plot(save_path=path + f"/chrom_{index}.png")
            chrom.plot_purity(
                start=start, end=end, save_path=path + "/chrom_purity.png"
            )

        for chrom in enumerate(self.chromatograms):
            self.plot_fraction_signal(
                chromatogram=chrom,
                start=start,
                end=end,
                save_path=path + f"/fractionation_signal_{index}.png",
                index=index,
            )

    def __str__(self) -> str:
        """str: String representation of the fractionator."""
        return self.__class__.__name__
