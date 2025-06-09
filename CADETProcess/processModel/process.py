from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional
from warnings import warn

import numpy as np
from addict import Dict
from scipy import integrate, interpolate

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import cached_property_if_locked
from CADETProcess.dynamicEvents import EventHandler, Section, TimeLine

from .componentSystem import ComponentSystem
from .flowSheet import FlowSheet
from .unitOperation import Inlet, Outlet


class Process(EventHandler):
    """
    Class for defining the dynamic changes of a flow sheet.

    Attributes
    ----------
    name : str
        Name of the process object to be simulated.
    system_state : np.ndarray
        State of the process object
    system_state_derivate : np.ndarray
        Derivative of the state

    See Also
    --------
    EventHandler
    CADETProcess.processModel.FlowSheet
    CADETProcess.simulation.Solver
    """

    _initial_states = ["system_state", "system_state_derivative"]

    def __init__(
        self,
        flow_sheet: FlowSheet,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize Process."""
        self.flow_sheet = flow_sheet
        self.name = name

        self.system_state = None
        self.system_state_derivative = None

        self._parameter_sensitivities = []

        self._meta_information = Dict()

        super().__init__(*args, **kwargs)

    @property
    def n_comp(self) -> int:
        """int: Number of components in the process."""
        return self.flow_sheet.n_comp

    @property
    def meta_information(self) -> dict:
        """dict: Meta information of the process."""
        # TODO: DO we still use this anywhere?
        return self._meta_information

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: Component system of the process."""
        return self.flow_sheet.component_system

    @property
    def flow_sheet(self) -> FlowSheet:
        """FlowSheet: flow sheet of the process model.

        Raises
        ------
        TypeError:
            If flow_sheet is not an instance of FlowSheet.

        """
        return self._flow_sheet

    @flow_sheet.setter
    def flow_sheet(self, flow_sheet: FlowSheet) -> None:
        if not isinstance(flow_sheet, FlowSheet):
            raise TypeError("Expected FlowSheet")
        self._flow_sheet = flow_sheet

    @cached_property_if_locked
    def m_feed(self) -> np.ndarray:
        """np.ndarray: Mass of feed components entering the system in one cycle."""
        flow_rate_timelines = self.flow_rate_timelines

        feed_all = np.zeros((self.n_comp,))
        for feed in self.flow_sheet.feed_inlets:
            feed_flow_rate_time_line = flow_rate_timelines[feed.name].total_out[None]
            feed_signal_param = f"flow_sheet.{feed.name}.c"
            if feed_signal_param in self.parameter_timelines:
                tl = self.parameter_timelines[feed_signal_param]
                feed_signal_time_line = tl
            else:
                feed_signal_time_line = TimeLine()
                feed_section = Section(0, self.cycle_time, feed.c, is_polynomial=True)
                feed_signal_time_line.add_section(feed_section)

            m_i = [
                integrate.quad(
                    lambda t: feed_flow_rate_time_line.value(t) * feed_signal_time_line.value(t)[comp],  # noqa: E501
                    0,
                    self.cycle_time,
                    points=self.event_times,
                )[0]
                for comp in range(self.n_comp)
            ]

            feed_all += np.array(m_i)

        return feed_all

    @cached_property_if_locked
    def V_eluent(self) -> float:
        """float: Volume of the eluent entering the system in one cycle."""
        flow_rate_timelines = self.flow_rate_timelines

        V_all = 0
        for eluent in self.flow_sheet.eluent_inlets:
            eluent_time_line = flow_rate_timelines[eluent.name]["total_out"][None]
            V_eluent = eluent_time_line.integral().squeeze()
            V_all += V_eluent

        return float(V_all)

    @cached_property_if_locked
    def V_solid(self) -> float:
        """float: Volume of all solid phase material used in flow sheet."""
        return sum([unit.volume_solid for unit in self.flow_sheet.units_with_binding])

    @cached_property_if_locked
    def flow_rate_timelines(self) -> dict:
        """Return TimeLine of flow_rate for all unit_operations."""
        flow_rate_timelines = {
            unit.name: {
                "total_in": defaultdict(TimeLine),
                "origins": defaultdict(
                    lambda: defaultdict(lambda: defaultdict(TimeLine))
                ),
                "total_out": defaultdict(TimeLine),
                "destinations": defaultdict(
                    lambda: defaultdict(lambda: defaultdict(TimeLine))
                ),
            }
            for unit in self.flow_sheet.units
        }

        # Create dummy section state for Processes without events
        if len(self.section_states) == 0:
            it = [(None, {})]
        else:
            it = self.section_states.items()

        for i, (time, state) in enumerate(it):
            start = self.section_times[i]
            end = self.section_times[i + 1]

            flow_rates = self.flow_sheet.get_flow_rates(state)

            for unit, flow_rate_dict in flow_rates.items():
                unit_flow_rates = flow_rate_timelines[unit]

                # If inlet, also use outlet for total_in

                if isinstance(self.flow_sheet[unit], Inlet):
                    for port in flow_rate_dict["total_out"]:
                        section = Section(
                            start,
                            end,
                            flow_rate_dict.total_out[port],
                            is_polynomial=True,
                        )
                        unit_flow_rates["total_in"][port].add_section(section)
                else:
                    for port in flow_rate_dict["total_in"]:
                        section = Section(
                            start,
                            end,
                            flow_rate_dict.total_in[port],
                            is_polynomial=True,
                        )
                        unit_flow_rates["total_in"][port].add_section(section)

                for port in flow_rate_dict.origins:
                    for orig, origin_port_dict in flow_rate_dict.origins[port].items():
                        for orig_port, flow_rate_orig in origin_port_dict.items():
                            section = Section(
                                start, end, flow_rate_orig, is_polynomial=True
                            )
                            unit_flow_rates["origins"][port][orig][orig_port].add_section(section)

                # If outlet, also use inlet for total_out
                if isinstance(self.flow_sheet[unit], Outlet):
                    for port in flow_rate_dict["total_in"]:
                        section = Section(
                            start,
                            end,
                            flow_rate_dict.total_in[port],
                            is_polynomial=True,
                        )
                        unit_flow_rates["total_out"][port].add_section(section)
                else:
                    for port in flow_rate_dict["total_out"]:
                        section = Section(
                            start,
                            end,
                            flow_rate_dict.total_out[port],
                            is_polynomial=True,
                        )
                        unit_flow_rates["total_out"][port].add_section(section)

                for port in flow_rate_dict.destinations:
                    for dest, dest_port_dict in flow_rate_dict.destinations[port].items():
                        for dest_port, flow_rate_dest in dest_port_dict.items():
                            section = Section(
                                start,
                                end,
                                flow_rate_dest,
                                is_polynomial=True,
                            )
                            unit_flow_rates["destinations"][port][dest][dest_port].add_section(section)

        return Dict(flow_rate_timelines)

    @cached_property_if_locked
    def flow_rate_section_states(self) -> dict:
        """Return flow rates for all units for every section time."""
        section_states = {
            time: {
                unit.name: {
                    "total_in": defaultdict(list),
                    "origins": defaultdict(
                        lambda: defaultdict(lambda: defaultdict(list))
                    ),
                    "total_out": defaultdict(list),
                    "destinations": defaultdict(
                        lambda: defaultdict(lambda: defaultdict(list))
                    ),
                }
                for unit in self.flow_sheet.units
            }
            for time in self.section_times[0:-1]
        }

        for sec_time in self.section_times[0:-1]:
            for unit, unit_flow_rates in self.flow_rate_timelines.items():
                if isinstance(self.flow_sheet[unit], Inlet):
                    for port in unit_flow_rates["total_out"]:
                        section_states[sec_time][unit]["total_in"][port] = \
                        unit_flow_rates["total_out"][port].coefficients(sec_time)
                else:
                    for port in unit_flow_rates["total_in"]:
                        section_states[sec_time][unit]["total_in"][port] = \
                            unit_flow_rates["total_in"][port].coefficients(sec_time)

                    for port, orig_dict in unit_flow_rates.origins.items():
                        for origin in orig_dict:
                            for origin_port, tl in orig_dict[origin].items():
                                section_states[sec_time][unit]["origins"][port][origin][origin_port] = \
                                    tl.coefficients(sec_time)  # noqa: E501

                if isinstance(self.flow_sheet[unit], Outlet):
                    for port in unit_flow_rates["total_in"]:
                        section_states[sec_time][unit]["total_out"][port] = \
                            unit_flow_rates["total_in"][port].coefficients(sec_time)
                else:
                    for port in unit_flow_rates["total_out"]:
                        section_states[sec_time][unit]["total_out"][port] = \
                            unit_flow_rates["total_out"][port].coefficients(sec_time)

                    for port, dest_dict in unit_flow_rates.destinations.items():
                        for dest in dest_dict:
                            for dest_port, tl in dest_dict[dest].items():
                                section_states[sec_time][unit]["destinations"][port][dest][dest_port] = \
                                    tl.coefficients(sec_time)  # noqa: E501

        return Dict(section_states)

    @property
    def n_sensitivities(self) -> int:
        """int: Number of parameter sensitivities."""
        return len(self.parameter_sensitivities)

    @property
    def parameter_sensitivities(self) -> list:
        """list: Parameter sensitivites."""
        return self._parameter_sensitivities

    @property
    def parameter_sensitivity_names(self) -> list:
        """list: Parameter sensitivity names."""
        return [sens.name for sens in self.parameter_sensitivities]

    def add_parameter_sensitivity(
        self,
        parameter_paths: str | list[str],
        name: Optional[str] = None,
        components: Optional[str | list[str]] = None,
        polynomial_coefficients: Optional[str | list[str]] = None,
        reaction_indices: Optional[int | list[int]] = None,
        bound_state_indices: Optional[int | list[int]] = None,
        section_indices: Optional[int | list[int]] = None,
        abstols: Optional[float | list[float]] = None,
        factors: Optional[int | list[int]] = None,
    ) -> None:
        """
        Add parameter sensitivty to Process.

        Parameters
        ----------
        parameter_paths : str or list of str
            The path to the parameter(s).
        name : str, optional
            The name of the parameter sensitivity.
            If not provided, the name of the first parameter will be used.
        components : str or list of str, optional
            The component(s) to which the parameter(s) belong.
            Must only be provided if parameter is specific to a certain compoment.
        polynomial_coefficients: str or list of str, optional
            The polynomial coefficients(s) to which the parameter(s) belong.
            Must only be provided if parameter is specific to a certain coefficient.
        reaction_indices : int or list of int, optional
            The index(es) of the reaction(s) in the associated model(s), if applicable.
            Must only be provided if parameter is specific to a certain reaction.
        bound_state_indices : int or list of int, optional
            The index(es) of the bound state(s) in the associated model(s), if applicable.
            Must only be provided if parameter is specific to a certain bound state.
        section_indices : int or list of int, optional
            The index(es) of the section(s) in the associated model(s), if applicable.
            Must only be provided if parameter is specific to a certain section.
        abstols : float or list of float, optional
            The absolute tolerances for each parameter.
            If not provided, a default tolerance will be used.
        factors : float or list of float, optional
            The factors for each parameter.
            If not provided, a default factor of 1 will be used.

        Raises
        ------
        CADETProcessError
            Number of indices do not match for:
            - components
            - polynomial_coefficients
            - reaction
            - bound_state
            - sections
            - tolerances
            - factors

            Component is not found.
            Unit is not found.
            Parameter is not found.
            Name is not provided (if number of parameters larger than 1).
            If sensitivity name already exists.

        Notes
        -----
        This functionality is still work in progress.

        .. todo::
            - [ ] Check if compoment/reaction/polynomial index are required.
            - [ ] Specify time instead of section index;
        """
        if not isinstance(parameter_paths, list):
            parameter_paths = [parameter_paths]
        n_params = len(parameter_paths)

        if name is None:
            if n_params > 1:
                raise CADETProcessError(
                    "Must provide sensitivity name if n_params > 1."
                )
            else:
                name = parameter_paths[0]

        if name in self.parameter_sensitivity_names:
            raise CADETProcessError(
                "Parameter sensitivity with same name already exists."
            )

        if components is None:
            components = n_params * [None]
        if not isinstance(components, list):
            components = [components]
        if len(components) != n_params:
            raise CADETProcessError("Number of component indices does not match.")
        if components is None:
            components = n_params * [None]

        if not isinstance(polynomial_coefficients, list):
            polynomial_coefficients = [polynomial_coefficients]
        if len(polynomial_coefficients) != n_params:
            raise CADETProcessError("Number of coefficient indices does not match.")

        if reaction_indices is None:
            reaction_indices = n_params * [None]
        if not isinstance(reaction_indices, list):
            reaction_indices = [reaction_indices]
        if len(reaction_indices) != n_params:
            raise CADETProcessError("Number of reaction indices does not match.")

        if bound_state_indices is None:
            bound_state_indices = n_params * [None]
        if not isinstance(bound_state_indices, list):
            bound_state_indices = [bound_state_indices]
        if len(bound_state_indices) != n_params:
            raise CADETProcessError("Number of bound_state indices does not match.")

        if section_indices is None:
            section_indices = n_params * [None]
        if not isinstance(section_indices, list):
            section_indices = [section_indices]
        if len(section_indices) != n_params:
            raise CADETProcessError("Number of section indices does not match.")

        if abstols is None:
            abstols = n_params * [None]
        if not isinstance(abstols, list):
            abstols = [abstols]
        if len(abstols) != n_params:
            raise CADETProcessError("Number of abstol entries does not match.")

        if factors is None:
            factors = n_params * [1]
        if not isinstance(factors, list):
            factors = [factors]
        if len(factors) != n_params:
            raise CADETProcessError("Number of factor entries does not match.")

        units = []
        associated_models = []
        parameters = []
        for param, comp, coeff, reac, state, section, tol, fac in zip(
            parameter_paths,
            components,
            polynomial_coefficients,
            reaction_indices,
            bound_state_indices,
            section_indices,
            abstols,
            factors,
        ):
            param_parts = param.split(".")
            unit = param_parts[0]
            parameter = param_parts[-1]
            if parameter == "flow_rate":
                raise CADETProcessError(
                    "Flow rate is currently not supported for sensitivities."
                )
            parameters.append(parameter)

            associated_model = None
            if len(param_parts) == 3:
                associated_model = param_parts[1]

            if comp is not None and comp not in self.component_system.species:
                raise CADETProcessError(f"Unknown component {comp}.")

            unit = self.flow_sheet[unit]
            if unit not in self.flow_sheet.units:
                raise CADETProcessError("Not a valid unit")
            units.append(unit)

            if coeff is not None and parameter not in unit.polynomial_parameters:
                raise CADETProcessError("Not a polynomial parameter.")
            if parameter in unit.polynomial_parameters and coeff is None:
                raise CADETProcessError("Polynomial coefficient must be provided.")

            if associated_model is None:
                if parameter not in unit.parameters:
                    raise CADETProcessError("Not a valid parameter.")
            else:
                associated_model = getattr(unit, associated_model)

                if state is not None and state > associated_model.n_binding_sites:
                    raise ValueError("Binding site index exceed number of binding sites.")
                if reac is not None and reac > associated_model.n_reactions:
                    raise ValueError("Reaction index exceed number of reactions.")

                if parameter not in associated_model.parameters:
                    raise CADETProcessError("Not a valid parameter")
            associated_models.append(associated_model)

        sens = ParameterSensitivity(
            name,
            units,
            parameters,
            associated_models,
            components,
            polynomial_coefficients,
            reaction_indices,
            bound_state_indices,
            section_indices,
            abstols,
            factors,
        )
        self._parameter_sensitivities.append(sens)

    @property
    def system_state(self) -> np.ndarray:
        """np.ndarray: State of the entire system."""
        return self._system_state

    @system_state.setter
    def system_state(self, system_state: np.ndarray) -> None:
        self._system_state = system_state

    @property
    def system_state_derivative(self) -> np.ndarray:
        """np.ndarray: State derivative of the entire system."""
        return self._system_state_derivative

    @system_state_derivative.setter
    def system_state_derivative(self, system_state_derivative: np.ndarray) -> None:
        self._system_state_derivative = system_state_derivative

    @property
    def parameters(self) -> dict:
        """dict: Parameters of the process."""
        parameters = super().parameters

        parameters["flow_sheet"] = self.flow_sheet.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        try:
            self.flow_sheet.parameters = parameters.pop("flow_sheet")
        except KeyError:
            pass

        super(Process, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self) -> Dict:
        """dict: Section dependent parameters of the process."""
        parameters = Dict()
        parameters.flow_sheet = self.flow_sheet.section_dependent_parameters
        return parameters

    @property
    def polynomial_parameters(self) -> Dict:
        """dict: Polynomial parameters of the process."""
        parameters = super().polynomial_parameters
        parameters.flow_sheet = self.flow_sheet.polynomial_parameters
        return parameters

    @property
    def sized_parameters(self) -> Dict:
        """dict: Sized parameters of the process."""
        parameters = super().sized_parameters
        parameters.flow_sheet = self.flow_sheet.sized_parameters
        return parameters

    @property
    def initial_state(self) -> dict:
        """dict: Initial state of the process."""
        initial_state = {state: getattr(self, state) for state in self._initial_states}
        initial_state["flow_sheet"] = self.flow_sheet.initial_state

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: dict) -> None:
        try:
            self.flow_sheet.initial_state = initial_state.pop("flow_sheet")
        except KeyError:
            pass

        for state_name, state_value in initial_state.items():
            if state_name not in self._initial_state:
                raise CADETProcessError("Not an valid state")
            setattr(self, state_name, state_value)

    @property
    def config(self) -> Dict:
        """dict[str, dict]: Parameters and initial state of the process."""
        return Dict(
            {"parameters": self.parameters, "initial_state": self.initial_state}
        )

    @config.setter
    def config(self, config: dict) -> None:
        self.parameters = config["parameters"]
        self.initial_state = config["initial_state"]

    def add_concentration_profile(
        self,
        unit: str,
        time: np.ndarray,
        c: np.ndarray,
        components: Optional[list[str]] = None,
        s: float = 1e-6,
        interpolation_method: Literal["cubic", "pchip", None] = "pchip",
    ) -> None:
        """
        Add concentration profile to Process.

        Parameters
        ----------
        unit : str
            The name of the inlet unit operation.
        time : np.ndarray
            1D array containing the time values of the concentration profile.
        c : np.ndarray
            2D array containing the concentration profile with shape
            (len(time), n_comp), where n_comp is the number of components
            specified in the `components` argument.
        components : list[str] | None, optional
            Component species for which the concentration profile shall be added.
            If `None`, the profile is expected to have shape (len(time), n_comp).
            If `-1`, the same (1D) profile is added to all components.
            Default is `None`.
        s : float, optional
            Smoothing factor used to generate the spline representation of the
            concentration profile. Default is `1e-6`.
        interpolation_method : Literal["linear", "cubic", "pchip", None], optional
            The interpolation method to use. Options:
            - `"cubic"` : Cubic spline interpolation.
            - `"pchip"` : Piecewise cubic Hermite interpolation (default).
            - `None` : No interpolation, use raw time data.

        Raises
        ------
        TypeError
            If the specified `unit` is not an Inlet unit operation.
        ValueError
            If the time values in `time` exceed the cycle time of the Process.
            If `c` has an invalid shape.
            If `interpolation_method` is unknown.
        """
        if isinstance(unit, str):
            unit = self.flow_sheet[unit]

        if unit not in self.flow_sheet.inlets:
            raise TypeError("Expected Inlet")

        if max(time) > self.cycle_time:
            raise ValueError("Inlet profile exceeds cycle time.")

        # Handle components and concentration shape
        if components is None:
            if c.shape[1] != self.n_comp:
                raise ValueError(
                    f"Expected shape ({len(time), self.n_comp}) for concentration array"
                    f". Got {c.shape}."
                )
            components = self.component_system.species
        elif components == -1:
            # Assume same profile for all components
            if c.ndim > 1:
                raise ValueError("Expected single concentration profile.")
            c = np.column_stack([c] * self.n_comp)
            components = self.component_system.species

        if not isinstance(components, list):
            components = [components]

        indices = [self.component_system.species_indices[comp] for comp in components]
        if len(indices) == 1 and c.ndim == 1:
            c = np.array(c, ndmin=2).T

        # Determine the interpolation method
        if interpolation_method is None:
            time_interp = time
        else:
            time_interp = np.linspace(0, max(time), max(1001, len(time)))

            match interpolation_method:
                case None:
                    pass
                case "cubic":
                    interpolator = interpolate.CubicSpline(time, c)
                case "pchip":
                    interpolator = interpolate.PchipInterpolator(time, c)
                case _:
                    raise ValueError(
                        f"Unknown `interpolation_method`: {interpolation_method}."
                    )

            c = interpolator(time_interp)

        # Process each component's profile
        for i, comp in enumerate(indices):
            profile = c[:, i]

            # Normalize the profile
            min_val = np.min(profile)
            max_val = np.max(profile)
            range_val = max_val - min_val

            if range_val == 0:
                warn(
                    f"Component {comp} has no variation in concentration; "
                    "scaling will be skipped."
                )
                range_val = 1  # Avoid division by zero

            normalized_profile = (profile - min_val) / range_val

            # Fit the spline on the normalized profile
            tck = interpolate.splrep(time_interp, normalized_profile, s=s)
            ppoly = interpolate.PPoly.from_spline(tck)

            # Add events with properly unscaled coefficients
            for j, (t, sec) in enumerate(zip(ppoly.x, ppoly.c.T)):
                if j < 3 or j > len(ppoly.x) - 5:
                    continue
                # Scale coefficients and adjust the constant term
                scaled_sec = sec * range_val
                scaled_sec[-1] += min_val  # Adjust the constant term for shifting
                self.add_event(
                    f"{unit}_inlet_{comp}_{j - 3}",
                    f"flow_sheet.{unit}.c",
                    np.flip(scaled_sec),
                    t,
                    comp,
                )

    def add_flow_rate_profile(
        self,
        unit: str,
        time: np.ndarray,
        flow_rate: np.ndarray,
        s: float = 1e-6,
        interpolation_method: Literal["cubic", "pchip", None] = "pchip",
    ) -> None:
        """
        Add flow rate profile to a SourceMixin unit operation.

        Parameters
        ----------
        unit : str
            The name of the SourceMixin unit operation.
        time : np.ndarray
            1D array containing the time values of the flow rate profile.
        flow_rate : np.ndarray
            1D array containing the flow rate values over time.
        s : float, optional
            Smoothing factor used to generate the spline representation of the flow rate profile.
            Default is `1e-6`.
        interpolation_method : Literal["linear", "cubic", "pchip", None], optional
            The interpolation method to use. Options:
            - `"cubic"` : Cubic spline interpolation.
            - `"pchip"` : Piecewise cubic Hermite interpolation (default).
            - `None` : No interpolation, use raw time data.

        Raises
        ------
        TypeError
            If the specified `unit` is not a SourceMixin unit operation.
        ValueError
            If the time values in `time` exceed the cycle time of the Process.
        """
        if isinstance(unit, str):
            unit = self.flow_sheet[unit]

        if unit not in self.flow_sheet.inlets + self.flow_sheet.cstrs:
            raise TypeError("Expected SourceMixin.")

        if max(time) > self.cycle_time:
            raise ValueError("Inlet profile exceeds cycle time.")

        # Compute min and max for scaling
        min_val = np.min(flow_rate)
        max_val = np.max(flow_rate)
        range_val = max_val - min_val

        if range_val == 0:
            warn("Flow rate has no variation; scaling will be skipped.")
            range_val = 1  # Avoid division by zero, effectively skipping scaling

        # Normalize flow_rate to [0, 1]
        normalized_flow_rate = (flow_rate - min_val) / range_val

        # Determine interpolation method
        if interpolation_method is None:
            time_interp = time
            interpolated_flow_rate = normalized_flow_rate
        else:
            time_interp = np.linspace(0, max(time), max(1001, len(time)))
            match interpolation_method:
                case "cubic":
                    interpolator = interpolate.CubicSpline(time, normalized_flow_rate)
                case "pchip":
                    interpolator = interpolate.PchipInterpolator(
                        time, normalized_flow_rate
                    )
                case _:
                    raise ValueError(
                        f"Unknown `interpolation_method`: {interpolation_method}."
                    )

            interpolated_flow_rate = interpolator(time_interp)

        # Fit the spline with the interpolated normalized flow rate
        tck = interpolate.splrep(time_interp, interpolated_flow_rate, s=s)
        ppoly = interpolate.PPoly.from_spline(tck)

        # Add events with unscaled coefficients
        for i, (t, sec) in enumerate(zip(ppoly.x, ppoly.c.T)):
            if i < 3 or i > len(ppoly.x) - 5:
                continue

            # Unscale all coefficients
            unscaled_sec = sec * range_val
            unscaled_sec[-1] += min_val  # Adjust the constant term for shifting

            self.add_event(
                f"{unit}_flow_rate_{i - 3}",
                f"flow_sheet.{unit}.flow_rate",
                np.flip(unscaled_sec),
                t,
            )

    def check_config(self) -> bool:
        """
        Validate that process config is setup correctly.

        Returns
        -------
        check : Bool
            True if process is setup correctly. False otherwise.
        """
        flag = super().check_config()

        missing_parameters = self.flow_sheet.missing_parameters
        if len(missing_parameters) > 0:
            for param in missing_parameters:
                if f"flow_sheet.{param}" not in self.event_parameters:
                    warn(f"Missing parameter {param}.")
                    flag = False

        if not self.flow_sheet.check_connections():
            flag = False

        if self.cycle_time is None:
            warn("Cycle time is not set")
            flag = False

        if not self.check_cstr_volume():
            flag = False

        return flag

    def check_cstr_volume(self) -> bool:
        """
        Check if CSTRs run empty.

        Returns
        -------
        flag : bool
            False if any of the CSTRs run empty. True otherwise.
        """
        flag = True
        for cstr in self.flow_sheet.cstrs:
            if cstr.flow_rate is None:
                continue
            V_0 = cstr.init_liquid_volume
            unit_index = self.flow_sheet.get_unit_index(cstr)
            for port in self.flow_sheet.units[unit_index].ports:
                V_in = self.flow_rate_timelines[cstr.name].total_in[port].integral()
                V_out = self.flow_rate_timelines[cstr.name].total_out[port].integral()
                if V_0 + V_in - V_out < 0:
                    flag = False
                    warn(f"CSTR {cstr.name} runs empty on port {port} during process.")

        return flag

    def __str__(self) -> str:
        """str: String representation of the process."""
        return self.name


@dataclass
class ParameterSensitivity:
    """Class for storing parameter sensitivity parameters."""

    name: str
    units: list
    parameters: list
    associated_models: list = None
    components: list = None
    polynomial_coefficients: list = None
    reaction_indices: list = None
    bound_state_indices: list = None
    section_indices: list = None
    abstols: list = None
    factors: list = None
