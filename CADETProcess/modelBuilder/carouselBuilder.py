from copy import deepcopy
from functools import wraps
import warnings

import numpy as np
import matplotlib.pyplot as plt

from CADETProcess import CADETProcessError
from CADETProcess import plotting

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import Integer, UnsignedInteger, UnsignedFloat

from CADETProcess.processModel import Linear, Langmuir
from CADETProcess.processModel import UnitBaseClass, FlowSheet, Process
from CADETProcess.processModel import Inlet, TubularReactorBase, Cstr, Outlet

from CADETProcess.solution import SolutionBase


__all__ = [
    'SerialZone',
    'ParallelZone',
    'CarouselBuilder',
    'LinearSMBBuilder',
    'LangmuirSMBBuilder',
]


class CarouselBuilder(Structure):
    switch_time = UnsignedFloat()

    def __init__(self, component_system, name):
        self.component_system = component_system
        self.name = name
        self._flow_sheet = FlowSheet(component_system, name)
        self._column = None

    @property
    def flow_sheet(self):
        return self._flow_sheet

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, column):
        if not isinstance(column, TubularReactorBase):
            raise TypeError
        if self.component_system is not column.component_system:
            raise CADETProcessError('Number of components does not match.')
        self._column = column

    @wraps(FlowSheet.add_unit)
    def add_unit(self, *args, **kwargs):
        """Wrapper around function of auxiliary flow_sheet."""
        self.flow_sheet.add_unit(*args, **kwargs)

    @wraps(FlowSheet.add_connection)
    def add_connection(self, *args, **kwargs):
        """Wrapper around function of auxiliary flow_sheet."""
        self.flow_sheet.add_connection(*args, **kwargs)

    @wraps(FlowSheet.set_output_state)
    def set_output_state(self, *args, **kwargs):
        """Wrapper around function of auxiliary flow_sheet."""
        self.flow_sheet.set_output_state(*args, **kwargs)

    @property
    def zones(self):
        """list: list of all zones in the carousel system."""
        return [
            unit for unit in self.flow_sheet.units
            if isinstance(unit, ZoneBaseClass)
        ]

    @property
    def zone_names(self):
        """list: Zone names."""
        return [zone.name for zone in self.zones]

    @property
    def zones_dict(self):
        """dict: Zone names and objects."""
        return {zone.name: zone for zone in self.zones}

    @property
    def n_zones(self):
        """int: Number of zones in the Carousel System"""
        return len(self.zones)

    @property
    def n_columns(self):
        """int: Number of columns in the Carousel System"""
        return sum([zone.n_columns for zone in self.zones])

    def build_flow_sheet(self):
        """Build flow_sheet."""
        if self.column is None:
            raise CADETProcessError("No column associated with Carousel.")

        flow_sheet = FlowSheet(self.component_system, self.name)

        self._add_units(flow_sheet)
        self._add_inter_zone_connections(flow_sheet)
        self._add_intra_zone_connections(flow_sheet)

        return flow_sheet

    def _add_units(self, flow_sheet):
        """Add units to flow_sheet"""
        col_index = 0
        for unit in self.flow_sheet.units:
            if not isinstance(unit, ZoneBaseClass):
                is_feed_inlet = unit in self.flow_sheet.feed_inlets
                is_eluent_inlet = unit in self.flow_sheet.eluent_inlets
                is_output_outlet = unit in self.flow_sheet.product_outlets
                flow_sheet.add_unit(
                    unit,
                    feed_inlet=is_feed_inlet,
                    eluent_inlet=is_eluent_inlet,
                    product_outlet=is_output_outlet,
                )
            else:
                flow_sheet.add_unit(unit.inlet_unit)
                flow_sheet.add_unit(unit.outlet_unit)
                for i_col in range(unit.n_columns):
                    col = deepcopy(self.column)
                    col.component_system = self.component_system
                    col.name = f'column_{col_index}'
                    if unit.initial_state is not None:
                        col.initial_state = unit.initial_state[i_col]
                    flow_sheet.add_unit(col)
                    col_index += 1

    def _add_inter_zone_connections(self, flow_sheet):
        """Add connections between zones."""
        for unit, connections in self.flow_sheet.connections.items():
            if isinstance(unit, ZoneBaseClass):
                origin = unit.outlet_unit
            else:
                origin = unit

            for destination in connections.destinations:
                if isinstance(destination, ZoneBaseClass):
                    destination = destination.inlet_unit

                flow_sheet.add_connection(origin, destination)

        for zone in self.zones:
            output_state = self.flow_sheet.output_states[zone]
            flow_sheet.set_output_state(zone.outlet_unit, output_state)

    def _add_intra_zone_connections(self, flow_sheet):
        """Add connections within zones."""
        for zone in self.zones:
            for col_index in range(self.n_columns):
                col = flow_sheet[f'column_{col_index}']
                flow_sheet.add_connection(zone.inlet_unit, col)
                col = flow_sheet[f'column_{col_index}']
                flow_sheet.add_connection(col, zone.outlet_unit)

        for col_index in range(self.n_columns):
            col_orig = flow_sheet[f'column_{col_index}']
            if col_index < self.n_columns - 1:
                col_dest = flow_sheet[f'column_{col_index + 1}']
            else:
                col_dest = flow_sheet[f'column_{0}']
            flow_sheet.add_connection(col_orig, col_dest)

    def build_process(self):
        """Build process."""
        flow_sheet = self.build_flow_sheet()
        process = Process(flow_sheet, self.name)

        self._add_events(process)

        return process

    @property
    def cycle_time(self):
        """float: cycle time of the process."""
        return self.n_columns * self.switch_time

    def _add_events(self, process):
        """Add events to process."""
        process.cycle_time = self.n_columns * self.switch_time
        process.add_duration('switch_time', self.switch_time)

        for carousel_state in range(self.n_columns):
            position_counter = 0
            for i_zone, zone in enumerate(self.zones):
                col_indices = np.arange(zone.n_columns)
                col_indices += position_counter
                col_indices = self.column_indices_at_state(
                    col_indices, carousel_state
                )

                if isinstance(zone, SerialZone):
                    evt = process.add_event(
                        f'{zone.name}_{carousel_state}',
                        f'flow_sheet.output_states.{zone.inlet_unit}',
                        col_indices[0]
                    )
                    process.add_event_dependency(
                        evt.name, 'switch_time', [carousel_state]
                    )
                    for i, col in enumerate(col_indices):
                        if i < (zone.n_columns - 1):
                            evt = process.add_event(
                                f'column_{col}_{carousel_state}',
                                f'flow_sheet.output_states.column_{col}',
                                self.n_zones
                            )
                        else:
                            evt = process.add_event(
                                f'column_{col}_{carousel_state}',
                                f'flow_sheet.output_states.column_{col}',
                                i_zone
                            )
                        process.add_event_dependency(
                            evt.name, 'switch_time', [carousel_state]
                        )
                elif isinstance(zone, ParallelZone):
                    output_state = self.n_columns * [0]
                    for col in col_indices:
                        output_state[col] = 1/zone.n_columns

                    evt = process.add_event(
                            f'{zone.name}_{carousel_state}',
                            f'flow_sheet.output_states.{zone.inlet_unit}',
                            output_state
                    )
                    process.add_event_dependency(
                            evt.name, 'switch_time', [carousel_state]
                    )

                    for col in col_indices:
                        evt = process.add_event(
                            f'column_{col}_{carousel_state}',
                            f'flow_sheet.output_states.column_{col}',
                            i_zone
                        )
                        process.add_event_dependency(
                            evt.name, 'switch_time', [carousel_state]
                        )

                for i, col in enumerate(col_indices):
                    evt = process.add_event(
                        f'column_{col}_{carousel_state}_velocity',
                        f'flow_sheet.column_{col}.flow_direction',
                        zone.flow_direction
                    )
                    process.add_event_dependency(
                        evt.name, 'switch_time', [carousel_state]
                    )

                position_counter += zone.n_columns

    def carousel_state(self, t):
        """int: Carousel state at given time.

        Parameters
        ----------
        t: float
            Time
        """
        return int(np.floor((t % self.cycle_time) / self.switch_time))

    def column_indices_at_state(
            self,
            carousel_positions: np.typing.NDArray[int],
            carousel_state: int
            ) -> np.ndarray[int]:
        """Determine index of column unit at given carousel position and state.

        Parameters
        ----------
        carousel_positions: np.typing.NDArray[np.int_]
            Carousel position indices (e.g. wash position, elute position).
        carousel_state : int
            Curent state of the carousel system.

        Returns
        -------
        np.ndarray[int]
            Indices of column units at given carousel positions and state.
        """
        carousel_positions = np.array(carousel_positions, dtype=int)

        return (carousel_positions + carousel_state) % self.n_columns

    def column_indices_at_time(
            self,
            t: float,
            carousel_positions: np.typing.NDArray[int],
            ) -> int:
        """Determine index of column unit at given carousel position and time.

        Parameters
        ----------
        t: float
            Time
        carousel_position : int
            Carousel position.

        Returns
        -------
        np.ndarray[int]
            Indices of column units at given carousel positions and time.
        """
        carousel_positions = np.array(carousel_positions, dtype=int)

        carousel_state = self.carousel_state(t)
        column_indices = self.column_indices_at_state(
            carousel_positions, carousel_state
        )

        return column_indices


class ZoneBaseClass(UnitBaseClass):
    n_columns = UnsignedInteger()
    flow_direction = Integer(default=1)
    valve_dead_volume = UnsignedFloat(default=1e-6)

    def __init__(
            self,
            component_system, name, n_columns=1,
            flow_direction=1, initial_state=None,
            *args, **kwargs):
        self.n_columns = n_columns
        self.flow_direction = flow_direction
        self.initial_state = initial_state

        self._inlet_unit = Cstr(component_system, f'{name}_inlet')

        self._inlet_unit.V = self.valve_dead_volume
        self._outlet_unit = Cstr(component_system, f'{name}_outlet')
        self._outlet_unit.V = self.valve_dead_volume

        super().__init__(component_system, name, *args, **kwargs)

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        if initial_state is None:
            self._initial_state = initial_state
            return

        if not isinstance(initial_state, list):
            initial_state = self.n_columns * [initial_state]

        if len(initial_state) != self.n_columns:
            raise CADETProcessError(f"Expected size {self.n_columns}")

        self._initial_state = initial_state

    @property
    def inlet_unit(self):
        return self._inlet_unit

    @property
    def outlet_unit(self):
        return self._outlet_unit


class SerialZone(ZoneBaseClass):
    pass


class ParallelZone(ZoneBaseClass):
    pass


class SMBBuilder(CarouselBuilder):
    binding_model_type = None

    def __init__(self, feed, eluent, column, name='SMB'):
        component_system = feed.component_system
        if not (component_system is eluent.component_system is column.component_system):
            raise CADETProcessError("ComponentSystems do not match.")
        if not isinstance(feed, Inlet):
            raise TypeError(f"Expected inlet object. Got {type(feed)}.")
        if not isinstance(eluent, Inlet):
            raise TypeError(f"Expected inlet object. Got {type(eluent)}.")

        if not isinstance(column, TubularReactorBase):
            raise TypeError(f"Expected Column object. Got {type(column)}.")

        self._validate_binding_model(column.binding_model)

        super().__init__(component_system, name)

        raffinate = Outlet(component_system, name='raffinate')
        extract = Outlet(component_system, name='extract')

        zone_I = SerialZone(component_system, 'zone_I', 1)
        zone_II = SerialZone(component_system, 'zone_II', 1)
        zone_III = SerialZone(component_system, 'zone_III', 1)
        zone_IV = SerialZone(component_system, 'zone_IV', 1)

        # Carousel Builder
        self.column = column
        self.add_unit(feed, feed_inlet=True)
        self.add_unit(eluent, eluent_inlet=True)

        self.add_unit(raffinate, product_outlet=True)
        self.add_unit(extract, product_outlet=True)

        self.add_unit(zone_I)
        self.add_unit(zone_II)
        self.add_unit(zone_III)
        self.add_unit(zone_IV)

        self.add_connection(eluent, zone_I)

        self.add_connection(zone_I, extract)
        self.add_connection(zone_I, zone_II)

        self.add_connection(zone_II, zone_III)

        self.add_connection(feed, zone_III)

        self.add_connection(zone_III, raffinate)
        self.add_connection(zone_III, zone_IV)

        self.add_connection(zone_IV, zone_I)

    def _validate_binding_model(self, binding_model):
        if not isinstance(binding_model, self.binding_model_type):
            raise TypeError(f'Invalid binding model. Expected {self.binding_model_type}.')

        if binding_model.n_comp != 2:
            raise CADETProcessError("This only works for 2-Component Systems.")

        if binding_model.is_kinetic:
            warnings.warn(
                "Isotherm uses kinetic binding, "
                "however, triangle theory assumes instant equilibrium."
            )

    def _get_zone_flow_rates(self, m, switch_time):
        m1, m2, m3, m4 = m

        Vc = self.column.volume
        et = self.column.total_porosity

        Q_I = Vc*(m1*(1-et)+et)/switch_time   # Flussrate Zone 1
        Q_II = Vc*(m2*(1-et)+et)/switch_time   # Flussrate Zone 2
        Q_III = Vc*(m3*(1-et)+et)/switch_time   # Flussrate Zone 3
        Q_IV = Vc*(m4*(1-et)+et)/switch_time   # Flussrate Zone 4

        return [Q_I, Q_II, Q_III, Q_IV]

    def _get_unit_flow_rates(self, Q_zones):
        Q_I, Q_II, Q_III, Q_IV = Q_zones

        Q_feed = Q_III - Q_II
        Q_eluent = Q_I - Q_IV
        Q_raffinate = Q_III - Q_IV
        Q_extract = Q_I - Q_II

        return [Q_feed, Q_eluent, Q_raffinate, Q_extract]

    def _get_split_ratios(self, Q_zones, Q_units):
        Q_I, Q_II, Q_III, Q_IV = Q_zones
        Q_feed, Q_eluent, Q_raffinate, Q_extract = Q_units

        w_r = Q_raffinate / Q_III
        w_e = Q_extract / Q_I

        return w_r, w_e

    def get_design_parameters(self, binding_model):
        raise NotImplementedError()

    def calculate_m_opt(self, *design_parameters):
        raise NotImplementedError()

    def apply_safety_factor(self, m_opt, *design_parameters, gamma):
        raise NotImplementedError()

    def triangle_design(
            self,
            binding_model=None,
            c_feed=None,
            switch_time=None,
            gamma=1,
            set_values=True,
            ):

        if binding_model is None:
            binding_model = self.column.binding_model
        elif set_values is True:
            warnings.warn("Cannot set values if binding_model is given.")
            set_values = False

        self._validate_binding_model(binding_model)

        if c_feed is None:
            c_feed = self.flow_sheet.feed.c[:, 0]
        elif set_values is True:
            self.flow_sheet.feed.c = c_feed

        design_parameters = self.get_design_parameters(binding_model, c_feed)
        m_opt = self.calculate_m_opt(*design_parameters)
        m = self.apply_safety_factor(m_opt, *design_parameters, gamma)

        if switch_time is None:
            switch_time = self.switch_time
        elif set_values is True:
            self.switch_time = switch_time

        Q_zones = self._get_zone_flow_rates(m, switch_time)
        Q_units = self._get_unit_flow_rates(Q_zones)
        Q_feed, Q_eluent, Q_raffinate, Q_extract = Q_units
        w_r, w_e = self._get_split_ratios(Q_zones, Q_units)

        if set_values:
            self.flow_sheet.feed.flow_rate = Q_feed
            self.flow_sheet.eluent.flow_rate = Q_eluent
            self.set_output_state('zone_I', [w_e, 1-w_e])
            self.set_output_state('zone_III', [w_r, 1-w_r])

        return [Q_feed, Q_eluent, w_r, w_e]

    def plot_triangle(
            self,
            binding_model=None,
            c_feed=None,
            gamma=1,
            operating_point=None,
            fig=None,
            ax=None,
            ):
        if binding_model is None:
            binding_model = self.column.binding_model
        self._validate_binding_model(binding_model)

        if c_feed is None:
            c_feed = self.flow_sheet.feed.c[:, 0]

        # Operating Point
        design_parameters = self.get_design_parameters(binding_model, c_feed)
        m_opt = self.calculate_m_opt(*design_parameters)
        m1, m2, m3, m4 = self.apply_safety_factor(m_opt, *design_parameters, gamma)

        # Setup figure
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Plot Triangle
        self._plot_triangle(ax, *design_parameters)

        ax.set_xlabel('$m_{II}$')
        ax.set_ylabel('$m_{III}$')

        # Operating point
        ax.plot(m2, m3, 'ok')

        return fig, ax

    def _plot_triangle(self, ax, *design_parameters):
        raise NotImplementedError()


class LinearSMBBuilder(SMBBuilder):
    binding_model_type = Linear

    def get_design_parameters(self, binding_model, c_feed):
        k_ads = np.array(binding_model.adsorption_rate)
        k_des = np.array(binding_model.desorption_rate)
        H = k_ads / k_des

        if H[1] < H[0]:
            HA, HB = H
        else:
            HB, HA = H

        return HA, HB

    def calculate_m_opt(self, HA, HB):
        m1 = HA
        m2 = HB
        m3 = HA
        m4 = HB

        return [m1, m2, m3, m4]

    def apply_safety_factor(self, m_opt, *design_parameters, gamma=1):
        m1_opt, m2_opt, m3_opt, m4_opt = m_opt

        if np.isscalar(gamma):
            gamma = 4 * [gamma]

        gamma_1, gamma_2, gamma_3, gamma_4 = gamma

        if gamma_2 * gamma_3 >= m3_opt / m2_opt:
            raise ValueError("gamma_2 * gamma_3 must be smaller than HA / HB ")

        m1 = gamma_1 * m1_opt
        m2 = gamma_2 * m2_opt
        m3 = m3_opt / gamma_3
        m4 = m4_opt / gamma_4

        return [m1, m2, m3, m4]

    def _plot_triangle(self, ax, HA, HB):
        """
        Plot SMB triangle for linear isotherm.

        Notable points:
        - a: [HA, HA]
        - b: [HB, HB]
        - w_opt: [HB, HA]

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        binding_model : CADETProcess.processModel.BindingBaseClass
            Binding model to use for plotting.
        gamma : list or float, optional
            Safety factor(s) for operating points.

        """
        # Bounds
        lb = HB - 0.3 * (HA - HB)
        ub = HA + 0.3 * (HA - HB)

        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)

        # Diagonal
        ax.plot((lb, ub), (lb, ub), 'k')

        # Henry coefficients
        for h in [HB, HA]:
            ax.hlines(h, 0, h, 'k', 'dashed')
            ax.vlines(h, h, ub, 'k', 'dashed')

        # Triangle
        ax.hlines(HA, HB, HA, 'k')
        ax.vlines(HB, HB, HA, 'k')

        # Label regions
        ax.text(
            (HB + (HA - HB) / 2), (0.95 * ub),
            'Pure extract',
            ha='center', va='center'
        )
        ax.text(
            (1.05 * lb), (HB + (HA - HB) / 2),
            'Pure raffinate',
            ha='center', va='center',
            rotation='vertical',
        )


class LangmuirSMBBuilder(SMBBuilder):
    binding_model_type = Langmuir

    def get_design_parameters(self, binding_model, c_feed):
        k_ads = np.array(binding_model.adsorption_rate)
        k_des = np.array(binding_model.desorption_rate)
        k_eq = k_ads / k_des
        q_sat = np.array(binding_model.capacity)
        H = [k_eq * q_s for k_eq, q_s in zip(k_eq, q_sat)]

        if H[1] < H[0]:
            HA, HB = H
            bA, bB = k_eq
            cFA, cFB = c_feed
        else:
            HB, HA = H
            bB, bA = k_eq
            cFB, cFA = c_feed

        a = -(HA * (1 + bB * cFB) + HB * (1 + bA * cFA)) / (1 + bB * cFB + bA * cFA)
        b = HA * HB / (1 + bB * cFB + bA * cFA)
        wG = -a / 2 + np.sqrt((-a / 2)**2 - b)
        wF = -a / 2 - np.sqrt((-a / 2)**2 - b)

        return HA, HB, bA, bB, cFA, cFB, wG, wF

    def calculate_m_opt(self, HA, HB, bA, bB, cFA, cFB, wG, wF):
        m1 = HA
        m2 = HB / HA * wG
        m3 = wG * (wF * (HA - HB) + HB * (HB - wF)) / (HB * (HA - wF))
        m4 = 1 / 2 * (HB + m3 + bB * cFB * (m3 - m2) - np.sqrt((HB + m3 + bB * cFB * (m3 - m2))**2 - 4 * HB * m3))

        return [m1, m2, m3, m4]

    def apply_safety_factor(self, m_opt, HA, HB, bA, bB, cFA, cFB, wG, wF, gamma=1):
        m1_opt, m2_opt, m3_opt, m4_opt = m_opt

        if np.isscalar(gamma):
            W_opt = np.array([m2_opt, m3_opt])
            B = np.array([HB, HB])
            R = [
                wG ** 2 / HA,
                wG * (wF * (HA - wG) * (HA - HB) + HB * wG * (HA - wF)) / (HA * HB * (HA - wF))
            ]

            # Calculating vectors WB and WA
            WB = B - W_opt
            WR = R - W_opt

            # Normalizing vectors
            norm_WB = WB / np.linalg.norm(WB)
            norm_WR = WR / np.linalg.norm(WR)

            # Calculating the bisector WB / WR
            bisector = norm_WB + norm_WR
            norm_dir_vec_bisector = bisector / np.linalg.norm(bisector)

            # Calculating the new point W' using the safety factor gamma
            W = W_opt + (gamma - 1) * norm_dir_vec_bisector

            m1 = gamma * m1_opt
            m2 = W[0]
            m3 = W[1]
            m4 = m4_opt / gamma

        else:
            gamma_1, gamma_2, gamma_3, gamma_4 = gamma

            m1 = gamma_1 * m1_opt
            m2 = gamma_2 * m2_opt
            m3 = m3_opt / gamma_3
            m4 = m4_opt / gamma_4

        return [m1, m2, m3, m4]

    def _plot_triangle(self, ax, HA, HB, bA, bB, cFA, cFB, wG, wF):
        """
        Plot SMB triangle for Langmuir isotherm.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        """
        m1, m2, m3, m4 = self.calculate_m_opt(HA, HB, bA, bB, cFA, cFB, wG, wF)
        W = [m2, m3]

        R = [
            wG ** 2 / HA,
            wG * (wF * (HA - wG) * (HA - HB) + HB * wG * (HA - wF)) / (HA * HB * (HA - wF))
        ]

        # Bounds
        lb = W[0] - 0.3 * (HA - W[0])
        ub = HA + 0.3 * (HA - W[0])

        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)

        # Diagonal
        ax.plot((lb, ub), (lb, ub), 'k')

        # Plot [W -> R]
        m2WR = np.linspace(W[0], R[0], 50)
        m3WR = 1 / (bA * cFA * wG) * (wG * (HA - wG) - (HA - wG * (1 + bA * cFA)) * m2WR)
        ax.plot(m2WR, m3WR, 'k-')

        # plot [W -> HB]
        m2WHB = np.linspace(W[0], HB, 10)
        m3WHB = 1 / (bA * cFA * HB) * (HB * (HA - HB) - (HA - HB * (1 + bA * cFA)) * m2WHB)
        ax.plot(m2WHB, m3WHB, 'k-')

        # plot [R -> HA]
        m2RHA = np.linspace(R[0], HA, 10)
        m3RHA = m2RHA + (np.sqrt(HA) - np.sqrt(m2RHA)) ** 2 / (bA * cFA)
        ax.plot(m2RHA, m3RHA, 'k-')

        # TODO: Equations that plot regions of pure extract / raffinate not clear yet.


class CarouselSolutionBulk(SolutionBase):
    """Solution at unit inlet or outlet.

    N_COLUMNS * NCOL * NRAD

    """
    _coordinates = ['axial_coordinates', 'radial_coordinates']

    def __init__(self, builder, simulation_results):
        self.builder = builder
        self.simulation_results = simulation_results

    @property
    def component_system(self):
        return self.builder.component_system

    @property
    def solution(self):
        return self.simulation_results.solution

    @property
    def axial_coordinates(self):
        return self.simulation_results.solution.column_0.bulk.axial_coordinates

    @property
    def radial_coordinates(self):
        radial_coordinates = \
            self.simulation_results.solution.column_0.bulk.radial_coordinates
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None

        return radial_coordinates

    @property
    def time(self):
        return self.simulation_results.solution.column_0.bulk.time

    def plot_at_time(
            self, t, overlay=None, y_min=None, y_max=None,
            ax=None, lines=None):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting
        ax : Axes
            Axes to plot on.

        See Also
        --------
        CADETProcess.plotting
        """

        n_cols = self.builder.n_columns
        if ax is None:
            fig, axs = plt.subplots(
                ncols=n_cols,
                figsize=(n_cols*4, 6),
                gridspec_kw=dict(wspace=0.0, hspace=0.0),
                sharey='row'
            )
        else:
            axs = ax

        t_i = np.where(t <= self.time)[0][0]

        x = self.axial_coordinates

        y_min_data = 0
        y_max_data = 0
        zone_counter = 0
        column_counter = 0

        if lines is None:
            _lines = []
        else:
            _lines = None

        for position, ax in enumerate(axs):
            col_index = self.builder.column_indices_at_time(t, position)

            y = self.solution[f'column_{col_index}'].bulk.solution[t_i, :]

            y_min_data = min(y_min_data, min(0, np.min(y)))
            y_max_data = max(y_max_data, 1.1*np.max(y))

            if lines is not None:
                for comp in range(self.n_comp):
                    lines[position][comp].set_ydata(y[..., comp])
            else:
                l = ax.plot(x, y)
                _lines.append(l)

            zone = self.builder.zones[zone_counter]

            if zone.n_columns > 1:
                ax.set_title(f'{zone.name}, position {column_counter}')
            else:
                ax.set_title(f'{zone.name}')

            if column_counter < (zone.n_columns - 1):
                column_counter += 1
            else:
                zone_counter += 1
                column_counter = 0

        plotting.add_text(ax, f'time = {t:.2f} s')

        if y_min is None:
            y_min = y_min_data
        if y_min is None:
            y_min = y_max_data

        for position, ax in enumerate(axs):
            ax.set_ylim((y_min, y_max))

        if _lines is None:
            _lines = lines

        return axs, _lines
