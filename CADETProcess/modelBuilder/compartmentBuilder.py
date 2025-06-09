from functools import wraps
from typing import Optional

import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import String, StructMeta, UnsignedInteger
from CADETProcess.processModel import (
    BindingBaseClass,
    BulkReactionBase,
    ComponentSystem,
    Cstr,
    FlowSheet,
    Inlet,
    Outlet,
    ParticleReactionBase,
    Process,
    ReactionBaseClass,
)


class CompartmentBuilder(metaclass=StructMeta):
    """
    Class to build complex compartment models of bioreactors.

    Attributes
    ----------
    name : str
        Name of the CompartmentBuilder.
    n_compartments : int
        Number of compartments in the bioreactor.
    """

    name = String()
    n_compartments = UnsignedInteger()

    def __init__(
        self,
        component_system: ComponentSystem,
        compartment_volumes: list[float],
        flow_rate_matrix: list[float],
        init_c: int | float | list | np.ndarray = 0,
        binding_model: Optional[BindingBaseClass] = None,
        bulk_reaction_model: Optional[BulkReactionBase] = None,
        particle_reaction_model: Optional[ParticleReactionBase] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize builder.

        Parameters
        ----------
        component_system : component_system
            DESCRIPTION.
        compartment_volumes : list
            Volume of compartments.
        flow_rate_matrix : list
            Flow rates between compartments.
        init_c : int, float, list, np.array, optional
            Initial concentrations. The default is 0.
        binding_model : BindingBaseClass, optional
            Binding model to set for all compartments. The default is None.
        bulk_reaction_model : TYPE, optional
            Bulk reaction model for all compartments. The default is None.
        particle_reaction_model : TYPE, optional
            Particle reaction model for all compartments. The default is None.
        name : str, optional
            Name of the model. The default is None.
        """
        self.component_system = component_system
        if name is None:
            name = "CompartmentBuilder"
        self.name = name

        self._compartment_model = CompartmentModel(
            component_system, "master compartment"
        )

        self.binding_model = binding_model
        self.bulk_reaction_model = bulk_reaction_model
        self.particle_reaction_model = particle_reaction_model

        self._flow_sheet = FlowSheet(component_system, name)
        self._process = Process(self._flow_sheet, name)

        self._add_compartments(compartment_volumes)
        self._add_connections(flow_rate_matrix)

        self.init_c = init_c

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def _real_compartments(self) -> list[str]:
        """list: Compartment units excluding pseudo units s.a. Inlet/Outlet."""
        compartments = []
        for i in range(self.n_compartments):
            name = f"compartment_{i}"
            compartment = self.flow_sheet[name]
            if isinstance(compartment, Cstr):
                compartments.append(compartment)

        return compartments

    @property
    @wraps(Cstr.binding_model)
    def binding_model(self) -> ReactionBaseClass:
        """Wrapper around master compartment to set binding model."""
        return self._compartment_model.binding_model

    @binding_model.setter
    @wraps(Cstr.binding_model)
    def binding_model(self, binding_model: BindingBaseClass) -> None:
        if binding_model is not None:
            self._compartment_model.binding_model = binding_model

            for compartment in self._real_compartments:
                compartment.binding_model = binding_model

    @property
    @wraps(Cstr.bulk_reaction_model)
    def bulk_reaction_model(self) -> BindingBaseClass:
        """Wrapper around master compartment to set bulk reaction model."""
        return self._compartment_model.bulk_reaction_model

    @bulk_reaction_model.setter
    @wraps(Cstr.bulk_reaction_model)
    def bulk_reaction_model(self, bulk_reaction_model: BulkReactionBase) -> None:
        if bulk_reaction_model is not None:
            self._compartment_model.bulk_reaction_model = bulk_reaction_model

            for compartment in self._real_compartments:
                compartment.bulk_reaction_model = bulk_reaction_model

    @property
    @wraps(Cstr.particle_reaction_model)
    def particle_reaction_model(self) -> ReactionBaseClass:
        """Wrapper around master compartment to set particle reaction model."""
        return self._compartment_model.particle_reaction_model

    @particle_reaction_model.setter
    @wraps(Cstr.particle_reaction_model)
    def particle_reaction_model(
        self, particle_reaction_model: ParticleReactionBase
    ) -> None:
        if particle_reaction_model is not None:
            self._compartment_model.particle_reaction_model = particle_reaction_model

            for compartment in self._real_compartments:
                compartment.particle_reaction_model = particle_reaction_model

    @property
    def flow_sheet(self) -> FlowSheet:
        """FlowSheet: FlowSheet of the compartment builder."""
        return self._flow_sheet

    @property
    def process(self) -> Process:
        """Process: Process of the compartment builder."""
        return self._process

    @property
    def cycle_time(self) -> float:
        """float: Cycle time of the process."""
        return self.process.cycle_time

    @cycle_time.setter
    def cycle_time(self, cycle_time: float) -> None:
        self.process.cycle_time = cycle_time

    def _add_compartments(self, compartment_volumes: list[float]) -> None:
        """Instantiate compartments and add to FlowSheet."""
        self.n_compartments = len(compartment_volumes)

        for i, vol in enumerate(compartment_volumes):
            name = f"compartment_{i}"

            if vol == "inlet":
                unit = Inlet(self.component_system, name)
            elif vol == "outlet":
                unit = Outlet(self.component_system, name)
            else:
                unit = Cstr(self.component_system, name)
                unit.init_liquid_volume = vol

            self.flow_sheet.add_unit(unit)

    def _add_connections(self, connections_matrix: npt.ArrayLike) -> None:
        """Add connections and flow rates between compartments to FlowSheet."""
        try:
            if isinstance(connections_matrix, list):
                arr = np.array(connections_matrix)
                connections_matrix = arr.reshape(-1, self.n_compartments)
        except ValueError:
            raise CADETProcessError("Expected square matrix")

        if connections_matrix.shape[0] != connections_matrix.shape[1]:
            raise CADETProcessError("Expected square matrix")

        for iOrigin, destinations in enumerate(connections_matrix):
            flow_rate = np.sum(destinations)

            origin = self.flow_sheet[f"compartment_{iOrigin}"]
            output_state = []

            for iDestination, flow in enumerate(destinations):
                if flow == 0:
                    continue

                destination = self.flow_sheet[f"compartment_{iDestination}"]
                output_state.append(flow / flow_rate)

                self.flow_sheet.add_connection(origin, destination)

            if isinstance(origin, Outlet):
                continue

            self.flow_sheet[origin.name].flow_rate = flow_rate
            self.flow_sheet.set_output_state(origin, output_state)

    @property
    def init_c(self) -> int | float | list | np.ndarray:
        """
        np.array: Initial conditions of compartments.

        Parameters
        ----------
        init_c : int, float, list, or np.array
            Initial concentration of compartments.
            If type is float or int, same value is used for all components
            and concentrations.
            If type is list, same component concentrations are used for all
            compartments.
            If np.array, explicit concentrations for components and
            compartments are set.

        Raises
        ------
        ValueError
            If init_c does not contain correct shape.
        """
        return self._init_c

    @init_c.setter
    def init_c(self, init_c: int | float | list | np.ndarray) -> None:
        if isinstance(init_c, (int, float)):
            init_c = init_c * np.ones((self.n_compartments, self.n_comp))
        elif isinstance(init_c, list) and len(init_c) == self.n_comp:
            init_c = np.tile(init_c, (self.n_compartments, 1))
        elif isinstance(init_c, np.ndarray) and init_c.shape == (
            self.n_compartments,
            self.n_comp,
        ):
            pass
        else:
            raise ValueError("unexpected value for init_c")

        self._init_c = init_c

        for i in range(self.n_compartments):
            compartment = self.flow_sheet[f"compartment_{i}"]
            if not isinstance(compartment, Outlet):
                compartment.c = init_c[i, :].tolist()

    def add_tracer(
        self,
        compartment_index: int,
        c: list,
        t_inj: float,
        flow_rate: Optional[float],
        t_start: Optional[float] = 0,
        flow_rate_filter: Optional[bool] = True,
    ) -> None:
        """
        Add tracer injection to compartment model.

        For this purpose, a new inlet source is instantiated and connected to
        the corresponding compartment. Then, an Event is added which modifies
        the flow rate of the inlet.

        Parameters
        ----------
        compartment_index : int
            Compartment to which tracer is injected.
        c : list
            Tracer concentration.
        t_inj : float
            Length of injection.
        flow_rate : float
            flow rate during injection.
        t_start : float, optional
            Time at which injection starts. The default is 0.
        flow_rate_filter : bool, optional
            If True, the compartment volume is kept constant by adding a flow rate filter.
            The default is True.

        Raises
        ------
        CADETProcessError
            If compartment is not a real compartment.
        """
        tracer = Inlet(self.component_system, "tracer")
        tracer.flow_rate = flow_rate

        compartment_unit = self.flow_sheet[f"compartment_{compartment_index}"]
        if compartment_unit not in self._real_compartments:
            raise CADETProcessError("Tracer must connect to real compartment")

        if flow_rate_filter:
            compartment_unit.flow_rate_filter = flow_rate

        self.flow_sheet.add_unit(tracer)
        self.flow_sheet.add_connection(tracer, compartment_unit)

        self.process.add_event("tracer_on", "flow_sheet.tracer.c", c, t_start)
        self.process.add_event("tracer_off", "flow_sheet.tracer.c", 0, t_start + t_inj)

    def validate_flow_rates(self) -> None:
        """Validate that compartment volume is constant."""
        flow_rates = self.flow_sheet.get_flow_rates()

        for comp in self._real_compartments:
            for port in flow_rates[comp.name].total_in:
                if not np.all(
                    np.isclose(
                        flow_rates[comp.name].total_in[port],
                        flow_rates[comp.name].total_out[port],
                    )
                ):
                    if comp.n_ports == 1:
                        msg = comp.name
                    else:
                        msg = f"{comp.name} at Port {port}"
                    raise CADETProcessError(
                        f"Unbalanced flow rate for compartment '{msg}'."
                    )


class CompartmentModel(Cstr):
    """Dummy Class for checking binding and reaction models."""

    pass
