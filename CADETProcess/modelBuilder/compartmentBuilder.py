import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import StructMeta

from CADETProcess.processModel import FlowSheet, Process
from CADETProcess.processModel import Cstr, Source, Sink

class CompartmentBuilder(metaclass=StructMeta):
    def __init__(
            self, component_system, name,
            compartment_volumes, connections_matrix,
            cycle_time,
            init_c=0
        ):
        self.component_system = component_system
        self.name = name
        self._flow_sheet = FlowSheet(component_system, name)
        self._process = Process(self._flow_sheet, name)
        self.process.cycle_time = cycle_time

        self._add_compartments(compartment_volumes, init_c)
        self._add_connections(connections_matrix)

    @property
    def n_comp(self):
        return self.component_system.n_comp

    @property
    def flow_sheet(self):
        return self._flow_sheet

    @property
    def process(self):
        return self._process

    def _add_compartments(self, compartment_volumes, init_c=0):
        self.n_compartments = len(compartment_volumes)

        if isinstance(init_c, (int, float)):
            init_c = init_c * np.ones((self.n_compartments, self.n_comp))
        elif isinstance(init_c, np.ndarray) \
            and init_c.shape == (self.n_compartments, self.n_comp):
            pass
        else:
            raise CADETProcessError("unexpected value for init_c")

        for i, (vol, c) in enumerate(zip(compartment_volumes, init_c)):
            name = f'compartment_{i}'

            if vol == 'inlet':
                unit = Source(self.component_system, name)
            elif vol == 'outlet':
                unit = Sink(self.component_system, name)
            else:
                unit = Cstr(self.component_system, name)
                unit.volume = vol
                unit.c = c.tolist()

            self.flow_sheet.add_unit(unit)

    def _add_connections(self, connections_matrix):
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

            origin = self.flow_sheet[f'compartment_{iOrigin}']
            output_state = []

            for iDestination, flow in enumerate(destinations):
                if flow == 0:
                    continue

                destination = self.flow_sheet[f'compartment_{iDestination}']
                output_state.append(flow/flow_rate)

                self.flow_sheet.add_connection(origin, destination)

            if isinstance(origin, Sink):
                continue

            self.flow_sheet[origin.name].flow_rate = flow_rate
            self.flow_sheet.set_output_state(origin, output_state)

    def add_tracer(self, compartment_index, c, t, flow_rate):
        tracer = Source(self.component_system, 'tracer')
        tracer.flow_rate = flow_rate

        compartment_unit = self.flow_sheet[f'compartment_{compartment_index}']

        compartment_unit.flow_rate_filter = flow_rate

        self.flow_sheet.add_unit(tracer)
        self.flow_sheet.add_connection(tracer, compartment_unit)

        self.process.add_event('tracer_on', 'flow_sheet.tracer.c', c, 0)
        self.process.add_event('tracer_off', 'flow_sheet.tracer.c', 0, t)

    def validate_flow_rates(self):
        flow_rates = self.flow_sheet.get_flow_rates()

        for iComp in range(self.n_compartments):
            unit = f'compartment_{iComp}'
            if isinstance(self.flow_sheet[unit], (Source, Sink)):
                continue
            if not np.all(
                    np.isclose(
                        flow_rates[unit].total_in, flow_rates[unit].total_out
                    )
                ):
                raise CADETProcessError(
                    f"Unbalanced flow rate for unit '{unit}'."
                )

    def build_process(self):
        process = Process(self.flow_sheet, self.name)

        return process
