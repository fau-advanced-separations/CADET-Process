from CADETProcess.processModel import Inlet, Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

class BatchElutionFlowSheet(FlowSheet):
    pass


class BatchElutionFlowSheetBuilder():
    def __init__(self, column, inlets):
        # Assert column type
        self.component_system = column.component_system

        self.flow_sheet = FlowSheet(self.component_system)

        self.flow_sheet.add_unit(column)

        for inlet in inlets:
            self.flow_sheet.add_unit(inlet)
            self.flow_sheet.add_connection(inlet, column)

        outlet = Outlet('outlet', self.component_system)
        self.flow_sheet.add_connection(column, outlet)


class BatchElutionProcessBuilder():
    def __init__(self, flow_sheet, t_cycle, t_inj, c_inj):
        self.process = Process(flow_sheet, 'batch_elution')

        # Add events. --> How to assign c_inj / t_inj to inlets?
