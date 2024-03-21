import unittest

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Inlet, Outlet, MCT
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

from CADETProcess.simulator import Cadet

import numpy as np

class TestMCT(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.setup_mct_flow_sheet()

    def setup_mct_flow_sheet(self):
        self.component_system = ComponentSystem(1)

        mct_flow_sheet = FlowSheet(self.component_system)

        inlet = Inlet(self.component_system, name='inlet')
        mct_3c = MCT(self.component_system,nchannel=3, name='mct_3c')
        mct_2c1 = MCT(self.component_system,nchannel=2, name='mct_2c1')
        mct_2c2 = MCT(self.component_system,nchannel=2, name='mct_2c2')
        outlet1 = Outlet(self.component_system, name='outlet1')
        outlet2 = Outlet(self.component_system, name='outlet2')

        mct_flow_sheet.add_unit(inlet)
        mct_flow_sheet.add_unit(mct_3c)
        mct_flow_sheet.add_unit(mct_2c1)
        mct_flow_sheet.add_unit(mct_2c2)
        mct_flow_sheet.add_unit(outlet1)
        mct_flow_sheet.add_unit(outlet2)

        mct_flow_sheet.add_connection(inlet, mct_3c, destination_port='channel_0')
        mct_flow_sheet.add_connection(mct_3c, mct_2c1, origin_port='channel_0', destination_port='channel_0')
        mct_flow_sheet.add_connection(mct_3c, mct_2c1, origin_port='channel_0', destination_port='channel_1')
        mct_flow_sheet.add_connection(mct_3c, mct_2c2, origin_port='channel_1', destination_port='channel_0')
        mct_flow_sheet.add_connection(mct_2c1, outlet1, origin_port='channel_0')
        mct_flow_sheet.add_connection(mct_2c1, outlet1, origin_port='channel_1')
        mct_flow_sheet.add_connection(mct_2c2, outlet2, origin_port='channel_1')

        self.mct_flow_sheet = mct_flow_sheet
    
    def test_simulation(self):

        mct_flow_sheet = self.mct_flow_sheet
        process = Process(mct_flow_sheet, 'Transport')
        process.cycle_time = 120

        process.add_event('Start', 'flow_sheet.inlet.c', [1], 0)
        process.add_event('Stop', 'flow_sheet.inlet.c', [0], 1*60)

        inlet = self.mct_flow_sheet['inlet']
        mct_3c = self.mct_flow_sheet['mct_3c']
        mct_2c1 = self.mct_flow_sheet['mct_2c1']
        mct_2c2 = self.mct_flow_sheet['mct_2c2']

        inlet.flow_rate = 1e-6

        mct_3c.discretization.ncol=120
        mct_3c.length = 30
        mct_3c.channel_cross_section_areas = [1,1,1]
        mct_3c.axial_dispersion = 0

        mct_3c.exchange_matrix = np.array([[
            [0.0,0.01,0.0],
            [0.02,0.0,0.03],
            [0.0,0.0,0.0]
        ]])

        mct_2c1.discretization.ncol=120
        mct_2c1.length = 30
        mct_2c1.channel_cross_section_areas = [1,1]
        mct_2c1.axial_dispersion = 0

        mct_2c1.exchange_matrix = np.array([[
            [0.0,0.01],
            [0.0,0.0],

        ]])

        mct_2c2.discretization.ncol=120
        mct_2c2.length = 30
        mct_2c2.channel_cross_section_areas = [1,1]
        mct_2c2.axial_dispersion = 0

        mct_2c2.exchange_matrix = np.array([[
            [0.0,0.01],
            [0.02,0.0],
        ]])
        mct_3c.solution_recorder.write_solution_bulk = 1
        mct_2c2.solution_recorder.write_solution_bulk = 1        
        process_simulator = Cadet(install_path='c:\\Users\dklau\Documents\Arbeitsprojekte\ModSim\cadet\CADET\out\install\\aRelease')
        self.assertTrue(process_simulator.check_cadet())
        simulation_results = process_simulator.simulate(process)

        simulation_results.solution







if __name__ == '__main__':
    unittest.main()
