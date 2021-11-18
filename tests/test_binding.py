import unittest

import CADETProcess
from CADETProcess.processModel import ComponentSystem, Langmuir

class Test_Binding(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)

    def setUp(self):
        
        component_system = ComponentSystem()
        component_system.add_component('A')
        component_system.add_component('B')
        
        binding_model = Langmuir(component_system, name='test')

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.saturation_capacity = [100, 100]

        self.binding_model = binding_model

    def test_setter(self):
        """
        Note
        ----
        AssertRaises tests actually test dataStructure descriptor functionality
        could/should be moved to dedicated test Class
        """
        with self.assertRaises(TypeError):
            self.binding_model.adsorption_rate = 1
        with self.assertRaises(ValueError):
            self.binding_model.adsorption_rate = [1]
        with self.assertRaises(ValueError):
            self.binding_model.adsorption_rate = [-1, -1]

    def test_get_parameters(self):
        parameters_expected = {
                'is_kinetic': True,
                'adsorption_rate': [0.02, 0.03],
                'desorption_rate': [1.0, 1.0],
                'saturation_capacity': [100.0, 100.0]
                }
        parameters = self.binding_model.parameters
        self.assertDictEqual(parameters_expected, parameters)
        

if __name__ == '__main__':
    unittest.main()