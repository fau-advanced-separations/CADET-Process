import unittest

class Test_Binding(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)

    def create_binding(self):
        import CADETProcess
        binding_model = CADETProcess.processModel.Langmuir(n_comp=2, name='test')

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.saturation_capacity = [100, 100]

        return binding_model

    def test_setter(self):
        """
        Note
        ----
        AssertRaises tests actually test dataStructure descriptor functionality
        could/should be moved to dedicated test Class
        """
        binding_model = self.create_binding()

        with self.assertRaises(TypeError):
            binding_model.adsorption_rate = 1
        with self.assertRaises(ValueError):
            binding_model.adsorption_rate = [1]
        with self.assertRaises(ValueError):
            binding_model.adsorption_rate = [-1, -1]

    def test_get_parameters(self):
        binding_model = self.create_binding()

        parameters_expected = {
                'is_kinetic': True,
                'adsorption_rate': [0.02, 0.03],
                'desorption_rate': [1.0, 1.0],
                'saturation_capacity': [100.0, 100.0]
                }

        self.assertDictEqual(parameters_expected, binding_model.parameters)


if __name__ == '__main__':
    unittest.main()