import unittest

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Langmuir, BiLangmuir, MultistateStericMassAction
)


class Test_Binding(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):

        component_system = ComponentSystem(2)

        binding_model = Langmuir(component_system, name='test')

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.capacity = [100, 100]

        self.langmuir = binding_model

        binding_model = BiLangmuir(component_system, name='test')

        binding_model.adsorption_rate = [0.02, 0.03, 0.001, 0.002]
        binding_model.desorption_rate = [1, 1, 2, 2]
        binding_model.capacity = [100, 100, 200, 200]

        self.bi_langmuir = binding_model

        binding_model = MultistateStericMassAction(
            component_system, name='test'
        )

        self.multi_state_sma = binding_model

    def test_get_parameters(self):
        parameters_expected = {
                'is_kinetic': True,
                'adsorption_rate': [0.02, 0.03],
                'desorption_rate': [1.0, 1.0],
                'capacity': [100.0, 100.0]
                }
        parameters = self.langmuir.parameters
        self.assertDictEqual(parameters_expected, parameters)

    def test_binding_sites(self):
        self.assertEqual(self.bi_langmuir.n_binding_sites, 2)
        self.assertEqual(self.bi_langmuir.n_bound_states, 4)

        self.bi_langmuir.n_binding_sites = 3
        self.assertEqual(self.bi_langmuir.n_binding_sites, 3)
        self.assertEqual(self.bi_langmuir.n_bound_states, 6)

        with self.assertRaises(ValueError):
            self.langmuir.n_binding_sites = 2

    def test_bound_states(self):
        self.assertEqual(self.multi_state_sma.n_bound_states, 2)

        self.multi_state_sma.bound_states = [3, 2]
        self.assertEqual(self.multi_state_sma.n_bound_states, 5)

        with self.assertRaises(ValueError):
            self.multi_state_sma.bound_states = [1, 2, 3]

        with self.assertRaises(ValueError):
            self.langmuir.bound_states = [2, 2]


if __name__ == '__main__':
    unittest.main()
