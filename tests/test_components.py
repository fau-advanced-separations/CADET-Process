import unittest
import numpy as np

from CADETProcess.processModel import ComponentSystem

class TestComponents(unittest.TestCase):
    def setUp(self):
        self.component_system = ComponentSystem()
        self.component_system.add_component(
            'Ammonia', 
            species=['NH4+', 'NH3'],
            charges=[1,0]
        )
        self.component_system.add_component(
            'Lysine', 
            ['Lys2+', 'Lys+', 'Lys', 'Lys'],
            [2,1,0,-1]
        )
        self.component_system.add_component(
            'H+', 
            charges=[1]
        )
        self.component_system.add_component()


    def test_labels(self):
        labels_expected = [
            'NH4+', 'NH3', 'Lys2+', 'Lys+', 'Lys', 'Lys', 'H+', '7'
        ]
        labels = self.component_system.labels
        np.testing.assert_equal(labels, labels_expected)
        
    def test_n_comp(self):
        n_comp_expected = 8
        n_comp = self.component_system.n_comp
        self.assertEqual(n_comp_expected, n_comp)


if __name__ == '__main__':
    unittest.main()