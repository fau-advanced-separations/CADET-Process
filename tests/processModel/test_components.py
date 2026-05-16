import unittest

import numpy as np
from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem


class TestComponents(unittest.TestCase):
    def setUp(self):
        self.component_system_0 = ComponentSystem(2)

        self.component_system_1 = ComponentSystem(["A", "B"])

        self.component_system_2 = ComponentSystem()
        self.component_system_2.add_component("A")
        self.component_system_2.add_component("B", species=["B+", "B-"])

        self.component_system_3 = ComponentSystem()
        self.component_system_3.add_component(
            "Ammonia",
            species=["NH4+", "NH3"],
            charge=[1, 0],
        )
        self.component_system_3.add_component(
            "Lysine", ["Lys2+", "Lys+", "Lys", "Lys"], [2, 1, 0, -1]
        )
        self.component_system_3.add_component(
            "H+",
            charge=1,
        )

        self.component_system_4 = ComponentSystem(2)
        self.component_system_4.add_component("manual_label")

        self.component_system_5 = ComponentSystem()
        self.component_system_5.add_component(
            "A", species=["A+", "A-"], molecular_weight=[1, 0]
        )

        self.component_system_6 = ComponentSystem()
        self.component_system_6.add_component("A", species=["A+", "A-"], density=[1, 0])

    def test_names(self):
        names_expected = ["0", "1"]
        names = self.component_system_0.names
        np.testing.assert_equal(names, names_expected)

        names_expected = ["A", "B"]
        names = self.component_system_1.names
        np.testing.assert_equal(names, names_expected)

        names_expected = ["A", "B"]
        names = self.component_system_2.names
        np.testing.assert_equal(names, names_expected)

        names_expected = ["Ammonia", "Lysine", "H+"]
        names = self.component_system_3.names
        np.testing.assert_equal(names, names_expected)

        names_expected = ["0", "1", "manual_label"]
        names = self.component_system_4.names
        np.testing.assert_equal(names, names_expected)

    def test_duplicate_name(self):
        with self.assertRaises(CADETProcessError):
            self.component_system_1.add_component("A")

    def test_species(self):
        species_expected = ["0", "1"]
        species = self.component_system_0.species
        np.testing.assert_equal(species, species_expected)

        species_expected = ["A", "B"]
        species = self.component_system_1.species
        np.testing.assert_equal(species, species_expected)

        species_expected = ["A", "B+", "B-"]
        species = self.component_system_2.species
        np.testing.assert_equal(species, species_expected)

        species_expected = ["NH4+", "NH3", "Lys2+", "Lys+", "Lys", "Lys", "H+"]
        species = self.component_system_3.species
        np.testing.assert_equal(species, species_expected)

        species_expected = ["0", "1", "manual_label"]
        species = self.component_system_4.species
        np.testing.assert_equal(species, species_expected)

    def test_indices(self):
        indices_expected = {
            "Ammonia": [0, 1],
            "Lysine": [2, 3, 4, 5],
            "H+": [6],
        }
        indices = self.component_system_3.indices
        np.testing.assert_equal(indices, indices_expected)

    def test_n_comp(self):
        n_components_expected = 2
        n_components = self.component_system_2.n_components
        self.assertEqual(n_components_expected, n_components)

        n_comp_expected = 3
        n_comp = self.component_system_2.n_comp
        self.assertEqual(n_comp_expected, n_comp)

        n_components_expected = 3
        n_components = self.component_system_3.n_components
        self.assertEqual(n_components_expected, n_components)

        n_comp_expected = 7
        n_comp = self.component_system_3.n_comp
        self.assertEqual(n_comp_expected, n_comp)

    def test_charge(self):
        charges_expected = [1, 0, 2, 1, 0, -1, 1]
        charges = self.component_system_3.charges
        np.testing.assert_equal(charges_expected, charges)

    def test_molecular_weights(self):
        molecular_weights_expected = [1, 0]
        molecular_weights = self.component_system_5.molecular_weights
        np.testing.assert_equal(molecular_weights_expected, molecular_weights)

    def test_densities(self):
        densities_expected = [1, 0]
        densities = self.component_system_6.densities
        np.testing.assert_equal(densities_expected, densities)


if __name__ == "__main__":
    unittest.main()
