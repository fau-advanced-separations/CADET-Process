import unittest

import numpy as np
from CADETProcess import equilibria
from CADETProcess.processModel import ComponentSystem, MassActionLaw

enable_plot = False


class TestBufferCapacity(unittest.TestCase):
    def setUp(self):
        self.components_simple = ComponentSystem(2, charges=[1, 2])

        # Ammonia
        self.components_ammonia = ComponentSystem()
        self.components_ammonia.add_component(
            "Ammonia", species=["NH4+", "NH3"], charge=[1, 0]
        )
        self.components_ammonia.add_component("H+", charge=1)

        self.reaction_ammonia = MassActionLaw(self.components_ammonia)
        self.reaction_ammonia.add_reaction(
            [0, 1, 2], [-1, 1, 1], 10 ** (-9.2) * 1e3, is_kinetic=False
        )

        # Ammonia, component order switched
        self.components_ammonia_switched = ComponentSystem()
        self.components_ammonia_switched.add_component("H+", charge=1)
        self.components_ammonia_switched.add_component(
            "Ammonia", species=["NH4+", "NH3"], charge=[1, 0]
        )

        self.reaction_ammonia_switched = MassActionLaw(self.components_ammonia_switched)
        self.reaction_ammonia_switched.add_reaction(
            [1, 2, 0], [-1, 1, 1], 10 ** (-9.2) * 1e3, is_kinetic=False
        )

        # Lysine
        self.components_lys = ComponentSystem()
        self.components_lys.add_component(
            "Lysine", species=["Lys2+", "Lys+", "Lys", "Lys-"], charge=[2, 1, 0, -1]
        )
        self.components_lys.add_component("H+", charge=1)
        self.reaction_lys = MassActionLaw(self.components_lys)
        self.reaction_lys.add_reaction(
            [0, 1, -1], [-1, 1, 1], 10 ** (-2.20) * 1e3, is_kinetic=False
        )
        self.reaction_lys.add_reaction(
            [1, 2, -1], [-1, 1, 1], 10 ** (-8.90) * 1e3, is_kinetic=False
        )
        self.reaction_lys.add_reaction(
            [2, 3, -1], [-1, 1, 1], 10 ** (-10.28) * 1e3, is_kinetic=False
        )

        # Lysine, component order switched
        self.components_lys_switched = ComponentSystem()
        self.components_lys_switched.add_component("H+", charge=1)
        self.components_lys_switched.add_component(
            "Lysine", species=["Lys2+", "Lys+", "Lys", "Lys-"], charge=[2, 1, 0, -1]
        )

        self.reaction_lys_switched = MassActionLaw(self.components_lys_switched)
        self.reaction_lys_switched.add_reaction(
            [1, 2, 0], [-1, 1, 1], 10 ** (-2.20) * 1e3, is_kinetic=False
        )
        self.reaction_lys_switched.add_reaction(
            [2, 3, 0], [-1, 1, 1], 10 ** (-8.90) * 1e3, is_kinetic=False
        )
        self.reaction_lys_switched.add_reaction(
            [3, 4, 0], [-1, 1, 1], 10 ** (-10.28) * 1e3, is_kinetic=False
        )

        # Ammonia and Lysine
        self.components_ammonia_lys = ComponentSystem()
        self.components_ammonia_lys.add_component(
            "Ammonia", species=["NH4+", "NH3"], charge=[1, 0]
        )
        self.components_ammonia_lys.add_component(
            "Lysine", species=["Lys2+", "Lys+", "Lys", "Lys-"], charge=[2, 1, 0, -1]
        )
        self.components_ammonia_lys.add_component("H+", charge=1)
        self.reaction_ammonia_lys = MassActionLaw(self.components_ammonia_lys)
        self.reaction_ammonia_lys.add_reaction(
            [0, 1, -1], [-1, 1, 1], 10 ** (-9.2) * 1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [2, 3, -1], [-1, 1, 1], 10 ** (-2.20) * 1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [3, 4, -1], [-1, 1, 1], 10 ** (-8.90) * 1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [4, 5, -1], [-1, 1, 1], 10 ** (-10.28) * 1e3, is_kinetic=False
        )

    def test_ionic_strength(self):
        c = [1, 1]
        i_excpected = 2.5
        i = equilibria.ionic_strength(self.components_simple, c)
        np.testing.assert_almost_equal(i, i_excpected)

        c = [1, 2]
        i_excpected = 4.5
        i = equilibria.ionic_strength(self.components_simple, c)
        np.testing.assert_almost_equal(i, i_excpected)

        c = [1, 0, 1]
        i_excpected = 1
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)

        # Uncharged species should have no effect
        c = [1, 1, 1]
        i_excpected = 1
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)

        c = [2, 0, 1]
        i_excpected = 1.5
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)

    def test_buffer_capacity(self):
        buffer = [0, 1000, 0]

        pH = 0
        b_expected = np.array([1.4528329738818527e-06, 2302.585092994069])
        b = equilibria.buffer_capacity(self.reaction_ammonia, buffer, pH)
        np.testing.assert_almost_equal(b_expected, b)

        pH = 7
        b_expected = np.array([14.346715296419855, 0.0004605170185988092])
        b = equilibria.buffer_capacity(self.reaction_ammonia, buffer, pH)
        np.testing.assert_almost_equal(b_expected, b)

        pH = 10
        b_expected = np.array([271.9140324242618, 0.23025873955791384])
        b = equilibria.buffer_capacity(self.reaction_ammonia, buffer, pH)
        np.testing.assert_almost_equal(b_expected, b)

        pH = 14
        b_expected = np.array([0.03649235765053616, 2302.585092994069])
        b = equilibria.buffer_capacity(self.reaction_ammonia, buffer, pH)
        np.testing.assert_almost_equal(b_expected, b)

        pH = 0
        b_expected = np.array([1.4528329738818527e-06, 2302.585092994069])
        b = equilibria.buffer_capacity(self.reaction_ammonia_switched, buffer, pH)
        np.testing.assert_almost_equal(b_expected, b)

    def test_charge_distribution(self):
        pH = 0
        eta_expected = np.array([0.9999999993690426, 6.30957344082087e-10])
        eta = equilibria.charge_distribution(self.reaction_ammonia, pH)
        np.testing.assert_almost_equal(eta_expected, eta)

        pH = 7
        eta_expected = np.array([0.9937299876585661, 0.006270012341433852])
        eta = equilibria.charge_distribution(self.reaction_ammonia, pH)
        np.testing.assert_almost_equal(eta_expected, eta)

        pH = 10
        eta_expected = np.array([0.13680688860320983, 0.8631931113967902])
        eta = equilibria.charge_distribution(self.reaction_ammonia, pH)
        np.testing.assert_almost_equal(eta_expected, eta)

        pH = 14
        eta_expected = np.array([1.5848680739948967e-05, 0.99998415131926])
        eta = equilibria.charge_distribution(self.reaction_ammonia, pH)
        np.testing.assert_almost_equal(eta_expected, eta)

        pH = 0
        eta_expected = np.array([0.9999999993690426, 6.30957344082087e-10])
        eta = equilibria.charge_distribution(self.reaction_ammonia_switched, pH)
        np.testing.assert_almost_equal(eta_expected, eta)

    def test_plot(self):
        if enable_plot:
            _ = equilibria.plot_charge_distribution(self.reaction_ammonia)
            _ = equilibria.plot_charge_distribution(self.reaction_ammonia_switched)
            _ = equilibria.plot_charge_distribution(
                self.reaction_lys_switched, plot_cumulative=True
            )
            _ = equilibria.plot_charge_distribution(
                self.reaction_ammonia_switched, plot_cumulative=True
            )


if __name__ == "__main__":
    enable_plot = True
    unittest.main()
