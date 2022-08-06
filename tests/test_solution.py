"""

To do
=====

## SolutionBase
- [x] Total concentration
- [x] Total concentration components
- [x] Local Purity
- [x] Local Purity Components
- [x] Purity with components excluded

## SolutionIO
- [x] Mass Flow
- [x] Fraction Mass
- [x] Fraction Volume
- [ ] Interpolation
- [ ] Resampling
- [ ] Normalization
- [ ] Smoothing
- [ ] Derivative
- [ ] plotting?

Notes
-----
To facilitate trouble shooting, make use of built-in plot functions.
They are available for the Solution, as well as the TimeLine objects.

"""

import unittest

import numpy as np
from scipy import stats

from CADETProcess.solution import SolutionIO
from CADETProcess.processModel import ComponentSystem
from CADETProcess.dynamicEvents import TimeLine, Section


class TestSolution(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        comp_2 = ComponentSystem(2)

        comp_2_3_species = ComponentSystem()
        comp_2_3_species.add_component('A')
        comp_2_3_species.add_component('B', species=['B+', 'B-'])

        comp_3_excluded = ComponentSystem(
            3, exclude_from_purity=['0']
        )

        # Note that signal is recorded at 10 Hz.
        time = np.linspace(0, 100, 1001)

        solution_2_constant = np.ones((len(time), 2))
        solution_2_constant[:, 1] *= 2

        solution_2_square = np.zeros((len(time), 2))
        solution_2_square[200:400, 0] = 2
        solution_2_square[300:500, 1] = 1

        solution_2_gaussian = np.zeros((len(time), 2))
        mu_0 = 30
        sigma_0 = 5
        solution_2_gaussian[:, 0] = stats.norm.pdf(time, mu_0, sigma_0)
        mu_1 = 40
        sigma_1 = 5
        solution_2_gaussian[:, 1] = stats.norm.pdf(time, mu_1, sigma_1)

        solution_3_linear = np.zeros((len(time), 3))

        solution_3_linear[100:201, 0] = 1
        solution_3_linear[200:401, 0] = np.linspace(1, 0, 201)

        solution_3_linear[200:401, 1] = np.linspace(0, 1, 201)
        solution_3_linear[400:601, 1] = np.linspace(1, 0, 201)

        solution_3_linear[300:501, 2] = np.linspace(0, 1, 201)
        solution_3_linear[500:701, 2] = np.linspace(1, 0, 201)

        q_const = np.ones(time.shape)

        q_interrupted = TimeLine()
        q_interrupted.add_section(Section(0, 30, 1, n_entries=1, degree=3))
        q_interrupted.add_section(Section(30, 40, 0, n_entries=1, degree=3))
        q_interrupted.add_section(Section(40, 100, 1, n_entries=1, degree=3))

        q_linear = TimeLine()
        q_linear.add_section(Section(0, 35, 0, n_entries=1, degree=3))
        q_linear.add_section(Section(35, 45, [0, 1/10], n_entries=1, degree=3))
        q_linear.add_section(Section(45, 100, 0, n_entries=1, degree=3))

        # 2 Components, constant concentration, constant flow
        self.solution_constant = SolutionIO(
            'simple', comp_2, time, solution_2_constant,
            q_const
        )

        # 2 Components, square peaks, constant flow
        self.solution_square = SolutionIO(
            'simple', comp_2, time, solution_2_square,
            q_const
        )

        # 3 Components (species), linear peaks, constant flow
        self.solution_species = SolutionIO(
            'simple', comp_2_3_species, time, solution_3_linear,
            q_const
        )

        # 3 Components (excluded), linear peaks, constant flow
        self.solution_excluded = SolutionIO(
            'simple', comp_3_excluded, time, solution_3_linear,
            q_const
        )

        # 2 Components, constant concentration, interrupted flow
        self.solution_constant_interrupted = SolutionIO(
            'simple', comp_2, time, solution_2_constant,
            q_interrupted
        )

        # 2 Components, constant concentration, linear flow
        self.solution_constant_linear = SolutionIO(
            'simple', comp_2, time, solution_2_constant,
            q_linear
        )

        # 2 Components, square peaks, interrupted flow
        self.solution_square_interrupted = SolutionIO(
            'simple', comp_2, time, solution_2_square,
            q_interrupted
        )

        # 2 Components, square peaks, linear flow
        self.solution_square_linear = SolutionIO(
            'simple', comp_2, time, solution_2_square,
            q_linear
        )

        # 2 Components, gaussian peaks, constant flow
        self.solution_gaussian = SolutionIO(
            'simple', comp_2, time, solution_2_gaussian,
            q_const
        )

        # 2 Components, gaussian peaks, interrupted flow
        self.solution_gaussian_interrupted = SolutionIO(
            'simple', comp_2, time, solution_2_gaussian,
            q_interrupted
        )

        # 2 Components, gaussian peaks, linear flow
        self.solution_gaussian_linear = SolutionIO(
            'simple', comp_2, time, solution_2_gaussian,
            q_linear
        )

    def test_total_concentration(self):
        # Simple case
        c_total_expected = 3
        c_total = self.solution_constant.total_concentration
        np.testing.assert_almost_equal(c_total, c_total_expected)

        # 2 Components, square peaks
        c_total = self.solution_square.total_concentration

        # Note that here t is not the acutal time but rather the index.
        t = 0
        c_total_expected = [0]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 250
        c_total_expected = [2]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 350
        c_total_expected = [3]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 450
        c_total_expected = [1]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 550
        c_total_expected = [0]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        # 2 Components, species
        c_total = self.solution_species.total_concentration

        t = 150
        c_total_expected = [1]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 300
        c_total_expected = [1]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 350
        c_total_expected = [1.25]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 450
        c_total_expected = [1.5]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 550
        c_total_expected = [1]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

        t = 650
        c_total_expected = [0.25]
        c_total_actual = c_total[t]
        np.testing.assert_almost_equal(c_total_actual, c_total_expected)

    def test_total_concentration_components(self):
        # 2 Components, species
        c_total_comp = self.solution_species.total_concentration_components

        t = 150
        c_total_comp_expected = [1, 0]
        c_total_comp_actual = c_total_comp[t]
        np.testing.assert_almost_equal(
            c_total_comp_actual, c_total_comp_expected
        )

        t = 300
        c_total_comp_expected = [0.5, 0.5]
        c_total_comp_actual = c_total_comp[t]
        np.testing.assert_almost_equal(
            c_total_comp_actual, c_total_comp_expected
        )

        t = 350
        c_total_comp_expected = [0.25, 1]
        c_total_comp_actual = c_total_comp[t]
        np.testing.assert_almost_equal(
            c_total_comp_actual, c_total_comp_expected
        )

        t = 450
        c_total_comp_expected = [0, 1.5]
        c_total_comp_actual = c_total_comp[t]
        np.testing.assert_almost_equal(
            c_total_comp_actual, c_total_comp_expected
        )

    def test_local_purity_components(self):
        # Simple case
        local_purity_expected = [1/3, 2/3]
        local_purity_actual = self.solution_constant.local_purity_components[0]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        # 2 Components, square peaks
        local_purity = self.solution_square.local_purity_components

        t = 0
        local_purity_expected = [0, 0]
        local_purity_actual = local_purity[t]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        t = 250
        local_purity_expected = [1, 0]
        local_purity_actual = local_purity[t]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        t = 350
        local_purity_expected = [2/3, 1/3]
        local_purity_actual = local_purity[t]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        t = 450
        local_purity_expected = [0, 1]
        local_purity_actual = local_purity[t]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        t = 550
        local_purity_expected = [0, 0]
        local_purity_actual = local_purity[t]
        np.testing.assert_almost_equal(
            local_purity_actual, local_purity_expected
        )

        # 2 Components, 3 species (summed up)
        local_purity_components = self.solution_species.local_purity_components

        t = 150
        local_purity_components_expected = [1, 0]
        local_purity_components_actual = local_purity_components[t]
        np.testing.assert_almost_equal(
            local_purity_components_actual, local_purity_components_expected
        )

        t = 300
        local_purity_components_expected = [0.5, 0.5]
        local_purity_components_actual = local_purity_components[t]
        np.testing.assert_almost_equal(
            local_purity_components_actual, local_purity_components_expected
        )

        t = 350
        local_purity_components_expected = [0.2, 0.8]
        local_purity_components_actual = local_purity_components[t]
        np.testing.assert_almost_equal(
            local_purity_components_actual, local_purity_components_expected
        )

        t = 450
        local_purity_components_expected = [0, 1]
        local_purity_components_actual = local_purity_components[t]
        np.testing.assert_almost_equal(
            local_purity_components_actual, local_purity_components_expected
        )

    def test_local_purity_species(self):
        # 2 Components, 3  species
        local_purity_species = self.solution_species.local_purity_species

        t = 150
        local_purity_species_expected = [1, 0, 0]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 300
        local_purity_species_expected = [0.5, 0.5, 0]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 350
        local_purity_species_expected = [0.2, 0.6, 0.2]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 450
        local_purity_species_expected = [0, 0.5, 0.5]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

    def test_local_purity_excluded(self):
        # 2 Components, 3  species
        local_purity_species = self.solution_excluded.local_purity_species

        t = 150
        local_purity_species_expected = [0, 0, 0]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 300
        local_purity_species_expected = [0, 1, 0]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 350
        local_purity_species_expected = [0, 0.75, 0.25]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

        t = 450
        local_purity_species_expected = [0, 0.5, 0.5]
        local_purity_species_actual = local_purity_species[t]
        np.testing.assert_almost_equal(
            local_purity_species_actual, local_purity_species_expected
        )

    def test_resampling(self):
        pass

    def test_normalization(self):
        pass

    def test_smoothing(self):
        pass

    def test_fraction_volume(self):
        volume_expected = 0
        volume_actual = self.solution_constant.fraction_volume(0, 0)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 10
        volume_actual = self.solution_constant.fraction_volume(0, 10)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 0
        volume_actual = self.solution_constant_interrupted.fraction_volume(30, 40)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 20
        volume_actual = self.solution_constant_interrupted.fraction_volume(20, 50)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 0
        volume_actual = self.solution_constant_linear.fraction_volume(0, 10)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 5
        volume_actual = self.solution_constant_linear.fraction_volume(25, 45)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

        volume_expected = 1.25
        volume_actual = self.solution_constant_linear.fraction_volume(30, 40)
        np.testing.assert_almost_equal(volume_actual, volume_expected)

    def test_fraction_mass(self):
        mass_expected = [10, 20]
        mass_actual = self.solution_constant.fraction_mass(0, 10)
        np.testing.assert_almost_equal(mass_actual, mass_expected)

        mass_expected = [10, 20]
        mass_actual = self.solution_constant.fraction_mass(0, 10)
        np.testing.assert_almost_equal(mass_actual, mass_expected)

        mass_expected = np.array([
            np.diff(stats.norm.cdf([-2, 2]))[0],
            np.diff(stats.norm.cdf([-4, 0]))[0]
        ])
        mass_actual = self.solution_gaussian.fraction_mass(20, 40)
        np.testing.assert_almost_equal(mass_actual, mass_expected)

        mass_expected = np.array([
            np.diff(stats.norm.cdf([-2, 0]))[0],
            np.diff(stats.norm.cdf([-4, -2]))[0]
        ])
        mass_actual = self.solution_gaussian_interrupted.fraction_mass(20, 40)
        np.testing.assert_almost_equal(mass_actual, mass_expected)


if __name__ == '__main__':
    unittest.main()