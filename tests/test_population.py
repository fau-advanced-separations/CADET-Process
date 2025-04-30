import unittest

import numpy as np
from CADETProcess import CADETProcessError
from CADETProcess.optimization import Individual, ParetoFront, Population

from tests.test_individual import setup_individual

enable_plot = False


def setup_population(n_ind, n_vars, n_obj, n_nonlin=0, n_meta=0, rng=None):
    population = Population()

    if rng is None:
        rng = np.random.default_rng(12345)

    for i in range(n_ind):
        ind = setup_individual(n_vars, n_obj, n_nonlin, n_meta, rng)
        population.add_individual(ind)

    return population


class TestPopulation(unittest.TestCase):
    def setUp(self):
        x = [1, 2]
        f = [-1]
        self.individual_1 = Individual(x, f=f)

        x = [2, 3]
        f = [-2]
        self.individual_2 = Individual(x, f=f)

        x = [1.001, 2]
        f = [-1.001]
        self.individual_similar = Individual(x, f=f)

        x = [1, 2]
        f = [-1, -2]
        self.individual_multi_1 = Individual(x, f=f)

        x = [1.001, 2]
        f = [-1.001, -2]
        self.individual_multi_2 = Individual(x, f=f)

        x = [1, 2]
        f = [-1]
        g = [3]
        self.individual_constr_1 = Individual(x, f=f, g=g)

        x = [2, 3]
        f = [-2]
        g = [0]
        self.individual_constr_2 = Individual(x, f=f, g=g)

        x = [2, 3]
        f = [-2, -2]
        g = [0]
        m = [-4]
        self.individual_multi_meta_1 = Individual(x, f=f, m=m)

        self.population = Population()
        self.population.add_individual(self.individual_1)
        self.population.add_individual(self.individual_2)
        self.population.add_individual(self.individual_similar)

        self.population_multi = Population()
        self.population_multi.add_individual(self.individual_multi_1)
        self.population_multi.add_individual(self.individual_multi_2)

        self.population_constr = Population()
        self.population_constr.add_individual(self.individual_constr_1)
        self.population_constr.add_individual(self.individual_constr_2)

        self.population_meta = Population()
        self.population_meta.add_individual(self.individual_multi_meta_1)

    def test_dimensions(self):
        dimensions_expected = (2, 1, 0, 0)
        dimensions = self.population.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        self.assertEqual(2, self.population.n_x)
        self.assertEqual(1, self.population.n_f)
        self.assertEqual(0, self.population.n_g)
        self.assertEqual(0, self.population.n_m)

        dimensions_expected = (2, 1, 1, 0)
        dimensions = self.population_constr.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        dimensions_expected = (2, 1, 1, 0)
        dimensions = self.population_constr.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        self.assertEqual(2, self.population_meta.n_x)
        self.assertEqual(2, self.population_meta.n_f)
        self.assertEqual(0, self.population_meta.n_g)
        self.assertEqual(1, self.population_meta.n_m)

    def test_values(self):
        x_expected = np.array(
            [
                [1, 2],
                [2, 3],
                [1.001, 2],
            ]
        )
        x = self.population.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array(
            [
                [-1],
                [-2],
                [-1.001],
            ]
        )
        f = self.population.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array(
            [
                [3],
                [0],
            ]
        )
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

    def test_add_remove(self):
        with self.assertRaises(TypeError):
            self.population.add_individual("foo")

        self.population.add_individual(self.individual_1, ignore_duplicate=True)

        with self.assertRaises(CADETProcessError):
            self.population.add_individual(self.individual_1, ignore_duplicate=False)

        new_individual = Individual([9, 10], f=[3], g=[-1])
        with self.assertRaises(CADETProcessError):
            self.population.add_individual(new_individual)

        self.population_constr.add_individual(new_individual)

        self.assertFalse(new_individual in self.population)
        self.assertTrue(self.individual_1 in self.population_constr)

        x_expected = np.array(
            [
                [1, 2],
                [2, 3],
                [9, 10],
            ]
        )
        x = self.population_constr.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array(
            [
                [-1],
                [-2],
                [3],
            ]
        )
        f = self.population_constr.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array(
            [
                [3],
                [0],
                [-1],
            ]
        )
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

        with self.assertRaises(TypeError):
            self.population.remove_individual("foo")

        with self.assertRaises(CADETProcessError):
            self.population.remove_individual(new_individual)

        self.population_constr.remove_individual(new_individual)
        x_expected = np.array(
            [
                [1, 2],
                [2, 3],
            ]
        )
        x = self.population_constr.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array(
            [
                [-1],
                [-2],
            ]
        )
        f = self.population_constr.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array(
            [
                [3],
                [0],
            ]
        )
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

    def test_min_max(self):
        f_min_expected = [-2]
        f_min = self.population_constr.f_min
        np.testing.assert_almost_equal(f_min, f_min_expected)

        f_max_expected = [-1]
        f_max = self.population_constr.f_max
        np.testing.assert_almost_equal(f_max, f_max_expected)

        g_min_expected = [0]
        g_min = self.population_constr.g_min
        np.testing.assert_almost_equal(g_min, g_min_expected)

        g_max_expected = [3]
        g_max = self.population_constr.g_max
        np.testing.assert_almost_equal(g_max, g_max_expected)

    def test_plot(self):
        if enable_plot:
            pass

    def test_to_dict(self):
        # Test that Population can be converted to a dictionary
        population_dict = self.population.to_dict()
        individuals_list = population_dict["individuals"]
        self.assertEqual(len(individuals_list), 3)
        self.assertEqual(population_dict["id"], str(self.population.id))

    def test_from_dict(self):
        # Test that a Population can be created from a dictionary
        population_dict = self.population.to_dict()
        new_population = Population.from_dict(population_dict)
        self.assertEqual(new_population.id, self.population.id)
        self.assertEqual(len(new_population), len(self.population))

        self.assertTrue(self.individual_1 in new_population)


class TestPareto(unittest.TestCase):
    def setUp(self):
        front = ParetoFront(3)

        population = [
            Individual([1, 2], [1, 2, 3]),
        ]

        front.update(population)

        population = [
            Individual([1, 2.01], [1, 2, 3.01]),
            Individual([4, 5], [4, 5, 6]),
        ]

        front.update(population)


if __name__ == "__main__":
    enable_plot = True

    unittest.main()
