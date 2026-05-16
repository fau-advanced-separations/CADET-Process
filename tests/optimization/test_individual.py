import unittest

import numpy as np
from CADETProcess.optimization import Individual, hash_array


def setup_individual(n_vars=2, n_obj=1, n_nonlin=0, n_meta=0, rng=None):
    if rng is None:
        rng = np.random.default_rng(12345)

    x = rng.random(n_vars)
    f = -rng.random(n_obj)
    if n_nonlin > 0:
        g = np.random.random(n_nonlin)
    else:
        g = None

    if n_nonlin > 0:
        m = np.sum(f)
    else:
        m = None

    return Individual(x, f=f, g=g, m=m)


class TestHashArray(unittest.TestCase):
    def test_hash_array(self):
        array = np.array([1, 2.0])
        expected_hash = 'dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae'
        self.assertEqual(hash_array(array), expected_hash)


class TestIndividual(unittest.TestCase):
    def setUp(self):
        x = [1, 2]
        f = [-1]
        self.individual_1 = Individual(x, f)

        x = [1, 2]
        f = [-1.001]
        self.individual_2 = Individual(x, f)

        x = [1, 2]
        f = [-1, -2]
        self.individual_multi_1 = Individual(x, f)

        x = [1, 2]
        f = [-1.001, -2]
        self.individual_multi_2 = Individual(x, f)

        x = [1, 2]
        f = [-1, -2.001]
        self.individual_multi_3 = Individual(x, f)

        x = [1, 2]
        f = [-1]
        g = [3]
        self.individual_constr_1 = Individual(x, f, g)

        x = [2, 3]
        f = [-1]
        g = [3]
        self.individual_constr_2 = Individual(x, f, g)

        x = [3, 4]
        f = [-1]
        g = [2]
        self.individual_constr_3 = Individual(x, f, g)

        x = [3, 4]
        f = [-1]
        g = [2]
        m = [1]
        self.individual_constr_meta = Individual(x, f, g, cv_nonlincon=g, m=m)

    def test_dimensions(self):
        dimensions_expected = (2, 1, 0, 0)
        dimensions = self.individual_1.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        self.assertEqual(2, self.individual_1.n_x)
        self.assertEqual(1, self.individual_1.n_f)
        self.assertEqual(0, self.individual_1.n_g)
        self.assertEqual(0, self.individual_1.n_m)

        dimensions_expected = (2, 2, 0, 0)
        dimensions = self.individual_multi_1.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        dimensions_expected = (2, 1, 1, 1)
        dimensions = self.individual_constr_meta.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        self.assertEqual(2, self.individual_constr_meta.n_x)
        self.assertEqual(1, self.individual_constr_meta.n_f)
        self.assertEqual(1, self.individual_constr_meta.n_g)
        self.assertEqual(1, self.individual_constr_meta.n_m)

    def test_domination(self):
        self.assertFalse(self.individual_1.dominates(self.individual_2))
        self.assertTrue(self.individual_2.dominates(self.individual_1))

        self.assertFalse(self.individual_multi_1.dominates(self.individual_multi_2))
        self.assertTrue(self.individual_multi_2.dominates(self.individual_multi_1))
        self.assertFalse(self.individual_multi_3.dominates(self.individual_multi_2))

        self.assertFalse(self.individual_multi_3.dominates(self.individual_multi_2))

    def test_similarity(self):
        self.assertTrue(self.individual_1.is_similar(self.individual_2, 1e-1))
        self.assertFalse(self.individual_2.is_similar(self.individual_1, 1e-8))

        self.assertTrue(
            self.individual_multi_1.is_similar(self.individual_multi_2, 1e-1)
        )
        self.assertFalse(
            self.individual_multi_2.is_similar(self.individual_multi_1, 1e-8)
        )

    def test_to_dict(self):
        data = self.individual_1.to_dict()

        np.testing.assert_equal(data["x"], self.individual_1.x)
        np.testing.assert_equal(data["f"], self.individual_1.f)
        np.testing.assert_equal(data["x_transformed"], self.individual_1.x_transformed)
        self.assertEqual(data["variable_names"], self.individual_1.variable_names)
        self.assertEqual(
            data["independent_variable_names"],
            self.individual_1.independent_variable_names,
        )

        # Missing: Test for labels.
        # self.assertEqual(data['objective_labels'], self.individual_1.objective_labels)
        # self.assertEqual(
        #     data['nonlinear_constraint_labels'],
        #     self.individual_1.nonlinear_constraint_labels,
        # )
        # self.assertEqual(data['meta_score_labels'], elf.individual_1.meta_score_labels)

    def test_from_dict(self):
        data = self.individual_1.to_dict()
        test_individual = Individual.from_dict(data)

        np.testing.assert_equal(test_individual.x, self.individual_1.x)
        np.testing.assert_equal(test_individual.f, self.individual_1.f)
        np.testing.assert_equal(test_individual.g, self.individual_1.g)
        np.testing.assert_equal(test_individual.m, self.individual_1.m)
        np.testing.assert_equal(
            test_individual.x_transformed, self.individual_1.x_transformed
        )
        self.assertEqual(
            test_individual.variable_names, self.individual_1.variable_names
        )
        self.assertEqual(
            test_individual.independent_variable_names,
            self.individual_1.independent_variable_names,
        )
        self.assertEqual(
            test_individual.objective_labels, self.individual_1.objective_labels
        )
        self.assertEqual(
            test_individual.nonlinear_constraint_labels,
            self.individual_1.nonlinear_constraint_labels,
        )
        self.assertEqual(
            test_individual.meta_score_labels, self.individual_1.meta_score_labels
        )
        self.assertEqual(test_individual.id, self.individual_1.id)


if __name__ == "__main__":
    unittest.main()
