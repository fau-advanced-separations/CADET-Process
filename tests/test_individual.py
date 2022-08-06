import unittest

from CADETProcess.optimization import Individual


class TestIndividual(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

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

    def test_dimensions(self):
        dimensions_expected = (2, 1, 0, 0)
        dimensions = self.individual_1.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        dimensions_expected = (2, 2, 0, 0)
        dimensions = self.individual_multi_1.dimensions
        self.assertEqual(dimensions, dimensions_expected)

        dimensions_expected = (2, 1, 1, 0)
        dimensions = self.individual_constr_1.dimensions
        self.assertEqual(dimensions, dimensions_expected)

    def test_domination(self):
        self.assertFalse(self.individual_1.dominates(self.individual_2))
        self.assertTrue(self.individual_2.dominates(self.individual_1))

        self.assertFalse(
            self.individual_multi_1.dominates(self.individual_multi_2)
        )
        self.assertTrue(
            self.individual_multi_2.dominates(self.individual_multi_1)
        )
        self.assertFalse(
            self.individual_multi_3.dominates(self.individual_multi_2)
        )
        
        self.assertFalse(
            self.individual_multi_3.dominates(self.individual_multi_2)
        )

    def test_similarity(self):
        self.assertTrue(self.individual_1.is_similar(self.individual_2, 1e-1))
        self.assertFalse(self.individual_2.is_similar(self.individual_1, 1e-8))

        self.assertTrue(
            self.individual_multi_1.is_similar(self.individual_multi_2, 1e-1)
        )
        self.assertFalse(
            self.individual_multi_2.is_similar(self.individual_multi_1, 1e-8)
        )


if __name__ == '__main__':
    unittest.main()
