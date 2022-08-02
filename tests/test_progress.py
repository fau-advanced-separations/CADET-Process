import unittest

from CADETProcess.optimization import (
    Individual,  OptimizationProgress, ResultsCache
)


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
        f = [-1]
        g = [3]
        self.individual_constr_1 = Individual(x, f, g)

        x = [1, 2]
        f = [-1]
        g = [3]
        self.individual_constr_2 = Individual(x, f, g)

    def test_domination(self):
        self.assertFalse(self.individual_1.dominates(self.individual_2))
        self.assertTrue(self.individual_2.dominates(self.individual_1))

        self.assertFalse(
            self.individual_multi_1.dominates(self.individual_multi_2)
        )
        self.assertTrue(
            self.individual_multi_2.dominates(self.individual_multi_1)
        )

    def test_similarity(self):
        self.assertTrue(self.individual_1.is_similar(self.individual_2, 1e-1))
        self.assertFalse(self.individual_2.is_similar(self.individual_1))

        self.assertTrue(
            self.individual_multi_1.is_similar(self.individual_multi_2, 1e-1)
        )
        self.assertFalse(
            self.individual_multi_2.is_similar(self.individual_multi_1)
        )


class TestProgress(unittest.TestCase):
    pass


class TestCache(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.cache_dict = ResultsCache(use_diskcache=False)
        self.cache_disk = ResultsCache(use_diskcache=True)

        x = [1, 2, 3]
        eval_obj = None

        step = 'objective'
        result = 'important result'
        self.cache_dict.set(eval_obj, step, x, result)
        self.cache_disk.set(eval_obj, step, x, result)

        step = 'intermediate'
        result = 'temporary result'

        self.cache_dict.set(eval_obj, step, x, result, 'temp')
        self.cache_disk.set(eval_obj, step, x, result, 'temp')

    def test_set(self):
        new_result = 'new'

        self.cache_dict.set(None, 'other', [1, 2, 3], new_result)
        cached_result = self.cache_dict.get(None, 'other', [1, 2, 3])
        self.assertEqual(cached_result, new_result)

        self.cache_disk.set(None, 'other', [1, 2, 3], new_result)
        cached_result = self.cache_disk.get(None, 'other', [1, 2, 3])
        self.assertEqual(cached_result, new_result)

    def test_get(self):
        result_expected = 'temporary result'
        cached_result = self.cache_dict.get(None, 'intermediate', [1, 2, 3])
        self.assertEqual(result_expected, cached_result)

        result_expected = 'temporary result'
        cached_result = self.cache_disk.get(None, 'intermediate', [1, 2, 3])
        self.assertEqual(result_expected, cached_result)

        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(None, 'false', [1, 2, 3])

        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(None, 'false', [1, 2, 3])

    def test_delete(self):
        self.cache_dict.delete(None, 'intermediate', [1, 2, 3])
        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(None, 'intermediate', [1, 2, 3])

        self.cache_disk.delete(None, 'intermediate', [1, 2, 3])
        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(None, 'intermediate', [1, 2, 3])

    def test_tags(self):
        tags_expected = ['temp']
        tags = list(self.cache_dict.tags.keys())
        self.assertEqual(tags_expected, tags)

        self.cache_dict.set(None, 'other', [1, 2, 4], 'new', 'foo')
        tags_expected = ['temp', 'foo']
        tags = list(self.cache_dict.tags.keys())
        self.assertEqual(tags_expected, tags)

    def test_prune(self):
        self.cache_dict.prune('temp')
        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(None, 'intermediate', [1, 2, 3])

        result_expected = 'important result'
        cached_result = self.cache_dict.get(None, 'objective', [1, 2, 3])
        self.assertEqual(result_expected, cached_result)

        self.cache_disk.prune('temp')
        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(None, 'intermediate', [1, 2, 3])

        result_expected = 'important result'
        cached_result = self.cache_disk.get(None, 'objective', [1, 2, 3])
        self.assertEqual(result_expected, cached_result)



if __name__ == '__main__':
    unittest.main()
