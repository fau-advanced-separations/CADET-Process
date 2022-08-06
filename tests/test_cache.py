import unittest

from CADETProcess.optimization import ResultsCache


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
