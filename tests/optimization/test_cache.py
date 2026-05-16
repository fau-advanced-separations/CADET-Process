import unittest

from CADETProcess.optimization import ResultsCache


class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache_dict = ResultsCache(use_diskcache=False)
        self.cache_disk = ResultsCache(use_diskcache=True)

        key = (None, "objective", str([1, 2, 3]))
        result = "important result"
        self.cache_dict.set(key, result)
        self.cache_disk.set(key, result)

        key = (None, "intermediate", str([1, 2, 3]))
        result = "temporary result"
        self.cache_dict.set(key, result, "temp")
        self.cache_disk.set(key, result, "temp")

    def tearDown(self):
        self.cache_disk.delete_database()

    def test_set(self):
        new_result = "new"
        key = (None, "other", str([1, 2, 3]))

        self.cache_dict.set(key, new_result)
        cached_result = self.cache_dict.get(key)
        self.assertEqual(cached_result, new_result)

        self.cache_disk.set(key, new_result)
        cached_result = self.cache_disk.get(key)
        self.assertEqual(cached_result, new_result)

    def test_get(self):
        key = (None, "intermediate", str([1, 2, 3]))
        result_expected = "temporary result"
        cached_result = self.cache_dict.get(key)
        self.assertEqual(result_expected, cached_result)

        key = (None, "intermediate", str([1, 2, 3]))
        result_expected = "temporary result"
        cached_result = self.cache_disk.get(key)
        self.assertEqual(result_expected, cached_result)

        key = (None, "false", str([1, 2, 3]))
        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(key)

        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(key)

    def test_delete(self):
        key = (None, "intermediate", str([1, 2, 3]))
        self.cache_dict.delete(key)
        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(key)

        self.cache_disk.delete(key)
        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(key)

    def test_tags(self):
        tags_expected = ["temp"]
        tags = list(self.cache_dict.tags.keys())
        self.assertEqual(tags_expected, tags)

        key = (None, "other", str([1, 2, 4]))
        self.cache_dict.set(key, "new", "foo")
        tags_expected = ["temp", "foo"]
        tags = list(self.cache_dict.tags.keys())
        self.assertEqual(tags_expected, tags)

    def test_prune(self):
        key = (None, "intermediate", str([1, 2, 3]))
        self.cache_dict.prune("temp")
        with self.assertRaises(KeyError):
            cached_result = self.cache_dict.get(key)

        key = (None, "objective", str([1, 2, 3]))
        result_expected = "important result"
        cached_result = self.cache_dict.get(key)
        self.assertEqual(result_expected, cached_result)

        key = (None, "intermediate", str([1, 2, 3]))
        self.cache_disk.prune("temp")
        with self.assertRaises(KeyError):
            cached_result = self.cache_disk.get(key)

        key = (None, "objective", str([1, 2, 3]))
        result_expected = "important result"
        cached_result = self.cache_disk.get(key)
        self.assertEqual(result_expected, cached_result)

    def test_delete_database(self):
        pass


if __name__ == "__main__":
    unittest.main()
