import copy
import unittest

from CADETProcess import CADETProcessError
from examples.batch_elution.process import process


class Test_process(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def create_process(self):
        return copy.deepcopy(process)

    def test_inlet_profile(self):
        pass

    def test_check_cstr_volume(self):
        pass


if __name__ == '__main__':
    unittest.main()
