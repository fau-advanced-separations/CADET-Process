import copy
import unittest

from CADETProcess import CADETProcessError
from examples.operating_modes.batch_elution import process

class Test_process(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def create_process(self):
        return copy.deepcopy(process)

    def test_inlet_profile(self):
        pass
        
        

if __name__ == '__main__':
    unittest.main()