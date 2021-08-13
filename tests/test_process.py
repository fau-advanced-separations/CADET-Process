import copy
import unittest


from CADETProcess import CADETProcessError
from examples.forward_simulation.batch_binary import batch_binary

class Test_process(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def create_process(self):
        process = copy.deepcopy(batch_binary)

        return process
    
    def test_inlet_profile(self):
        pass
        
        

if __name__ == '__main__':
    unittest.main()