import unittest

import numpy as np
from CADETProcess.processModel import ComponentSystem
from CADETProcess.fractionation import Fractionator
from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.fractionation import SimulationResults


from solution_fixtures import TestSolutionIOConstant, TestSolutionIOGaussian

component_system = ComponentSystem(['A', 'B'])

test_solution_const = TestSolutionIOConstant(component_system)
test_solution_gauss = TestSolutionIOGaussian(component_system)


frac = Fractionator():

class Test

class Test_Fractionator(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

if __name__ == '__main__':
    unittest.main()
