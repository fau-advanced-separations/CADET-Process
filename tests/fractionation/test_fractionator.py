import unittest

from CADETProcess.fractionation import Fractionator
from CADETProcess.processModel import ComponentSystem
from solution_fixtures import TestSolutionIOConstant, TestSolutionIOGaussian

component_system = ComponentSystem(['A', 'B'])

test_solution_const = TestSolutionIOConstant(component_system)
test_solution_gauss = TestSolutionIOGaussian(component_system)


frac = Fractionator()


class Test_Fractionator(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)


if __name__ == '__main__':
    unittest.main()
