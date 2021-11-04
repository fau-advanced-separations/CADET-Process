"""
ToDo
----
- add/remove events (check component indices)
- add/remove durations
- add/remove dependencies


- event times (especially considerung time modulo cycle time)
- event /parameter/ performer timelines 
- section states (especially piecewise poly times)

- conflicing events at same time?

Notes
-----
Since the EventHandler defines an interface, that requires the implementation 
of some methods, a TestHandler class is defined.

Maybe this is too complicated, just use Process instead?
"""

import unittest

from addict import Dict
import numpy as np

import CADETProcess
from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Float, Switch, SizedTuple

class TestPerformer(metaclass=StructMeta):
    param_1 = Float(default=0)
    param_2 = SizedTuple(minlen=2, maxlen=2, default=(1,1))
    param_3 = Switch(valid=[-1,1], default=1)

    _parameters = ['param_1', 'param_2', 'param_3']
    _section_dependent_parameters = ['param_1', 'param_2']

    @property
    def section_dependent_parameters(self):
        parameters = {
            param: getattr(self, param) 
            for param in self._section_dependent_parameters
        }
        return parameters

    @property
    def parameters(self):
        parameters = {
            param: getattr(self, param) 
            for param in self._parameters
        }
        return parameters
    
    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, param, value)

class TestHandler(CADETProcess.dynamicEvents.EventHandler):
    def __init__(self):
        self.performer = TestPerformer()
        super().__init__()

    @property
    def parameters(self):
        parameters = super().parameters

        parameters['performer'] = self.performer.parameters

        return Dict(parameters)        

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.performer.parameters = parameters.pop('performer')
        except KeyError:
            pass

        super(TestHandler, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self):
        parameters = Dict()
        parameters['performer'] = self.performer.section_dependent_parameters
        return parameters

class Test_Events(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def create_event_handler(self):
        event_handler = TestHandler()
        
        event_handler.add_event('evt0', 'performer.param_1', 0, 0)
        event_handler.add_event('evt1', 'performer.param_1', 1, 0)
        event_handler.add_event('evt2', 'performer.param_2', (2,1), 0)
        event_handler.add_event('evt3', 'performer.param_2', (3,2), 0)
        event_handler.add_event('evt4', 'performer.param_2', (3,3), 0)

        return event_handler 
    
    def test_adding_events(self):
        event_handler = self.create_event_handler()

        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event('wrong_path', 'performer.wrong', 1)
            
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event('wrong_value', 'performer.param_1', 'wrong')
            
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event('duplicate', 'performer.param_1', 1)
            event_handler.add_event('duplicate', 'performer.param_1', 1)

        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event('not_sec_dependent', 'performer.param_3', 1)

    
    def test_event_times(self):
        event_handler = self.create_event_handler()
        
        self.assertEqual(event_handler.event_times, [0.0])

        event_handler.evt0.time = 1
        self.assertEqual(event_handler.event_times, [0.0, 1.0])
        
        event_handler.cycle_time = 10
        event_handler.evt0.time = 11
        self.assertEqual(event_handler.event_times, [0.0, 1.0])

        
    def test_dependenciies(self):
        event_handler = self.create_event_handler()
        
        event_handler.add_event_dependency('evt1', 'evt0')
        self.assertEqual(event_handler.event_times, [0.0])

        event_handler.evt0.time = 1
        self.assertEqual(event_handler.event_times, [0.0, 1.0])

        event_handler.add_event_dependency('evt2', 'evt1', 2)
        self.assertEqual(event_handler.event_times, [0.0, 1.0, 2.0])

        event_handler.add_event_dependency('evt3', ['evt1', 'evt0'], [2,1])
        self.assertEqual(event_handler.event_times, [0.0, 1.0, 2.0, 3.0])

        # Dependent event
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.evt1.time = 1
        # Event does not exist
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency('evt3', 'evt0')
        # Duplicate
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency('evt1', 'evt0')
        # factors not matching
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency('evt1', 'evt0', [1,1])
    
    def test_section_states(self):
        pass
        
        
    #     self.assertEqual(event_handler.event_times, [0.0])


    
    def test_timelines(self):
        pass
        
        
if __name__ == '__main__':
    unittest.main()
