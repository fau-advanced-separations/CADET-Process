import numpy as np

from CADETProcess.dynamicEvents import TimeLine
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionBase, SolutionIO


class ReferenceBase(SolutionBase):
    pass


class ReferenceIO(SolutionIO):
    def __init__(
            self, name, time, solution,
            flow_rate=None, component_system=None):

        solution = np.array(solution, ndmin=2).reshape(len(time), -1)

        if component_system is None:
            n_comp = solution.shape[1]
            component_system = ComponentSystem(n_comp)

        if flow_rate is None:
            flow_rate = 1
        if isinstance(flow_rate, (int,  float)):
            flow_rate = TimeLine.from_constant(time[0], time[-1], flow_rate)

        super().__init__(name, component_system, time, solution, flow_rate)
