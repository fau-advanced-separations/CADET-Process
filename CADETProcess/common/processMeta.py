import numpy as np

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure.parameter import  UnsignedFloat, NdArray

class ProcessMeta(Structure):
    """Additional information required for calculating performance

    Attributes
    ----------
    cycle_time : float
        Cycle time of process
    m_feed : ndarray
        Ammount of feed used in the process
        value is None.
    V_solid : UnsignedFloat
        Volume of the solid phase used in the process
    V_eluent : UnsignedFloat
        Volume of the consumed eluent used in the process
    """
    _meta_keys = ['cycle_time', 'm_feed', 'V_solid', 'V_eluent']

    cycle_time = UnsignedFloat()
    m_feed = NdArray()
    V_solid = UnsignedFloat()
    V_eluent = UnsignedFloat()

    def to_dict(self):
        return {key: getattr(self, key) for key in self._meta_keys}

    def __repr__(self):
        return '{}(cycle_time={}, m_feed={}, V_solid={}, V_eluent={})'.format(
            self.__class__.__name__, self.cycle_time,
            np.array_repr(self.m_feed), self.V_solid, self.V_eluent
        )

feed = np.linspace(1,10)
foo = ProcessMeta(1, feed, 3, 4)