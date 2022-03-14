import numpy as np

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure.parameter import UnsignedFloat, NdArray


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
        return \
            f"{self.__class__.__name__}(cycle_time={self.cycle_time}, "\
            f"m_feed={np.array_repr(self.m_feed)}, V_solid={self.V_solid}, "\
            f"V_eluent={self.V_eluent})"
