from CADETProcess.dataStructure import (
    Bool,
    Float,
    Structure,
)

__all__ = ["ParameterParameterDependencyBase", "PowerLaw"]


class ParameterParameterDependencyBase(Structure):
    """Base class for parameter-parameter dependencies."""

    pass


class PowerLaw(ParameterParameterDependencyBase):
    """Parameter-parmeter dependency following a power law."""

    base = Float()
    exponent = Float()
    calculate_absolute_value = Bool(default=True)

    _parameters = ["base", "exponent", "calculate_absolute_value"]
