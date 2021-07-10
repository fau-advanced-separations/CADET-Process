from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta, String, UnsignedInteger, \
    UnsignedFloat, DependentlySizedList, DependentlySizedUnsignedList

class ReactionBaseClass(metaclass=StructMeta):
    """Abstract base class for parameters of binding models.

    Attributes
    ----------
    n_comp : UnsignedInteger
        number of components.
    parameters : dict
        dict with parameter values.
    name : String
        name of the binding model.
    """
    name = String()
    n_comp = UnsignedInteger()
    
    _parameters = []

    def __init__(self, n_comp, n_reac, name):
        self.n_comp = n_comp
        self.n_reac = n_reac
        self.name = name

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        return {param: getattr(self, param) for param in self._parameters}

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            setattr(self, param, value)


    def __repr__(self):
        return '{}(n_comp={}, name=\'{}\')'.format(self.__class__.__name__,
            self.n_comp, self.name)

    def __str__(self):
        return self.name

class NoReaction(ReactionBaseClass):
    """Dummy class for units that do not experience reaction behavior.

    The number of components is set to zero for this class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(n_comp=0, n_reac=0, name='NoReaction')

class MassAction(ReactionBaseClass):
    """Parameters for Linear binding model.

    Attributes
    ----------
    forward_rate : list of unsigned floats. Length depends on n_reac.
        Forward reaction rates.
    backward_rate : list of unsigned floats. Length depends on n_reac.
        Backward reaction rates.
    stoichiometric_matrix : list of unsinged floats. 
        Stoichiometric matrix of reactions as n_comp x n_reac
    forward_modifier_exponent : list of unsinged floats. 
        Exponent matrix for forward reaction modifier.
    backward_modifier_exponent : list of unsinged floats. 
        Exponent matrix for backward reaction modifier.
    """
    forward_rate = DependentlySizedUnsignedList(dep='n_reac')
    backward_rate = DependentlySizedUnsignedList(dep='n_reac', default=1)
    
    stoichiometric_matrix = DependentlySizedList(
        dep=('n_comp', 'n_reac'))
    
    forward_modifier_exponent = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_reac'))
    backward_modifier_exponent = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_reac'))
    
    _parameters = ReactionBaseClass._parameters + [
        'forward_rate', 'backward_rate', 'stoichiometric_matrix',
        'forward_modifier_exponent', 'forward_modifier_exponent',] 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class BulkReaction(ReactionBaseClass):
    """Parameters for Linear binding model.

    Attributes
    ----------
    forward_rate : list of unsigned floats. Length depends on n_reac.
        Forward reaction rates.
    backward_rate : list of unsigned floats. Length depends on n_reac.
        Backward reaction rates.
    stoichiometric_matrix : list of unsinged floats. 
        Stoichiometric matrix of reactions as n_comp x n_reac
    forward_modifier_exponent : list of unsinged floats. 
        Exponent matrix for forward reaction modifier.
    backward_modifier_exponent : list of unsinged floats. 
        Exponent matrix for backward reaction modifier.
    """
    forward_rate = DependentlySizedUnsignedList(dep='n_reac')
    backward_rate = DependentlySizedUnsignedList(dep='n_reac', default=1)
    
    stoichiometric_matrix = DependentlySizedList(
        dep=('n_comp', 'n_reac'))
    
    forward_modifier_exponent = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_reac'))
    backward_modifier_exponent = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_reac'))
    
    _parameters = ReactionBaseClass._parameters + [
        'forward_rate', 'backward_rate', 'stoichiometric_matrix',
        'forward_modifier_exponent', 'forward_modifier_exponent',] 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
