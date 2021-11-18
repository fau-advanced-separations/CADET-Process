from addict import Dict
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    String, UnsignedInteger, UnsignedFloat, 
    DependentlySizedList, DependentlySizedUnsignedList
)
    
from .componentSystem import ComponentSystem

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
    
    _parameter_names = []

    def __init__(self, component_system, name=None):
        self.component_system = component_system
        self.name = name
        
        self._parameters = {
            param: getattr(self, param)
            for param in self._parameter_names
        }
        
    @property
    def model(self):
        return self.__class__.__name__
    
    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system
    
    @property
    def n_comp(self):
        return self.component_system.n_comp
    
    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        return {param: getattr(self, param) for param in self._parameter_names}

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameter_names:
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
        super().__init__(ComponentSystem(), name='NoReaction')

class MassActionLaw(ReactionBaseClass):
    _parameter_names = ReactionBaseClass._parameter_names + [
        'stoich', 'exponents_fwd', 'exponents_bwd', 'k_fwd', 'k_bwd'
    ]
    
    def __init__(self, *args, **kwargs):
        self._reactions = []

        super().__init__(*args, **kwargs)
    
    def add_reaction(self, *args, **kwargs):
        r = Reaction(self.n_comp, *args, **kwargs)
        self._reactions.append(r)

    @property
    def reactions(self):
        return self._reactions

    @property
    def n_reactions(self):
        return len(self.reactions)
    
    @property
    def stoich(self):
        stoich = np.zeros((self.n_comp, self.n_reactions))

        for i, r in enumerate(self.reactions):
            stoich[:,i] = r.stoich

        return stoich

    @property
    def exponents_fwd(self):
        exponents = np.zeros((self.n_comp, self.n_reactions))
        
        for i, r in enumerate(self.reactions):
            exponents[:,i] = r.exponents_fwd
        
        return exponents
    
    @property
    def exponents_bwd(self):
        exponents = np.zeros((self.n_comp, self.n_reactions))
        
        for i, r in enumerate(self.reactions):
            exponents[:,i] = r.exponents_bwd
        
        return exponents
    
    @property
    def k_fwd(self):
        return [r.k_fwd for r in self.reactions]

    @property
    def k_bwd(self):
        return [r.k_bwd for r in self.reactions]

    @property
    def k_eq(self):
        return [f/b for f, b in zip(self. k_fwd, self.k_bwd)]
    

class MassActionLawParticle(ReactionBaseClass):
    _parameter_names = ReactionBaseClass._parameter_names + [
        'stoich_liquid', 'exponents_fwd_liquid', 'exponents_bwd_liquid', 
        'k_fwd_liquid', 'k_bwd_liquid', 
        'exponents_fwd_liquid_modsolid', 'exponents_bwd_liquid_modsolid', 
        'stoich_solid', 'exponents_fwd_solid', 'exponents_bwd_solid', 
        'k_fwd_solid', 'k_bwd_solid',
        'exponents_fwd_solid_modliquid', 'exponents_bwd_solid_modliquid'
    ]
    
    def __init__(self, *args, **kwargs):
        self._liquid_reactions = []
        self._solid_reactions = []
        self._cross_phase_reactions = []
        
        super().__init__(*args, **kwargs)
    
    def add_liquid_reaction(self, *args, **kwargs):
        r = Reaction(self.n_comp, *args, **kwargs)
        self._liquid_reactions.append(r)

    def add_solid_reaction(self, *args, **kwargs):
        r = Reaction(self.n_comp, *args, **kwargs)
        self._solid_reactions.append(r)

    def add_cross_phase_reaction(self, *args, **kwargs):
        r = CrossPhaseReaction(self.n_comp, *args, **kwargs)
        self._cross_phase_reactions.append(r)

    ## Pore Liquid
    @property
    def liquid_reactions(self):
        return self._liquid_reactions + self.cross_phase_reactions

    @property
    def n_liquid_reactions(self):
        return len(self.liquid_reactions)

    @property
    def stoich_liquid(self):
        stoich_liquid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                stoich_liquid[:,i] = r.stoich_liquid
            else:
                stoich_liquid[:,i] = r.stoich

        return stoich_liquid

    @property
    def exponents_fwd_liquid(self):
        exponents = np.zeros((self.n_comp, self.n_liquid_reactions))
        
        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_fwd_liquid
            else:
                exponents[:,i] = r.exponents_fwd
        
        return exponents
    
    @property
    def exponents_bwd_liquid(self):
        exponents = np.zeros((self.n_comp, self.n_liquid_reactions))
        
        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_bwd_liquid
            else:
                exponents[:,i] = r.exponents_bwd
        
        return exponents

    @property
    def k_fwd_liquid(self):
        return [r.k_fwd for r in self.liquid_reactions]

    @property
    def k_bwd_liquid(self):
        return [r.k_bwd for r in self.liquid_reactions]
    
    @property
    def k_eq_liquid(self):
        return [f/b for f, b in zip(self. k_fwd_liquid, self.k_bwd_liquid)]

    ## Solid
    @property
    def solid_reactions(self):
        return self._solid_reactions + self.cross_phase_reactions

    @property
    def n_solid_reactions(self):
        return len(self.solid_reactions)

    @property
    def stoich_solid(self):
        stoich_solid = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                stoich_solid[:,i] = r.stoich_solid
            else:
                stoich_solid[:,i] = r.stoich

        return stoich_solid

    @property
    def exponents_fwd_solid(self):
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))
        
        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_fwd_solid
            else:
                exponents[:,i] = r.exponents_fwd
        
        return exponents
    
    @property
    def exponents_bwd_solid(self):
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))
        
        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_bwd_solid
            else:
                exponents[:,i] = r.exponents_bwd
        
        return exponents

    @property
    def k_fwd_solid(self):
        return [r.k_fwd for r in self.solid_reactions]

    @property
    def k_bwd_solid(self):
        return [r.k_bwd for r in self.solid_reactions]
    
    @property
    def k_eq_solid(self):
        return [f/b for f, b in zip(self. k_fwd_solid, self.k_bwd_solid)]

    ## Cross Phase
    @property
    def cross_phase_reactions(self):
        return self._cross_phase_reactions

    @property
    def n_cross_phase_reactions(self):
        return len(self.cross_phase_reactions)
    
    @property
    def exponents_fwd_liquid_modsolid(self):
        if self.n_cross_phase_reactions == 0:
            return Dict()
            
        liquid_fwd_modsolid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                liquid_fwd_modsolid[:,i] = r.exponents_fwd_liquid_modsolid

        return liquid_fwd_modsolid

    @property
    def exponents_bwd_liquid_modsolid(self):
        if self.n_cross_phase_reactions == 0:
            return Dict()
        
        liquid_bwd_modsolid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                liquid_bwd_modsolid[:,i] = r.exponents_bwd_liquid_modsolid

        return liquid_bwd_modsolid    
    
    @property
    def exponents_fwd_solid_modliquid(self):
        if self.n_cross_phase_reactions == 0:
            return Dict()
        
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_fwd_solid_modliquid

        return exponents

    @property
    def exponents_bwd_solid_modliquid(self):
        if self.n_cross_phase_reactions == 0:
            return Dict()
        
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:,i] = r.exponents_bwd_solid_modliquid

        return exponents    
    
        
class Reaction():
    """Helper class to store information about individual MAL reactions
    """
    def __init__(
            self, n_comp, indices, coefficients, 
            k_fwd, k_bwd=1, is_kinetic=True, k_fwd_min=100,
            exponents_fwd=None, exponents_bwd=None):
        
        self.n_comp = n_comp
        
        self.stoich = np.zeros((n_comp,))
        for i, c in zip(indices, coefficients):
            self.stoich[i] = c

        if not is_kinetic:
            self.k_fwd, self.k_bwd = scale_to_rapid_equilibrium(k_fwd, k_fwd_min)
        else:
            self.k_fwd = k_fwd
            self.k_bwd = k_bwd
            
        self._exponents_fwd = exponents_fwd
        self._exponents_bwd = exponents_bwd
        
    @property
    def exponents_fwd(self):
        if self._exponents_fwd is not None:
            return self._exponents_fwd
        else:
            return np.maximum(np.zeros((self.n_comp,)), -self.stoich)
    
    @property
    def exponents_bwd(self):
        if self._exponents_bwd is not None:
            return self._exponents_bwd
        else:
            return np.maximum(np.zeros((self.n_comp,)), self.stoich)
            

class CrossPhaseReaction():
    """Helper class to store information about individual cross-phase MAL reactions
    """
    def __init__(
            self, n_comp, indices, coefficients, phases, 
            k_fwd, k_bwd=1, is_kinetic=True, k_fwd_min=100,
            exponents_fwd_liquid=None, exponents_bwd_liquid=None,
            exponents_fwd_solid=None, exponents_bwd_solid=None,):
        
        self.n_comp = n_comp

        self.stoich_liquid = np.zeros((n_comp,))
        self.stoich_solid = np.zeros((n_comp,))
        self.exponents_fwd_solid_modliquid = np.zeros((n_comp,))
        self.exponents_bwd_solid_modliquid = np.zeros((n_comp,))
        self.exponents_fwd_liquid_modsolid = np.zeros((n_comp,))
        self.exponents_bwd_liquid_modsolid = np.zeros((n_comp,))

        if phases is None:
            phases = [0 for n in indices]

        for i, c, p in zip(indices, coefficients, phases):
            if p == 0:
                self.stoich_liquid[i] = c
                if c < 0:
                    self.exponents_fwd_solid_modliquid[i] = abs(c)
                elif c > 0:
                    self.exponents_bwd_solid_modliquid[i] = c
            elif p == 1:
                self.stoich_solid[i] = c
                if c < 0:
                    self.exponents_fwd_liquid_modsolid[i] = abs(c)
                elif c > 0:
                    self.exponents_bwd_liquid_modsolid[i] = c

        if not is_kinetic:
            self.k_fwd, self.k_bwd = scale_to_rapid_equilibrium(k_fwd, k_fwd_min)
        else:
            self.k_fwd = k_fwd
            self.k_bwd = k_bwd
            
        self._exponents_fwd_liquid = exponents_fwd_liquid
        self._exponents_bwd_liquid = exponents_bwd_liquid
        self._exponents_fwd_solid = exponents_fwd_solid
        self._exponents_bwd_solid = exponents_bwd_solid
        
    @property
    def exponents_fwd_liquid(self):
        if self._exponents_fwd_liquid is not None:
            return self._exponents_fwd_liquid
        else:
            return np.maximum(np.zeros((self.n_comp,)), -self.stoich_liquid)
    
    @property
    def exponents_bwd_liquid(self):
        if self._exponents_bwd_liquid is not None:
            return self._exponents_bwd_liquid
        else:
            return np.maximum(np.zeros((self.n_comp,)), self.stoich_liquid)            
        
    @property
    def exponents_fwd_solid(self):
        if self._exponents_fwd_solid is not None:
            return self._exponents_fwd_solid
        else:
            return np.maximum(np.zeros((self.n_comp,)), -self.stoich_solid)
    
    @property
    def exponents_bwd_solid(self):
        if self._exponents_bwd_solid is not None:
            return self._exponents_bwd_solid
        else:
            return np.maximum(np.zeros((self.n_comp,)), self.stoich_solid)            


def scale_to_rapid_equilibrium(k_eq, k_fwd_min=10):
    """Scale forward and backward reaction rates if only k_eq is known.
    
    Parameters
    ----------
    k_eq : float
        Equilibrium constant.
    k_fwd_min : float, optional
        Minimum value for forwards reaction. The default is 10.

    Returns
    -------
    k_fwd : float
        Forward reaction rate.
    k_bwd : float
        Backward reaction rate.
    """
    k_fwd = k_fwd_min
    k_bwd = k_fwd_min/k_eq
    
    return k_fwd, k_bwd    