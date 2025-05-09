from typing import Optional

from functools import wraps
import warnings

from addict import Dict
import numpy as np
import numpy.typing as npt


from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Aggregator, SizedAggregator, SizedClassDependentAggregator,
)
from CADETProcess.dataStructure import (
    Bool, String, SizedList, SizedNdArray, UnsignedInteger, UnsignedFloat
)

from CADETProcess.dataStructure import deprecated_alias

from CADETProcess.processModel.componentSystem import ComponentSystem

class ReactionBase(Structure):
    """Abstract base class for parameters of reaction models.

    Attributes
    ----------
    n_comp : UnsignedInteger
        number of components.
    parameters : dict
        dict with parameter values.
    name : String
        name of the reaction model.

    """
    name = String()
    n_comp = UnsignedInteger()

    _parameters = []

    def __init__(self, component_system, name=None, *args, **kwargs):
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def component_system(self):
        """ComponentSystem: Component System"""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def n_comp(self):
        """int: Number of components."""
        return self.component_system.n_comp

    def __repr__(self):
        return \
            f'{self.__class__.__name__}(' \
            f'n_comp={self.n_comp}, name={self.name}' \
            f')'

    def __str__(self):
        if self.name is None:
            return self.__class__.__name__
        return self.name

class NoReaction(ReactionBase):
    """Dummy class for units that do not experience reaction behavior.

    The number of components is set to zero for this class.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(ComponentSystem(), name='NoReaction')

class InhibitionBase():
    pass

class CompetitiveInhibition(InhibitionBase):
    _name = "Competitive Inhibition"

    _parameters = [
        'component_system',
        'substrate',
        'inhibitors',
        'ki'
    ]

    def __init__(
        self,
        component_system: ComponentSystem,
        substrate: str,
        inhibitors: str | list[str],
        ki: float | list[float],
        ):

        self.component_system = component_system
        self.substrate = substrate
        self.inhibitors = inhibitors
        self.ki = ki

    @property
    def name(self):
        """str: Name of the inhibition reaction."""
        return self._name

class UnCompetitiveInhibition(InhibitionBase):

    _name = "Uncompetitive Inhibition"

    _parameters = [
        'component_system',
        'substrate',
        'inhibitors',
        'ki'
    ]

    def __init__(
        self,
        component_system: ComponentSystem,
        substrate: str,
        inhibitors: str | list[str],
        ki: float | list[float],
        ):

        self.component_system = component_system
        self.substrate = substrate
        self.inhibitors = inhibitors
        self.ki = ki

    @property
    def name(self):
        """str: Name of the inhibition reaction."""
        return self._name

class NonCompetitiveInhibition(InhibitionBase):

    _name = "Non-competitive Inhibition"
    _parameters = [
        'component_system',
        'substrate',
        'inhibitors',
        'ki'
    ]

    def __init__(
        self,
        component_system: ComponentSystem,
        substrate: str,
        inhibitors: str | list[str],
        ki: float | list[float],
        ):

        self.component_system = component_system
        self.substrate = substrate
        self.inhibitors = inhibitors
        self.ki = ki

    @property
    def name(self):
        """str: Name of the inhibition reaction."""
        return self._name

class MichaelisMenten(ReactionBase):

    """Michaelis-Menten reaction model.
    Parameters
    ----------
    component_system : ComponentSystem
        Component system.
    components : list[str]
        List of components.
    coefficients : list[float]
        List of coefficients.
    km : float
        Michaelis-Menten constant.
    vmax : float
        Maximum reaction rate.
    inhibition_reactions : list[InhibitionBase], optional
        List of inhibition reactions. Default is None.
    """
    name = "Michaelis-Menten"
    _stoichVec = None


    _parameters = [
        'component_system',
        'components',
        'coefficients',
        'km',
        'vmax',
        'inhibition_reactions'
    ]

    def __init__(
            self,
            component_system: ComponentSystem,
            components: list[str],
            coefficients :list[float],
            km: list[float],
            vmax: float,
            inhibition_reactions: Optional[list[InhibitionBase]] = None,
        ):

        # Pass on parameters
        self._km = km
        self._vmax = vmax
        self._component_system = component_system

        indices = [component_system.species.index(i) for i in components]
        self._stoichVec = np.zeros((self.n_comp))
        for i, c in zip(indices, coefficients):
            self._stoichVec[i] = c

        self._ki_competative = np.zeros((self.n_comp,self.n_comp))
        self._ki_uncompetative = np.zeros((self.n_comp,self.n_comp))

        # set inhibition parameters
        for inhibition in inhibition_reactions:
            if isinstance(inhibition, CompetitiveInhibition, NonCompetitiveInhibition):
                for inh in inhibition.inhibitors:
                    inhibitor_index = component_system.species.index(inh)
                    substrate_index = component_system.species.index(inhibition.substrat)
                    self._ki_competative[substrate_index][inhibitor_index] = inhibition.ki[inhibitor_index]
            elif isinstance(inhibition, UnCompetitiveInhibition, NonCompetitiveInhibition):
                for inh in inhibition.inhibitors:
                    inhibitor_index = component_system.species.index(inh)
                    substrate_index = component_system.species.index(inhibition.substrat)
                    self._ki_uncompetative[substrate_index][inhibitor_index] = inhibition.ki[inhibitor_index]
            else:
                raise TypeError('Unknown inhibition type')

    @property
    def n_comp(self):
        """int: Number of components."""
        return self._component_system.n_comp

    @property
    def components(self):
        """list[str]: List of components."""
        return self.components

    @property
    def coefficients(self):
        """list[float]: List of coefficients."""
        return self.coefficients

    @property
    def km(self):
        """float: Michaelis-Menten constant."""
        return self.km

    @property
    def vmax(self):
        """float: Maximum reaction rate."""
        return self.vmax

    @property
    def inhibition_reactions(self):
        """list[InhibitionBase]: List of inhibition reactions."""
        return self.inhibition_reactions
