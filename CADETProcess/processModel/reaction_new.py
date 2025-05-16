from typing import Optional

from functools import wraps
import warnings

from addict import Dict
import numpy as np
import numpy.typing as npt


from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import frozen_attributes

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Aggregator, SizedAggregator, SizedClassDependentAggregator,
)
from CADETProcess.dataStructure import (
    Bool, String, SizedList, SizedNdArray, UnsignedInteger, UnsignedFloat, SizedUnsignedList
)

from CADETProcess.dataStructure import deprecated_alias

from CADETProcess.processModel.componentSystem import ComponentSystem

@frozen_attributes
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

    _name = String()
    _parameters = []

    def __init__(self, component_system, components, coefficients, name=None, *args, **kwargs):
        self._component_system = component_system
        self._name = name
        self._components = components
        self._coefficients = coefficients
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def name(self):
        """str: Name of the reaction model."""
        return self._name

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
        return self._component_system.n_comp

    @property
    def reactions(self):
        """list[ReactionBulkBase]: List of reactions."""
        return self._reactions

    @property
    def components(self):
        """list[str]: List of components."""
        return self._components

    @property
    def coefficients(self):
        """list[float]: List of coefficients."""
        return self._coefficients

    def __repr__(self):
        return \
            f'{self.__class__.__name__}(' \
            f'n_comp={self.n_comp}, name={self.name}' \
            f')'

    def __str__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

class NoReaction(ReactionBase):
    """Dummy class for units that do not experience reaction behavior.

    The number of components is set to zero for this class.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(ComponentSystem(), components=[], coefficients= [], name='NoReaction')

class InhibitionBase():
    """Abstract base class for inhibition reactions."""


    _parameters = [
        'component_system',
        'substrate',
        'inhibitors',
        'ki'
    ]

    def __init__(
        self,
        type: str,
        component_system: ComponentSystem,
        substrate: str,
        inhibitors: str | list[str],
        ki: float | list[float],
        ):

        self._type = type
        self._component_system = component_system
        self._substrate = substrate
        self._inhibitors = inhibitors
        self._ki = ki

        #check if substrate is in component system
        self.validate_types()
        self.validate_components()


    def validate_types(self):
        """Validate the inhibition reaction."""
        if not isinstance(self._ki, (float, list)):
            raise TypeError("Inhibition constant must be a float or list of floats")
        if isinstance(self._ki, list):
            for ki in self._ki:
                if not isinstance(ki, float):
                    raise TypeError("Inhibition constant must be a float or list of floats")
        if not isinstance(self._component_system, ComponentSystem):
            raise TypeError("Component system must be a ComponentSystem object")

    def validate_components(self):
        """ Validate, if substrate and inhibitors are correctly set """

        if self._substrate not in self._component_system.species:
            raise ValueError(f"Substrate {self._substrate} not in component system")

        if isinstance(self._inhibitors, list):
            for i in self._inhibitors:
                if i not in self._component_system.species:
                    raise ValueError(f"Inhibitor {i} not in component system")
        else:
            if self._inhibitors not in self._component_system.species:
                raise ValueError(f"Inhibitor {self._inhibitors} not in component system")

    @property
    def type(self):
        """str: Type of inhibition"""
        return self._type

    @property
    def ki(self):
        """float: Inhibition constant."""
        if isinstance(self._ki, list):
            return np.array(self._ki)
        return self._ki

    @ki.setter
    def ki(self, ki):
        if isinstance(ki, list):
            self._ki = np.array(ki)
        else:
            self._ki = ki

    @property
    def component_system(self):
        """ComponentSystem: Component System"""
        return self._component_system

    @property
    def substrate(self):
        """str: Substrate"""
        return self._substrate

    @property
    def inhibitors(self):
        """str | list[str]: Inhibitors"""
        if isinstance(self._inhibitors, list):
            return np.array(self._inhibitors)
        return self._inhibitors

class CompetitiveInhibition(InhibitionBase):
    _type = "Competitive Inhibition"

    parameters = [ "ki" ]

class UnCompetitiveInhibition(InhibitionBase):
    _type = "Uncompetitive Inhibition"
    parameters = [ "ki" ]

class NonCompetitiveInhibition(InhibitionBase):
    _type = "Non-competitive Inhibition"
    parameters = [ "ki" ]

class MichaelisMenten(ReactionBase):

    """Michaelis-Menten reaction model.
    Parameters
    ----------
    km : float
        Michaelis-Menten constant.
    vmax : float
        Maximum reaction rate.
    inhibition_reactions : list[InhibitionBase], optional
        List of inhibition reactions. Default is None.
    """
    _type = "Michaelis-Menten"


    _km = list()
    _vmax = UnsignedFloat()
    _inhibition_reactions = list()

    _parameters = [ 'km', 'vmax' ]

    def __init__(self, component_system, components, coefficients, name=None, *args, **kwargs):
        super().__init__(component_system, components, coefficients, name, *args, **kwargs)
        self._inhibition_reactions = []

    @property
    def inhibition_reactions(self):
        """list[InhibitionBase]: List of inhibition reactions."""
        if not hasattr(self, '_inhibition_reactions'):
            self._inhibition_reactions = []
        return self._inhibition_reactions


    def add_inhibition_reaction(self, inhibition_reaction):
        """Add inhibition reaction to the model."""
        if not isinstance(inhibition_reaction, InhibitionBase):
            raise TypeError('Expected InhibitionBase')
        self._inhibition_reactions.append(inhibition_reaction)


    def __str__(self):
        for inh in self.inhibition_reactions:
            if isinstance(inh, CompetitiveInhibition):
                return f"{self._type} with {inh.type}"
            elif isinstance(inh, UnCompetitiveInhibition):
                return f"{self._type} with {inh.type}"
            elif isinstance(inh, NonCompetitiveInhibition):
                return f"{self._type} with {inh.type}"
        return f"{self._type} without inhibition"

    @property
    def km(self):
        """float | list[float]: Michaelis-Menten constant."""
        if isinstance(self._km, list):
            return np.array(self._km)
        return self._km

    @km.setter
    def km(self, value):
        self._km = value

    @property
    def vmax(self):
        """float: Michaelis-Menten constant."""
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = value



class MassActionLaw(ReactionBase):
    """Parameters for Mass Action Law reaction model."""

    _type = "Mass Action Law"

    forward_rate = UnsignedFloat()
    reverse_rate = UnsignedFloat()

    _parameters = [
        'forward_rate',
        'reverse_rate',
    ]

class Cristilazation(ReactionBase):
    pass

class ActivatedSludgeModel(ReactionBase):
    pass
