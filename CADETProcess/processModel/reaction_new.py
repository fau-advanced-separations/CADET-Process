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
    Bool, String, SizedList, SizedNdArray, UnsignedInteger, UnsignedFloat, SizedUnsignedList, List
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

class InhibitionBase(Structure):
    """Abstract base class for inhibition reactions."""


    _name = String()
    _parameters = []

    def __init__(self, component_system, name=None, *args, **kwargs):
        self._component_system = component_system
        self._name = name
        super().__init__(*args, **kwargs)


class EnzyemeInhibtion(InhibitionBase):


    competitive_rate = List()
    uncompetitive_rate = List()
    noncompetitive_rate = List()

    substrate = String()
    inhibitors = List()

    _parameters = [
        'substrate',
        'inhibitors',
        'competitive_rate',
        'uncompetitive_rate',
        'noncompetitive_rate',
    ]

    def __init__(self, component_system,  name=None, *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)

        # Validate the types and components
        self.validate_all()

    def validate_all(self):
        """Perform all validations."""
        self.validate_required_parameters()
        self.validate_components(self._component_system)
        self.validate_inhibition_type()
        self.validate_rate_consistency()
        #? kann ein substrat auch ein inhibitor sein?

    def validate_required_parameters(self):
        """Validate that required parameters are provided."""
        if not self.substrate:
            raise ValueError("Substrate must be specified")

        if not self.inhibitors:
            raise ValueError("At least one inhibitor must be specified")

        # Check if at least one inhibition type is specified
        has_competitive = self.competitive_rate and len(self.competitive_rate) > 0
        has_uncompetitive = self.uncompetitive_rate and len(self.uncompetitive_rate) > 0
        has_noncompetitive = self.noncompetitive_rate and len(self.noncompetitive_rate) > 0

        if not (has_competitive or has_uncompetitive or has_noncompetitive):
            raise ValueError("At least one inhibition rate (competitive, uncompetitive, or noncompetitive) must be specified")

    def validate_components(self, component_system):
        """Validate, if substrate and inhibitors are correctly set"""
        if self.substrate not in component_system.species:
            raise ValueError(f"Substrate '{self.substrate}' not in component system. Available species: {component_system.species}")

        inhibitors_list = self.inhibitors if isinstance(self.inhibitors, list) else [self.inhibitors]

        for inhibitor in inhibitors_list:
            if inhibitor not in component_system.species:
                raise ValueError(f"Inhibitor '{inhibitor}' not in component system. Available species: {component_system.species}")

    def validate_inhibition_type(self):
        """Validate that only one type of inhibition is active at a time."""
        active_types = []

        if self.competitive_rate and len(self.competitive_rate) > 0:
            active_types.append('competitive')
        if self.uncompetitive_rate and len(self.uncompetitive_rate) > 0:
            active_types.append('uncompetitive')
        if self.noncompetitive_rate and len(self.noncompetitive_rate) > 0:
            active_types.append('noncompetitive')

        if len(active_types) > 1:
            raise ValueError(f"Multiple inhibition types specified: {active_types}. "
                         "Only one inhibition type is allowed", UserWarning)

    def validate_rate_consistency(self):
        """Validate that rates are consistent with number of inhibitors."""
        inhibitors_list = self.inhibitors if isinstance(self.inhibitors, list) else [self.inhibitors]
        n_inhibitors = len(inhibitors_list)

        for rate_type, rates in [
            ('competitive_rate', self.competitive_rate),
            ('uncompetitive_rate', self.uncompetitive_rate),
            ('noncompetitive_rate', self.noncompetitive_rate)
        ]:
            if rates and len(rates) > 0:
                if len(rates) != n_inhibitors:
                    raise ValueError(f"Number of {rate_type} values ({len(rates)}) must match "
                                   f"number of inhibitors ({n_inhibitors})")

                # Check for negative rates
                if any(rate < 0 for rate in rates):
                    raise ValueError(f"All {rate_type} values must be non-negative")

                # Warn about zero rates
                if any(rate == 0 for rate in rates):
                    warnings.warn(f"Zero values found in {rate_type}. "
                                 "This effectively disables inhibition for those inhibitors.", UserWarning)
    @property
    def inhibition_type(self):
        """Get the active inhibition type."""
        if self.competitive_rate and len(self.competitive_rate) > 0:
            return 'competitive'
        elif self.uncompetitive_rate and len(self.uncompetitive_rate) > 0:
            return 'uncompetitive'
        elif self.noncompetitive_rate and len(self.noncompetitive_rate) > 0:
            return 'noncompetitive'
        else:
            return 'none'

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

    km = List()
    vmax = UnsignedFloat()

    _inhibition_reactions = []
    _inhibited_substrate = []
    _parameters = [ 'km', 'vmax' ]

    def __init__(self, component_system, components, coefficients, name=None, *args, **kwargs):
        super().__init__(component_system, components, coefficients, name, *args, **kwargs)
        self._inhibition_reactions = []
        self._inhibited_substrate = []

        self.validate_reaction_setup()

    @property
    def inhibition_reactions(self) -> list[InhibitionBase]:
        """list[InhibitionBase]: List of inhibition reactions."""
        return self._inhibition_reactions

    def add_inhibition_reaction(self, inhibition_reaction):
        """Add inhibition reaction to the model."""
        if not isinstance(inhibition_reaction, InhibitionBase):
            raise TypeError('Expected InhibitionBase')

        # Check if the substrate already has an inhibition of the same type
        if inhibition_reaction.substrate in self._inhibited_substrate:
            warnings.warn(f"Substrate {inhibition_reaction.substrate} already has an inhibition of type {inhibition_reaction.type}. "
                          "Adding multiple inhibitions of the same type is not recommended.", UserWarning)
        else:
            self._inhibited_substrate.append(inhibition_reaction.substrate)
            self._inhibition_reactions.append(inhibition_reaction)


    @property
    def parameters(self):
        parameters = super().parameters

        parameters["inhibition"] = {
            inhibition.name: inhibition.parameters
            for inhibition in self._inhibition_reactions.parameters
        }

    def validate_reaction_setup(self):
        """Validate the basic reaction setup."""
        self.validate_components_coefficients()
        self.validate_parameters()
        self.validate_stoichiometry()

    def validate_components_coefficients(self):
        """Validate components and coefficients consistency."""
        if len(self.components) != len(self.coefficients):
            raise ValueError(f"Number of components ({len(self.components)}) must match "
                           f"number of coefficients ({len(self.coefficients)})")

        # Check if all components exist in component system
        for component in self.components:
            if component not in self.component_system.species:
                raise ValueError(f"Component '{component}' not in component system. "
                               f"Available species: {self.component_system.species}")

    def validate_parameters(self):
        """Validate Michaelis-Menten parameters."""
        # Validate km
        if not self.km:
            raise ValueError("km parameter must be specified")

        if any(k <= 0 for k in self.km):
            raise ValueError("All km values must be positive")

        # Validate vmax
        if self.vmax is None:
            raise ValueError("vmax parameter must be specified")

        if self.vmax <= 0:
            raise ValueError("vmax must be positive")

        # Check if number of km values matches substrates
        substrates = [comp for comp, coeff in zip(self.components, self.coefficients) if coeff < 0]
        if len(self.km) != len(substrates):
            warnings.warn(f"Number of km values ({len(self.km)}) does not match "
                         f"number of substrates ({len(substrates)}). "
                         "This may indicate a configuration issue.", UserWarning)

    def validate_stoichiometry(self):
        """Validate reaction stoichiometry."""
        if all(coeff >= 0 for coeff in self.coefficients):
            warnings.warn("No negative coefficients found. "
                         "This reaction has no substrates (only products).", UserWarning)

        if all(coeff <= 0 for coeff in self.coefficients):
            warnings.warn("No positive coefficients found. "
                         "This reaction has no products (only substrates).", UserWarning)


    def __str__(self):
        """String representation of the Michaelis-Menten reaction."""
        base_str = f"{self._type}"

        if self._inhibition_reactions:
            inhibition_types = [inh.inhibition_type for inh in self._inhibition_reactions]
            unique_types = list(set(inhibition_types))
            if len(unique_types) == 1:
                base_str += f" with {unique_types[0]} inhibition"
            else:
                base_str += f" with multiple inhibitions ({', '.join(unique_types)})"
        else:
            base_str += " without inhibition"

        return base_str

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
