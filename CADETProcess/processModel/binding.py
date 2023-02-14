from warnings import warn

import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Bool, String, \
    RangedInteger, UnsignedInteger, UnsignedFloat, DependentlySizedList, \
    DependentlySizedRangedIntegerList, DependentlySizedUnsignedIntegerList, \
    DependentlySizedUnsignedList, DependentlyModulatedUnsignedList

from .componentSystem import ComponentSystem

__all__ = [
    'BindingBaseClass',
    'NoBinding',
    'Linear',
    'Langmuir',
    'LangmuirLDF',
    'BiLangmuir',
    'BiLangmuirLDF',
    'FreundlichLDF',
    'StericMassAction',
    'AntiLangmuir',
    'Spreading',
    'SelfAssociation',
    'BiStericMassAction',
    'MultistateStericMassAction',
    'SimplifiedMultistateStericMassAction',
    'Saska',
    'GeneralizedIonExchange',
]


@frozen_attributes
class BindingBaseClass(metaclass=StructMeta):
    """Abstract base class for parameters of binding models.

    Attributes
    ----------
    name : String
        name of the binding model.
    component_system : ComponentSystem
        system of components.
    n_comp : int
        number of components.
    n_binding_sites : int
        Number of binding sites.
        Relevant for Multi-Site isotherms such as Bi-Langmuir.
        The default is 1.
    bound_states : list of unsigned integers.
        Number of binding sites per component.
    non_binding_component_indices : list
        (Hardcoded) list of non binding modifier components (e.g. pH).
    is_kinetic : bool
        If False, adsorption is assumed to be in rapid equilibriu.
        The default is True.
    parameters : dict
        dict with parameter values.

    """

    name = String()
    is_kinetic = Bool(default=True)

    n_binding_sites = RangedInteger(lb=1, ub=1, default=1)
    _bound_states = DependentlySizedRangedIntegerList(
        dep=('n_binding_sites', 'n_comp'), lb=0, ub=1, default=1
    )
    non_binding_component_indices = []

    _parameter_names = ['is_kinetic']
    _required_parameters = []

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
    def bound_states(self):
        bound_states = self._bound_states
        for i in self.non_binding_component_indices:
            bound_states[i] = 0
        return bound_states

    @bound_states.setter
    def bound_states(self, bound_states):
        indices = self.non_binding_component_indices
        if any(bound_states[i] > 0 for i in indices):
            raise CADETProcessError(
                "Cannot set bound state for non-binding component."
            )

        self._bound_states = bound_states

    @property
    def n_bound_states(self):
        return sum(self.bound_states)

    @property
    def parameters(self):
        """dict: Dictionary with parameter values."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            setattr(self, param, value)

    @property
    def required_parameters(self):
        return self._required_parameters

    @property
    def missing_parameters(self):
        """list: List of missing parameters."""
        missing_parameters = []
        for param in self.required_parameters:
            if getattr(self, param) is None:
                missing_parameters.append(param)

        return missing_parameters

    def check_required_parameters(self):
        """Check if al required parameters are set.

        Returns
        -------
        bool
            True if all parameters are set. False otherwise.

        Raises
        ------
        Warning
            If any required parameters are missing.

        """
        if len(self.missing_parameters) == 0:
            return True
        else:
            for param in self.missing_parameters:
                warn(f'Binding Model {self.name}: Missing parameter "{param}".')
            return False

    def __repr__(self):
        return f"{self.__class__.__name__}(\
            component_system={self.component_system}, name={self.name})')"

    def __str__(self):
        if self.name is None:
            return self.__class__.__name__
        return self.name


class NoBinding(BindingBaseClass):
    """Dummy class for units that do not experience binging behavior.

    The number of components is set to zero for this class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(ComponentSystem(), name='NoBinding')


class Linear(BindingBaseClass):
    """Parameters for Linear binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants.

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate', 'desorption_rate'
    ]


class Langmuir(BindingBaseClass):
    """Parameters for Multi Component Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants.
    capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate', 'desorption_rate', 'capacity'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate', 'desorption_rate', 'capacity'
    ]


class LangmuirLDF(BindingBaseClass):
    """Parameters for Multi Component Langmuir binding model.

    Attributes
    ----------
    equilibrium_constant : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    driving_force_coefficient : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants
    capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.

    """

    equilibrium_constant = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_states')
    )
    driving_force_coefficient = DependentlySizedUnsignedList(
        dep=('n_comp', 'n_states')
    )
    capacity = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))

    _parameter_names = BindingBaseClass._parameter_names + [
        'equilibrium_constant',
        'driving_force_coefficient',
        'capacity',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'equilibrium_constant',
        'driving_force_coefficient',
        'capacity',
    ]


class BiLangmuir(BindingBaseClass):
    """Parameters for Multi Component Bi-Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants
    capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.

    """

    n_binding_sites = UnsignedInteger(default=2)

    adsorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    desorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    capacity = DependentlySizedUnsignedList(dep='n_bound_states')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
    ]

    def __init__(self, *args, n_binding_sites=2, **kwargs):
        self.n_binding_sites = n_binding_sites

        super().__init__(*args, **kwargs)


class BiLangmuirLDF(BindingBaseClass):
    """Parameters for Multi Component Bi-Langmuir binding model.

    Attributes
    ----------
    equilibrium_constant : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    driving_force_coefficient : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants
    capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.

    """

    n_binding_sites = UnsignedInteger(default=2)

    equilibrium_constant = DependentlySizedUnsignedList(dep='n_bound_states')
    driving_force_coefficient = DependentlySizedUnsignedList(dep='n_bound_states')
    capacity = DependentlySizedUnsignedList(dep='n_bound_states')

    _parameter_names = BindingBaseClass._parameter_names + [
        'equilibrium_constant',
        'driving_force_coefficient',
        'capacity',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'equilibrium_constant',
        'driving_force_coefficient',
        'capacity',
    ]

    def __init__(self, *args, n_binding_sites=2, **kwargs):
        self.n_binding_sites = n_binding_sites

        super().__init__(*args, **kwargs)


class FreundlichLDF(BindingBaseClass):
    """Parameters for the Freundlich isotherm model.

    Attributes
    ----------
    driving_force_coefficient : list of unsigned floats.
        Adsorption rate constants. Length depends on n_comp.
    freundlich_coefficient : list of unsigned floats.
        Freundlich coefficient for each component. Length depends on n_comp.
    exponent : list of unsigned floats.
        Exponent for each component. Length depends on n_comp.

    """

    driving_force_coefficient = DependentlySizedUnsignedList(dep='n_comp')
    freundlich_coefficient = DependentlySizedUnsignedList(dep='n_comp')
    exponent = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'driving_force_coefficient',
        'freundlich_coefficient',
        'exponent',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'driving_force_coefficient',
        'freundlich_coefficient',
        'exponent',
    ]


class StericMassAction(BindingBaseClass):
    """Parameters for Steric Mass Action Law binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants.
    characteristic_charge : list of unsigned floats. Length depends on n_comp.
        The characteristic charge of the protein: The number sites v that
        protein interacts on the resin surface.
    steric_factor : list of unsigned floats. Length depends on n_comp.
        Steric factors of the protein: The number of sites o on the surface
        that are shileded by the protein and prevented from exchange with salt
        counterions in solution.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default is 1.0
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default is 1.0

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1.0)
    reference_solid_phase_conc = UnsignedFloat(default=1.0)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
    ]

    @property
    def adsorption_rate_transformed(self):
        nu = np.array(self.characteristic_charge)
        return \
            self.adsorption_rate / \
            self.reference_solid_phase_conc**(-nu)

    @adsorption_rate_transformed.setter
    def adsorption_rate_transformed(self, adsorption_rate_transformed):
        nu = np.array(self.characteristic_charge)
        self.adsorption_rate = \
            adsorption_rate_transformed * \
            self.reference_solid_phase_conc**(-nu)

    @property
    def desorption_rate_transformed(self):
        nu = np.array(self.characteristic_charge)
        return \
            self.desorption_rate / \
            self.reference_liquid_phase_conc**(-nu)

    @desorption_rate_transformed.setter
    def desorption_rate_transformed(self, desorption_rate_transformed):
        nu = np.array(self.characteristic_charge)
        self.desorption_rate = \
            desorption_rate_transformed * \
            self.reference_liquid_phase_conc**(-nu)


class AntiLangmuir(BindingBaseClass):
    """Multi Component Anti-Langmuir adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on n_comp.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants. Length depends on n_comp.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on n_comp.
    antilangmuir : list of unsigned floats, optional.
        Anti-Langmuir coefficients. Length depends on n_comp.

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    antilangmuir = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'antilangmuir'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'antilangmuir'
    ]


class Spreading(BindingBaseClass):
    """Multi Component Spreading adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats.
        Desorption rate constants.
    capacity : list of unsigned floats.
        Maximum adsoprtion capacities in state-major ordering.
    exchange_from_1_2 : list of unsigned floats.
        Exchange rates from the first to the second bound state.
    exchange_from_2_1 : list of unsigned floats.
        Exchange rates from the second to the first bound state.

    """

    n_binding_sites = RangedInteger(lb=2, ub=2, default=2)

    adsorption_rate = DependentlySizedUnsignedList(dep='n_total_bound')
    desorption_rate = DependentlySizedUnsignedList(dep='n_total_bound')
    capacity = DependentlySizedUnsignedList(dep='n_total_bound')
    exchange_from_1_2 = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_2_1 = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'activation_temp',
        'capacity',
        'exchange_from_1_2',
        'exchange_from_2_1'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'activation_temp',
        'capacity',
        'exchange_from_1_2',
        'exchange_from_2_1'
    ]


class MobilePhaseModulator(BindingBaseClass):
    """Mobile Phase Modulator adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats.
        Desorption rate constants.
    capacity : list of unsigned floats.
        Maximum adsorption capacities.
    ion_exchange_characteristic : list of unsigned floats.
        Parameters describing the ion-exchange characteristics (IEX).
    hydrophobicity : list of unsigned floats.
        Parameters describing the hydrophobicity (HIC).

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    ion_exchange_characteristic = DependentlySizedUnsignedList(dep='n_comp')
    beta = ion_exchange_characteristic
    hydrophobicity = DependentlySizedUnsignedList(dep='n_comp')
    gamma = hydrophobicity

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'ion_exchange_characteristic',
        'hydrophobicity',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'ion_exchange_characteristic',
        'hydrophobicity',
    ]


class ExtendedMobilePhaseModulator(BindingBaseClass):
    """Mobile Phase Modulator adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats.
        Desorption rate constants.
    capacity : list of unsigned floats.
        Maximum adsorption capacities.
    ion_exchange_characteristic : list of unsigned floats.
        Parameters describing the ion-exchange characteristics (IEX).
    hydrophobicity : list of unsigned floats.
        Parameters describing the hydrophobicity (HIC).
    component_mode : list of unsigned floats.
        Mode of each component;
        0 denotes the modifier component,
        1 is linear binding,
        2 is modified Langmuir binding.

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    ion_exchange_characteristic = DependentlySizedUnsignedList(dep='n_comp')
    beta = ion_exchange_characteristic
    hydrophobicity = DependentlySizedUnsignedList(dep='n_comp')
    gamma = hydrophobicity
    component_mode = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'ion_exchange_characteristic',
        'hydrophobicity',
        'component_mode',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'ion_exchange_characteristic',
        'hydrophobicity',
        'component_mode',
    ]


class SelfAssociation(BindingBaseClass):
    """Self Association adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants.
    adsorption_rate_dimerization : list of unsigned floats.
        Adsorption rate constants of dimerization.
    desorption_rate : list of unsigned floats.
        Desorption rate constants.
    characteristic_charge : list of unsigned floats.
        The characteristic charge v of the protein.
    steric_factor : list of unsigned floats.
        Steric factor of of the protein.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float
        Reference liquid phase concentration (optional, default value = 1.0).
        The default = 1.0
    reference_solid_phase_conc : unsigned float
        Reference liquid phase concentration (optional)
        The default = 1.0

    """

    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    adsorption_rate_dimerization = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1.0)
    reference_solid_phase_conc = UnsignedFloat(default=1.0)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'adsorption_rate_dimerization',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'adsorption_rate_dimerization',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
    ]


class BiStericMassAction(BindingBaseClass):
    """Bi Steric Mass Action adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants in state-major ordering.
    desorption_rate : list of unsigned floats.
        Desorption rate constants in state-major ordering.
    characteristic_charge : list of unsigned floats.
        Characteristic charges v(i,j) of the it-h protein with respect to the
        j-th binding site type in state-major ordering.
    steric_factor : list of unsigned floats.
        Steric factor o (i,j) of the it-h protein with respect to the j-th
        binding site type in state-major ordering.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration for each binding site type or one
        value for all types.
        The default is 1.0
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration for each binding site type or one
        value for all types.
        The default is 1.0

    """

    n_binding_sites = UnsignedInteger(default=2)

    adsorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    adsorption_rate_dimerization = DependentlySizedUnsignedList(
        dep='n_bound_states'
    )
    desorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_bound_states')
    steric_factor = DependentlySizedUnsignedList(dep='n_bound_states')
    capacity = DependentlySizedUnsignedList(dep='n_binding_sites')
    reference_liquid_phase_conc = DependentlySizedUnsignedList(
        dep='n_binding_sites', default=1
    )
    reference_solid_phase_conc = DependentlySizedUnsignedList(
        dep='n_binding_sites', default=1
    )

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
    ]

    def __init__(self, *args, n_states=2, **kwargs):
        self.n_states = n_states
        super().__init__(*args, **kwargs)


class MultistateStericMassAction(BindingBaseClass):
    """Multistate Steric Mass Action adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants of the components to different bound states
        in component-major ordering.
    desorption_rate : list of unsigned floats.
        Desorption rate constants of the components to different bound states
        in component-major ordering.
    characteristic_charge : list of unsigned floats.
        Characteristic charges of the components to different bound states in
        component-major ordering.
    steric_factor : list of unsigned floats.
        Steric factor of the components to different bound states in
        component-major ordering.
    conversion_rate : list of unsigned floats.
        Conversion rates between different bound states in
        component-major ordering.
        Length: $sum_{i=1}^{n_{comp}} n_{bound, i}$
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default = 1.0
    reference_solid_phase_conc : unsigned float, optional
        Reference solid phase concentration.
        The default = 1.0

    """

    bound_states = DependentlySizedUnsignedIntegerList(
        dep=('n_binding_sites', 'n_comp'), default=1
    )

    adsorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    desorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_bound_states')
    steric_factor = DependentlySizedUnsignedList(dep='n_bound_states')
    conversion_rate = DependentlySizedUnsignedList(dep='_conversion_entries')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'conversion_rate',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'conversion_rate',
        'capacity',
    ]

    @property
    def _conversion_entries(self):
        n = 0
        for state in self.bound_states[1:]:
            n += state**2

        return n


class SimplifiedMultistateStericMassAction(BindingBaseClass):
    """Simplified multistate Steric Mass Action adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate :list of unsigned floats.
        Adsorption rate constants of the components to different bound states
        in component-major ordering.
    desorption_rate : list of unsigned floats.
        Desorption rate constants of the components to different bound states
        in component-major ordering.
    characteristic_charge_first : list of unsigned floats.
        Characteristic charges of the components in the first (weakest) bound
        state.
    characteristic_charge_last : list of unsigned floats.
        Characteristic charges of the components in the last (strongest) bound
        state.
    quadratic_modifiers_charge : list of unsigned floats.
        Quadratic modifiers of the characteristic charges of the different
        components depending on the index of the bound state.
    steric_factor_first : list of unsigned floats.
        Steric factor of the components in the first (weakest) bound state.
    steric_factor_last : list of unsigned floats.
        Steric factor of the components in the last (strongest) bound state.
    quadratic_modifiers_steric : list of unsigned floats.
        Quadratic modifiers of the sterif factors of the different components
        depending on the index of the bound state.
    capacity : unsigned floats.
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding sites available on the resin surface.
    exchange_from_weak_stronger : list of unsigned floats.
        Exchangde rated from a weakly bound state to the next stronger bound
        state.
    linear_exchange_ws : list of unsigned floats.
        Linear exchange rate coefficients from a weakly bound state to the next
        stronger bound state.
    quadratic_exchange_ws : list of unsigned floats.
        Quadratic exchange rate coefficients from a weakly bound state to the
        next stronger bound state.
    exchange_from_stronger_weak : list of unsigned floats.
        Exchange rate coefficients from a strongly bound state to the next
        weaker bound state.
    linear_exchange_sw : list of unsigned floats.
        Linear exchange rate coefficients from a strongly bound state to the
        next weaker bound state.
    quadratic_exchange_sw : list of unsigned floats.
        Quadratic exchange rate coefficients from a strongly bound state to the
        next weaker bound state.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration (optional, default value = 1.0).

    """

    bound_states = DependentlySizedUnsignedIntegerList(
        dep=('n_binding_sites', 'n_comp'), default=1
    )

    adsorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    desorption_rate = DependentlySizedUnsignedList(dep='n_bound_states')
    characteristic_charge_first = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge_last = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_modifiers_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor_first = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor_last = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_modifiers_steric = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    exchange_from_weak_stronger = DependentlySizedUnsignedList(dep='n_comp')
    linear_exchange_ws = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_exchange_ws = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_stronger_weak = DependentlySizedUnsignedList(dep='n_comp')
    linear_exchange_sw = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_exchange_sw = DependentlySizedUnsignedList(dep='n_comp')
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge_first',
        'characteristic_charge_last',
        'quadratic_modifiers_charge',
        'steric_factor_first',
        'steric_factor_last',
        'quadratic_modifiers_steric',
        'capacity',
        'exchange_from_weak_stronger',
        'linear_exchange_ws',
        'quadratic_exchange_ws',
        'exchange_from_stronger_weak',
        'linear_exchange_sw',
        'quadratic_exchange_sw',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge_first',
        'characteristic_charge_last',
        'quadratic_modifiers_charge',
        'steric_factor_first',
        'steric_factor_last',
        'quadratic_modifiers_steric',
        'capacity',
        'exchange_from_weak_stronger',
        'linear_exchange_ws',
        'quadratic_exchange_ws',
        'exchange_from_stronger_weak',
        'linear_exchange_sw',
        'quadratic_exchange_sw',
    ]


class Saska(BindingBaseClass):
    """Quadratic Isotherm.

    Attributes
    ----------
    henry_const : list of unsigned floats.
        The Henry coefficient.
    quadratic_factor : list of unsigned floats.
        Quadratic factors.

    """

    henry_const = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_factor = DependentlySizedUnsignedList(dep=('n_comp', 'n_comp'))

    _parameter_names = BindingBaseClass._parameter_names + [
        'henry_const', 'quadratic_factor'
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'henry_const', 'quadratic_factor'
    ]


class GeneralizedIonExchange(BindingBaseClass):
    """Generalized Ion Exchange isotherm model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    adsorption_rate_linear : list of unsigned floats. Length depends on n_comp.
        Linear dependence coefficient of adsorption rate on modifier component
    adsorption_rate_quadratic : list of unsigned floats. Length depends on n_comp.
        Quadratic dependence coefficient of adsorption rate on modifier component.
    adsorption_rate_cubic : list of unsigned floats. Length depends on n_comp.
        Cubic dependence coefficient of adsorption rate on modifier component.
    adsorption_rate_salt : list of unsigned floats. Length depends on n_comp.
        Salt coefficient of adsorption rate;
        difference of water-protein and salt-protein interactions.
    adsorption_rate_protein : list of unsigned floats. Length depends on n_comp.
        Protein coefficient of adsorption rate;
        difference of water-protein and protein-protein interactions.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants.
    desorption_rate_linear : list of unsigned floats. Length depends on n_comp.
        Linear dependence coefficient of desorption rate on modifier component.
    desorption_rate_quadratic : list of unsigned floats. Length depends on n_comp.
        Quadratic dependence coefficient of desorption rate on modifier component.
    desorption_rate_cubic : list of unsigned floats. Length depends on n_comp.
        Cubic dependence coefficient of desorption rate on modifier component.
    desorption_rate_salt : list of unsigned floats. Length depends on n_comp.
        Salt coefficient of desorption rate;
        difference of water-protein and salt-protein interactions.
    desorption_rate_protein : list of unsigned floats. Length depends on n_comp.
        Protein coefficient of desorption rate;
        difference of water-protein and protein-protein interactions
    characteristic_charge : list of unsigned floats. Length depends on n_comp.
        The characteristic charge of the protein: The number sites v that
        protein interacts on the resin surface.
    characteristic_charge_linear : list of unsigned floats. Length depends on n_comp.
        Linear dependence coefficient of characteristic charge on modifier component.
    characteristic_charge_quadratic : list of unsigned floats. Length depends on n_comp.
        Quadratic dependence coefficient of characteristic charge on modifier component.
    characteristic_charge_cubic : list of unsigned floats. Length depends on n_comp.
        Cubic dependence coefficient of characteristic charge on modifier component .
    characteristic_charge_breaks : list of unsigned floats. Length depends on n_comp.
        Cubic dependence coefficient of characteristic charge on modifier component .
    steric_factor : list of unsigned floats. Length depends on n_comp.
        Steric factors of the protein: The number of sites o on the surface
        that are shileded by the protein and prevented from exchange with salt
        counterions in solution.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).

    """

    non_binding_component_indices = [1]

    adsorption_rate = DependentlySizedList(dep='n_comp')
    adsorption_rate_linear = DependentlySizedList(dep='n_comp')
    adsorption_rate_quadratic = DependentlySizedList(
        dep='n_comp', default=0
    )
    adsorption_rate_cubic = DependentlySizedList(
        dep='n_comp', default=0
    )
    adsorption_rate_salt = DependentlySizedList(
        dep='n_comp', default=0
    )
    adsorption_rate_protein = DependentlySizedList(
        dep='n_comp', default=0
    )
    desorption_rate = DependentlySizedList(dep='n_comp')
    desorption_rate_linear = DependentlySizedList(
        dep='n_comp', default=0
    )
    desorption_rate_quadratic = DependentlySizedList(
        dep='n_comp', default=0
    )
    desorption_rate_cubic = DependentlySizedList(
        dep='n_comp', default=0
    )
    desorption_rate_salt = DependentlySizedList(
        dep='n_comp', default=0
    )
    desorption_rate_protein = DependentlySizedList(
        dep='n_comp', default=0
    )
    characteristic_charge_breaks = DependentlyModulatedUnsignedList(
        dep='n_comp'
    )
    characteristic_charge = DependentlySizedList(
        dep=('n_pieces', 'n_comp'),
    )
    characteristic_charge_linear = DependentlySizedList(
        dep=('n_pieces', 'n_comp'), default=0
    )
    characteristic_charge_quadratic = DependentlySizedList(
        dep=('n_pieces', 'n_comp'), default=0
    )
    characteristic_charge_cubic = DependentlySizedList(
        dep=('n_pieces', 'n_comp'), default=0
    )
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'adsorption_rate_linear',
        'adsorption_rate_quadratic',
        'adsorption_rate_cubic',
        'adsorption_rate_salt',
        'adsorption_rate_protein',
        'desorption_rate',
        'desorption_rate_linear',
        'desorption_rate_quadratic',
        'desorption_rate_cubic',
        'desorption_rate_salt',
        'desorption_rate_protein',
        'characteristic_charge_breaks',
        'characteristic_charge',
        'characteristic_charge_linear',
        'characteristic_charge_quadratic',
        'characteristic_charge_cubic',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc',
    ]
    _required_parameters = BindingBaseClass._required_parameters + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
    ]

    @property
    def n_pieces(self):
        """int: Number of pieces for cubic polynomial description of nu."""
        if self.characteristic_charge_breaks is None:
            return 1

        n_pieces_all = len(self.characteristic_charge_breaks) - self.n_comp
        n_pieces_comp = int(n_pieces_all / self.n_comp)

        return n_pieces_comp
