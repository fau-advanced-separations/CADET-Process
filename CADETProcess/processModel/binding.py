from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Bool, String, Integer, \
    UnsignedInteger, UnsignedFloat, DependentlySizedUnsignedList

from .componentSystem import ComponentSystem

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
    is_kinetic : bool
        If False, adsorption is assumed to be in rapid equilibriu.
        The default is True.
    parameters : dict
        dict with parameter values.

    """
    name = String()
    is_kinetic = Bool(default=True)
    n_states = Integer(lb=1, ub=1, default=1)

    _parameter_names = ['is_kinetic']

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
        """dict: Dictionary with parameter values."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            setattr(self, param, value)

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
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)

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
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    capacity = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
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
    equilibrium_constant = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    driving_force_coefficient = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'), default=1)
    capacity = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))

    _parameter_names = BindingBaseClass._parameter_names + [
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
    adsorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    desorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'), default=1)
    capacity = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    n_states = UnsignedInteger()

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'n_states'
    ]

    def __init__(self, *args, n_states=2, **kwargs):
        self.n_states = n_states

        super().__init__(*args, **kwargs)

    @property
    def n_total_states(self):
        return self.n_comp * self.n_states

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
    equilibrium_constant = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    driving_force_coefficient = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'), default=1)
    capacity = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    n_states = UnsignedInteger()

    _parameter_names = BindingBaseClass._parameter_names + [
        'equilibrium_constant',
        'driving_force_coefficient',
        'capacity',
        'n_states'
    ]

    def __init__(self, *args, n_states=2, **kwargs):
        self.n_states = n_states

        super().__init__(*args, **kwargs)

    @property
    def n_total_states(self):
        return self.n_comp * self.n_states

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
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).

    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat()
    reference_solid_phase_conc = UnsignedFloat()

    _parameter_names = BindingBaseClass._parameter_names + [
            'adsorption_rate',
            'desorption_rate',
            'characteristic_charge',
            'steric_factor',
            'capacity',
            'reference_liquid_phase_conc',
            'reference_solid_phase_conc'
        ]


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

    def __init__(self, *args, **kwargs):
        self.adsorption_rate = [0.0] * self.n_comp
        self.desorption_rate = [0.0] * self.n_comp
        self.capacity = [0.0] * self.n_comp
        self.antilangmuir = [0.0] * self.n_comp

        super().__init__(*args, **kwargs)


class KumarMultiComponentLangmuir(BindingBaseClass):
    """Kumar Multi Component Langmuir adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats.
        Desorption rate constants.
    capacity : list of unsigned floats.
        Maximum adsoprtion capacities.
    characteristic_charge: list of unsigned floats.
        Salt exponents/characteristic charges.
    activation_temp : list of unsigned floats.
        Activation temperatures.
    temperature : unsigned float.
        Temperature.

    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp', default=1)
    activation_temp = DependentlySizedUnsignedList(dep='n_comp')
    temperature = UnsignedFloat()

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'capacity',
        'characteristic_charge',
        'activation_temp',
        'temperature'
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
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_1_1 = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_2_1 = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
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
    capacity : list of unsigned floats.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : Parmater
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : Parmater
        Reference liquid phase concentration (optional, default value = 1.0).

    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    adsorption_rate_dimerization = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    reference_liquid_phase_conc = DependentlySizedUnsignedList(dep='n_comp')
    reference_solid_phase_conc = DependentlySizedUnsignedList(dep='n_comp')

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


class BiStericMassAction(BindingBaseClass):
    """ Bi Steric Mass Action adsoprtion isotherm.

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
    capacity : list of unsigned floats.
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding site types.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration for each binding site type or one
        value for all types (optional, default value = 1.0).
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration for each binding site type or one
        value for all types (optional, default value = 1.0).

    """
    adsorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    adsorption_rate_dimerization = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    desorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    characteristic_charge = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    steric_factor = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    capacity = DependentlySizedUnsignedList(dep='n_states')
    reference_liquid_phase_conc = DependentlySizedUnsignedList(dep='n_states', default=1)
    reference_solid_phase_conc = DependentlySizedUnsignedList(dep='n_states', default=1)

    _parameter_names = BindingBaseClass._parameter_names + [
        'adsorption_rate',
        'desorption_rate',
        'characteristic_charge',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc'
    ]

    def __init__(self, *args, n_states=2, **kwargs):
        self.n_states = n_states
        super().__init__(*args, **kwargs)


class MultistateStericMassAction(BindingBaseClass):
    """ Multistate Steric Mass Action adsoprtion isotherm.

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
    capacity : list of unsigned floats.
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration (optional, default value = 1.0).

    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    conversion_rate = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    reference_liquid_phase_conc = DependentlySizedUnsignedList(dep='n_comp', default=1)
    reference_solid_phase_conc = DependentlySizedUnsignedList(dep='n_comp', default=1)

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


class SimplifiedMultistateSteric_Mass_Action(BindingBaseClass):
    """ Simplified multistate Steric Mass Action adsoprtion isotherm.

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
    capacity : list of unsigned floats.
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
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge_first = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge_last = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_modifiers_charge = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor_first = DependentlySizedUnsignedList(dep='n_comp')
    steric_factor_last = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_modifiers_steric = DependentlySizedUnsignedList(dep='n_comp')
    capacity = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_weak_stronger = DependentlySizedUnsignedList(dep='n_comp')
    linear_exchange_ws = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_exchange_ws = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_stronger_weak = DependentlySizedUnsignedList(dep='n_comp')
    linear_exchange_sw = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_exchange_sw = DependentlySizedUnsignedList(dep='n_comp')
    reference_liquid_phase_conc = DependentlySizedUnsignedList(dep='n_comp')
    reference_solid_phase_conc = DependentlySizedUnsignedList(dep='n_comp')

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
    quadratic_factor = DependentlySizedUnsignedList(dep='n_comp')

    _parameter_names = BindingBaseClass._parameter_names + [
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
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    adsorption_rate_linear = DependentlySizedUnsignedList(dep='n_comp')
    adsorption_rate_quadratic = DependentlySizedUnsignedList(dep='n_comp', default=0)
    adsorption_rate_cubic = DependentlySizedUnsignedList(dep='n_comp', default=0)
    adsorption_rate_salt = DependentlySizedUnsignedList(dep='n_comp', default=0)
    adsorption_rate_protein = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    desorption_rate_linear = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate_quadratic = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate_cubic = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate_salt = DependentlySizedUnsignedList(dep='n_comp', default=0)
    desorption_rate_protein = DependentlySizedUnsignedList(dep='n_comp', default=0)
    characteristic_charge = DependentlySizedUnsignedList(dep='_break_length', default=1)
    characteristic_charge_linear = DependentlySizedUnsignedList(dep='characteristic_charge', default=0)
    characteristic_charge_quadratic = DependentlySizedUnsignedList(dep='characteristic_charge', default=0)
    characteristic_charge_cubic = DependentlySizedUnsignedList(dep='characteristic_charge', default=0)
    characteristic_charge_breaks = DependentlySizedUnsignedList(dep='_break_length', default=0)
    steric_factor = DependentlySizedUnsignedList(dep='n_comp')
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat()
    reference_solid_phase_conc = UnsignedFloat()

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
        'characteristic_charge',
        'characteristic_charge_linear',
        'characteristic_charge_quadratic',
        'characteristic_charge_cubic',
        'characteristic_charge_breaks',
        'steric_factor',
        'capacity',
        'reference_liquid_phase_conc',
        'reference_solid_phase_conc',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _multiples(self):
        pass

    @property
    def _break_length(self):
        return len(self.characteristic_charge) + 1
