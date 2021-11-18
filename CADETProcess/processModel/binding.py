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
    n_comp : UnsignedInteger
        number of components.
    parameters : dict
        dict with parameter values.
    name : String
        name of the binding model.
    """
    name = String()
    is_kinetic = Bool(default=True)
    n_states = Integer(lb=1, ub=1, default=1)

    def __init__(self, component_system, name=None):
        self._parameter_names = ['is_kinetic']
        self._parameters = {
            param: getattr(self, param)
            for param in self._parameter_names
        }

        self.component_system = component_system
        self.name = name

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('')
        self._component_system = component_system
    
    @property
    def n_comp(self):
        return self.component_system.n_comp
    
    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            setattr(self, param, value)


    def __repr__(self):
        return '{}(n_comp={}, name=\'{}\')'.format(
            self.__class__.__name__, self.n_comp, self.name
        )

    def __str__(self):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._parameter_names += ['adsorption_rate', 'desorption_rate']


class Langmuir(BindingBaseClass):
    """Parameters for Multi Component Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants.
    saturation_capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    saturation_capacity = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
            'adsorption_rate', 'desorption_rate', 'saturation_capacity'
        ]


class BiLangmuir(BindingBaseClass):
    """Parameters for Multi Component Bi-Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats. Length depends on n_comp.
        Adsorption rate constants.
    desorption_rate : list of unsigned floats. Length depends on n_comp.
        Desorption rate constants
    saturation_capacity : list of unsigned floats. Length depends on n_comp.
        Maximum adsorption capacities.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    desorption_rate = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'), default=1)
    saturation_capacity = DependentlySizedUnsignedList(dep=('n_comp', 'n_states'))
    n_states = UnsignedInteger()

    def __init__(self, *args, n_states=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_states = n_states

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate',
            'saturation_capacity',
            'n_states'
        ]
        
    @property
    def n_total_states(self):
        return self.n_comp * self.n_states


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
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
    maximum_adsorption_capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on n_comp.
    antilangmuir : list of unsigned floats, optional.
        Anti-Langmuir coefficients. Length depends on n_comp.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    maximum_adsorption_capacity = DependentlySizedUnsignedList(dep='n_comp')
    antilangmuir = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adsorption_rate = [0.0] * self.n_comp
        self.desorption_rate = [0.0] * self.n_comp
        self.maximum_adsorption_capacity = [0.0] * self.n_comp
        self.antilangmuir = [0.0] * self.n_comp

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate',
            'maximum_adsorption_capacity',
            'antilangmuir'
        ]


class KumarMultiComponentLangmuir(BindingBaseClass):
    """Kumar Multi Component Langmuir adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants.
    desorption_rate : Parameter
        Desorption rate constants.
    maximum_adsorption_capacity : Parameter
        Maximum adsoprtion capacities.
    characteristic_charge: Parameter
        Salt exponents/characteristic charges.
    activation_temp : Parameter
        Activation temperatures.
    temperature : unsigned float.
        Temperature.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    maximum_adsorption_capacity = DependentlySizedUnsignedList(dep='n_comp')
    characteristic_charge = DependentlySizedUnsignedList(dep='n_comp', default=1)
    activation_temp = DependentlySizedUnsignedList(dep='n_comp')
    temperature = UnsignedFloat()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate', 
            'maximum_adsorption_capacity', 
            'characteristic_charge',
            'activation_temp',
            'temperature'
        ]


class Spreading(BindingBaseClass):
    """Multi Component Spreading adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants.
    desorption_rate : Parameter
        Desorption rate constants.
    maximum_adsorption_capacity : Parameter
        Maximum adsoprtion capacities in state-major ordering.
    exchange_from_1_2 : Parameter
        Exchange rates from the first to the second bound state.
    exchange_from_2_1 : Parameter
        Exchange rates from the second to the first bound state.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp', default=1)
    maximum_adsorption_capacity = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_1_1 = DependentlySizedUnsignedList(dep='n_comp')
    exchange_from_2_1 = DependentlySizedUnsignedList(dep='n_comp')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate',
            'activation_temp',
            'maximum_adsorption_capacity',
            'exchange_from_1_2',
            'exchange_from_2_1'
        ]


class MobilePhaseModulator(BindingBaseClass):
    """Mobile Phase Modulator adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants.
    desorption_rate : Parameter
        Desorption rate constants.
    maximum_adsorption_capacity : Parameter
        Maximum adsorption capacities.
    ion_exchange_characteristic : Parameter
        Parameters describing the ion-exchange characteristics (IEX).
    hydrophobicity : Parameter
        Parameters describing the hydrophobicity (HIC).
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    maximum_adsorption_capacity = DependentlySizedUnsignedList(dep='n_comp')
    ion_exchange_characteristic = DependentlySizedUnsignedList(dep='n_comp')
    beta = ion_exchange_characteristic
    hydrophobicity = DependentlySizedUnsignedList(dep='n_comp')
    gamma = hydrophobicity
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate',
            'maximum_adsorption_capacity',
            'ion_exchange_characteristic',
            'hydrophobicity',
        ]

class ExtendedMobilePhaseModulator(BindingBaseClass):
    """Mobile Phase Modulator adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants.
    desorption_rate : Parameter
        Desorption rate constants.
    maximum_adsorption_capacity : Parameter
        Maximum adsorption capacities.
    ion_exchange_characteristic : Parameter
        Parameters describing the ion-exchange characteristics (IEX).
    hydrophobicity : Parameter
        Parameters describing the hydrophobicity (HIC).
    component_mode : Parameter
        Mode of each component; 
        0 denotes the modifier component, 
        1 is linear binding, 
        2 is modified Langmuir binding.
    """
    adsorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    desorption_rate = DependentlySizedUnsignedList(dep='n_comp')
    maximum_adsorption_capacity = DependentlySizedUnsignedList(dep='n_comp')
    ion_exchange_characteristic = DependentlySizedUnsignedList(dep='n_comp')
    beta = ion_exchange_characteristic
    hydrophobicity = DependentlySizedUnsignedList(dep='n_comp')
    gamma = hydrophobicity
    component_mode = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
            'adsorption_rate',
            'desorption_rate',
            'maximum_adsorption_capacity',
            'ion_exchange_characteristic',
            'hydrophobicity',
            'component_mode',
        ]
        

class SelfAssociation(BindingBaseClass):
    """Self Association adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants.
    adsorption_rate_dimerization : Parameter
        Adsorption rate constants of dimerization.
    desorption_rate : Parameter
        Desorption rate constants.
    characteristic_charge : Parameter
        The characteristic charge v of the protein.
    steric_factor : Parameter
        Steric factor of of the protein.
    capacity : Parameter
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
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
    adsorption_rate : Parameter
        Adsorption rate constants in state-major ordering.
    desorption_rate : Parameter
        Desorption rate constants in state-major ordering.
    characteristic_charge : Parameter
        Characteristic charges v(i,j) of the it-h protein with respect to the
        j-th binding site type in state-major ordering.
    steric_factor : Parameter
        Steric factor o (i,j) of the it-h protein with respect to the j-th
        binding site type in state-major ordering.
    capacity : Parameter
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding site types.
    reference_liquid_phase_conc : Parameter
        Reference liquid phase concentration for each binding site type or one
        value for all types (optional, default value = 1.0).
    reference_solid_phase_conc : Parameter
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
    
    def __init__(self, *args, n_states=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_states = n_states

        self._parameter_names += [
            'adsorption_rate', 
            'desorption_rate', 
            'characteristic_charge',
            'steric_factor',
            'capacity',
            'reference_liquid_phase_conc',
            'reference_solid_phase_conc'
        ]


class MultistateStericMassAction(BindingBaseClass):
    """ Multistate Steric Mass Action adsoprtion isotherm.

    Attributes
    ----------
    adsorption_rate : Parameter
        Adsorption rate constants of the components to different bound states
        in component-major ordering.
    desorption_rate : Parameter
        Desorption rate constants of the components to different bound states
        in component-major ordering.
    characteristic_charge : Parameter
        Characteristic charges of the components to different bound states in
        component-major ordering.
    steric_factor : Parameter
        Steric factor of the components to different bound states in
        component-major ordering.
    conversion_rate : Parameter
        Conversion rates between different bound states in
        component-major ordering.
    capacity : Parameter
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : Parameter
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : Parameter
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
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
    adsorption_rate :Parameter
        Adsorption rate constants of the components to different bound states
        in component-major ordering.
    desorption_rate : Parameter
        Desorption rate constants of the components to different bound states
        in component-major ordering.
    characteristic_charge_first : Parameter
        Characteristic charges of the components in the first (weakest) bound
        state.
    characteristic_charge_last : Parameter
        Characteristic charges of the components in the last (strongest) bound
        state.
    quadratic_modifiers_charge : Parameter
        Quadratic modifiers of the characteristic charges of the different
        components depending on the index of the bound state.
    steric_factor_first : Parameter
        Steric factor of the components in the first (weakest) bound state.
    steric_factor_last : Parameter
        Steric factor of the components in the last (strongest) bound state.
    quadratic_modifiers_steric : Parameter
        Quadratic modifiers of the sterif factors of the different components
        depending on the index of the bound state.
    capacity : Parameter
        Stationary phase capacity (monovalent salt counterions): The total
        number of binding sites available on the resin surface.
    exchange_from_weak_stronger : Parameter
        Exchangde rated from a weakly bound state to the next stronger bound
        state.
    linear_exchange_ws : Parameter
        Linear exchange rate coefficients from a weakly bound state to the next
        stronger bound state.
    quadratic_exchange_ws : Parameter
        Quadratic exchange rate coefficients from a weakly bound state to the
        next stronger bound state.
    exchange_from_stronger_weak : Parameter
        Exchange rate coefficients from a strongly bound state to the next
        weaker bound state.
    linear_exchange_sw : Parameter
        Linear exchange rate coefficients from a strongly bound state to the
        next weaker bound state.
    quadratic_exchange_sw : Parameter
        Quadratic exchange rate coefficients from a strongly bound state to the
        next weaker bound state.
    reference_liquid_phase_conc : Parameter
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : Parameter
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += [
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
    """ Multistate Steric Mass Action adsoprtion isotherm.

    Attributes
    ----------
    henry_const : Parameter
        The Henry coefficient.
    quadratic_factor : Parameter
        Quadratic factors.
    """
    henry_const = DependentlySizedUnsignedList(dep='n_comp')
    quadratic_factor = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameter_names += ['henry_const', 'quadratic_factor']