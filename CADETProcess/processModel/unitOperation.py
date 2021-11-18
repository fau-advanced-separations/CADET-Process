import math
import numpy as np

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    String, Switch,
    UnsignedInteger, UnsignedFloat,
    DependentlySizedUnsignedList, DependentlySizedNdArray,
    Polynomial, NdPolynomial
)

from .componentSystem import ComponentSystem
from .binding import BindingBaseClass, NoBinding
from .reaction import ReactionBaseClass, NoReaction
from .discretization import (
    NoDiscretization, LRMDiscretizationFV, LRMPDiscretizationFV, GRMDiscretizationFV
)

@frozen_attributes
class UnitBaseClass(metaclass=StructMeta):
    """Base class for all UnitOperation classes.

    A UnitOperation object stores model parameters and states of a unit. 
    Every unit operation can be assotiated with a binding behavior and a 
    reaction model.
    UnitOperations can be connected in a FlowSheet. 

    Attributes
    ----------
    n_comp : UnsignedInteger
        Number of components in a system.
    parameters : list
        list of parameter names.
    name : String
        name of the unit_operation.
    binding_model : BindingBaseClass
        binding behavior of the unit. Defaults to NoBinding.
    
    See also
    --------
    FlowSheet
    CADETProcess.binding
    CADETProcess.reaction
    """
    name = String()

    _parameter_names = []
    _section_dependent_parameters = []
    _polynomial_parameters = []
    _initial_state = []
    
    supports_bulk_reaction = False
    supports_particle_reaction = False
    
    def __init__(self, component_system, name):
        self.name = name
        self.component_system = component_system

        self._binding_model = NoBinding()

        self._bulk_reaction_model = NoReaction()
        self._particle_reaction_model = NoReaction()

        self._discretization = NoDiscretization()
        
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
        parameters = self._parameters
        
        if not isinstance(self.binding_model, NoBinding):
            parameters['binding_model'] = self.binding_model.parameters
        if not isinstance(self.bulk_reaction_model, NoReaction):
            parameters['bulk_reaction_model'] = self.bulk_reaction_model.parameters
        if not isinstance(self.particle_reaction_model, NoReaction):
            parameters['particle_reaction_model'] = \
                self.particle_reaction_model.parameters
        if not isinstance(self.discretization, NoDiscretization):
            parameters['discretization'] = \
                self.discretization.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.binding_model.parameters = parameters.pop('binding_model')
        except KeyError:
            pass
        try:
            self.bulk_reaction_model.parameters = parameters.pop('bulk_reaction_model')
        except KeyError:
            pass        
        try:
            self.particle_reaction_model.parameters = parameters.pop('particle_reaction_model')
        except KeyError:
            pass     
        try:
            self.discretization.parameters = parameters.pop('discretization')
        except KeyError:
            pass     
        
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, param, value)

    @property
    def section_dependent_parameters(self):
        parameters = {
            key: value for key, value in self.parameters.items() 
            if key in self._section_dependent_parameters
        }
        
        return parameters

    @property
    def polynomial_parameters(self):
        parameters = {
            key: value for key, value in self.parameters.items() 
            if key in self._polynomial_parameters
        }
        return parameters
        
    @property
    def initial_state(self):
        """dict: Dictionary with initial states.
        """
        initial_state = {st: getattr(self, st) for st in self._initial_state}

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        for st, value in initial_state.items():
            if st not in self._initial_state:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, st, value)

    @property
    def binding_model(self):
        """binding_model: BindingModel of the unit_operation.

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.
        """
        return self._binding_model

    @binding_model.setter
    def binding_model(self, binding_model):
        if not isinstance(binding_model, BindingBaseClass):
            raise TypeError('Expected BindingBaseClass')

        if binding_model.component_system is not self.component_system:
            raise CADETProcessError('Component systems do not match.')
            
        self._binding_model = binding_model
        
    @property
    def _n_bound_states(self):
        return self.binding_model.n_states
        
    @property
    def bulk_reaction_model(self):
        """bulk_reaction_model: Reaction model in the bulk phase

        Raises
        ------
        TypeError
            If bulk_reaction_model object is not an instance of ReactionBaseClass.
        CADETProcessError
            If unit does not support bulk reaction model.
            If number of components do not match.
        """
        return self._bulk_reaction_model

    @bulk_reaction_model.setter
    def bulk_reaction_model(self, bulk_reaction_model):
        if not isinstance(bulk_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')
        if not self.supports_bulk_reaction:
            raise CADETProcessError('Unit does not support bulk reactions.')

        if bulk_reaction_model.component_system is not self.component_system \
                and not isinstance(bulk_reaction_model, NoReaction):
            raise CADETProcessError('Component systems do not match.')

        self._bulk_reaction_model = bulk_reaction_model
    
    @property
    def particle_reaction_model(self):
        """particle_liquid_reaction_model: Reaction model in the particle liquid phase

        Raises
        ------
        TypeError
            If particle_reaction_model object is not an instance of ReactionBaseClass.
        CADETProcessError
            If unit does not support particle reaction model.
            If number of components do not match.
        """
        return self._particle_reaction_model

    @particle_reaction_model.setter
    def particle_reaction_model(self, particle_reaction_model):
        if not isinstance(particle_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')
        if not self.supports_bulk_reaction:
            raise CADETProcessError('Unit does not support particle reactions.')            

        if particle_reaction_model.component_system is not self.component_system \
                and not isinstance(particle_reaction_model, NoReaction):
            raise CADETProcessError('Component systems do not match.')

        self._particle_reaction_model = particle_reaction_model
        
    @property
    def discretization(self):
        """discretization: Discretization Parameters
        """
        return self._discretization

    def __repr__(self):
        """String-depiction of the object, can be changed into an object by
        calling the method eval.

        Returns
        -------
        class.name(parameters with values) : str
            Information about the class's name of an object and its parameters
            like number of components and object name, depicted as a string.
        """
        return '{}(n_comp={}, name=\'{}\')'.format(self.__class__.__name__,
            self.n_comp, self.name)

    def __str__(self):
        """Returns the information von __repr__ as a string object.

        Returns
        -------
        name : String
            Information about the class's name of an object and its paremeters
            like number of components and object name
        """
        return self.name
    

class SourceMixin(metaclass=StructMeta):
    """Mixin class for Units that have Source-like behavior
    
    See also
    --------
    SinkMixin
    Source
    Cstr
    """
    _n_poly_coeffs = 4
    flow_rate = Polynomial(dep=('_n_poly_coeffs'), default=0)
    _parameter_names = ['flow_rate']
    _section_dependent_parameters = ['flow_rate']
    _polynomial_parameters = ['flow_rate']


class SinkMixin():
    """Mixin class for Units that have Sink-like behavior

    See also
    --------
    SourceMixin
    Cstr
    """
    pass


class Source(UnitBaseClass, SourceMixin):
    """Pseudo unit operation model for streams entering the system.
    """
    c = NdPolynomial(dep=('n_comp', '_n_poly_coeffs'), default=0)
    _n_poly_coeffs = 4
    _parameter_names = \
        UnitBaseClass._parameter_names + \
        SourceMixin._parameter_names + \
        ['c']
    _section_dependent_parameters = \
        UnitBaseClass._section_dependent_parameters + \
        SourceMixin._section_dependent_parameters + \
        ['c']
    _polynomial_parameters = \
        UnitBaseClass._polynomial_parameters + \
        SourceMixin._polynomial_parameters + \
        ['c']

        
class Sink(UnitBaseClass, SinkMixin):
    """Pseudo unit operation model for streams leaving the system.
    """
    pass


class MixerSplitter(UnitBaseClass):
    """Pseudo unit operation model for mixing/splitting streams in the system.
    """
    pass


class TubularReactor(UnitBaseClass):
    """Class for tubular reactors.
    
    Class can be used for a regular tubular reactor. Also serves as parent for
    other tubular models like the GRM by providing methods for calculating
    geometric properties such as the cross section area and volume, as well as
    methods for convective and dispersive properties like mean residence time
    or NTP.
    
    Notes
    -----
    For subclassing, check that the total porosity and interstitial cross 
    section area are computed correctly depending on the model porosities!

    Attributes
    ----------
    length : UnsignedFloat
        Length of column.
    diameter : UnsignedFloat
        Diameter of column.
    axial_dispersion : UnsignedFloat
        Dispersion rate of compnents in axial direction.
    c : List of unsinged floats. Length depends on n_comp
        Initial concentration of the reactor.
             
    """
    supports_bulk_reaction = True
    length = UnsignedFloat(default=0)
    diameter = UnsignedFloat(default=0)
    axial_dispersion = UnsignedFloat()
    total_porosity = 1
    flow_direction = Switch(valid=[-1,1], default=1)
    _parameter_names = UnitBaseClass._parameter_names + [
        'length', 'diameter','axial_dispersion', 'flow_direction'
    ]
    _section_dependent_parameters = UnitBaseClass._section_dependent_parameters + [
        'axial_dispersion', 'flow_direction'
    ]
    
    c = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = UnitBaseClass._initial_state + ['c']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discretization = LRMDiscretizationFV()
        
    @property
    def cross_section_area(self):
        """float: Cross section area of a Column.
        
        See also
        --------
        volume
        cross_section_area_interstitial
        cross_section_area_liquid
        cross_section_area_solid
        """
        return math.pi/4 * self.diameter**2
    
    @cross_section_area.setter
    def cross_section_area(self, cross_section_area):
        self.diameter = (4*cross_section_area/math.pi)**0.5

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.
        
        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from 
        the flow rate, velocity, and porosity.
        

        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
        Needs to be overwritten depending on the model porosities!

        """
        self.cross_section_area = Q/(u0*self.total_porosity)
        
    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.
        
        Notes
        -----
        Needs to be overwritten depending on the model porosities!
        
        See also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid
        """        
        return self.total_porosity * self.cross_section_area
    
    @property
    def cross_section_area_liquid(self):
        """float: Liquid fraction of column cross section area.
        
        See also
        --------
        cross_section_area
        cross_section_area_interstitial
        cross_section_area_solid
        volume
        """        
        return self.total_porosity * self.cross_section_area

    @property
    def cross_section_area_solid(self):
        """float: Liquid fraction of column cross section area.
        
        
        See also
        --------
        cross_section_area
        cross_section_area_interstitial
        cross_section_area_liquid
        """        
        return (1 - self.total_porosity) * self.cross_section_area    

    @property
    def volume(self):
        """float: Volume of the TubularReactor.

        See also
        --------
        cross_section_area
        """
        return self.cross_section_area * self.length
    
    @property
    def volume_interstitial(self):
        """float: Interstitial volume between particles.

        See also
        --------
        cross_section_area
        """        
        return self.cross_section_area_interstitial * self.length

    @property
    def volume_liquid(self):
        """float: Volume of the liquid phase.
        """
        return self.cross_section_area_liquid * self.length
    
    @property
    def volume_solid(self):
        """float: Volume of the solid phase.
        """
        return self.cross_section_area_solid * self.length

    def t0(self, flow_rate):
        """Mean residence time of a (non adsorbing) volume element.
        
        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        t0 : float
            Mean residence time    

        See also
        --------
        u0
        """
        return self.volume_interstitial / flow_rate

    def u0(self, flow_rate):
        """Flow velocity of a (non adsorbint) volume element.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        u0 : float
            interstitial flow velocity
            
        See also
        --------
        t0
        NTP
        """
        return self.length/self.t0(flow_rate)
    
    def NTP(self, flow_rate):
        """Number of theoretical plates.
        
        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Calculated using the axial dispersion coefficient:
        :math: NTP = \frac{u \cdot L_{Column}}{2 \cdot D_a}

        Returns
        -------
        NTP : float
            Number of theretical plates
        """
        return self.u0(flow_rate) * self.length / (2 * self.axial_dispersion)
    
    def set_axial_dispersion_from_NTP(self, NTP, flow_rate):
        """
        Parameters
        ----------
        NTP : float
            Number of theroetical plates
        flow_rate : float
            volumetric flow rate

        Calculated using the axial dispersion coefficient:
        :math: NTP = \frac{u \cdot L_{Column}}{2 \cdot D_a}

        Returns
        -------
        NTP : float
            Number of theretical plates
            
        See also
        --------
        u0
        NTP
        """
        self.axial_dispersion = self.u0(flow_rate) * self.length / (2 * NTP)


class LumpedRateModelWithoutPores(TubularReactor):
    """Parameters for a lumped rate model without pores.

    Attributes
    ----------
    total_porosity : UnsignedFloat between 0 and 1.
        Total porosity of the column.
    q : List of unsinged floats. Length depends on n_comp
        Initial concentration of the bound phase.
    
    Notes
    -----
    Although technically the LumpedRateModelWithoutPores does not have 
    particles, the particle reactions interface is used to support reactions 
    in the solid phase and cross-phase reactions.
    """
    supports_bulk_reaction = False
    supports_particle_reaction = True
    
    total_porosity = UnsignedFloat(ub=1)
    _parameters = TubularReactor._parameter_names + ['total_porosity']

    q = DependentlySizedUnsignedList(dep=('n_comp','_n_bound_states'), default=0)
    _initial_state = TubularReactor._initial_state + ['q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discretization = LRMDiscretizationFV()
        

class LumpedRateModelWithPores(TubularReactor):
    """Parameters for the lumped rate model with pores.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    film_diffusion : List of unsinged floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
    pore_accessibility : List of unsinged floats. Length depends on n_comp.
        Accessibility of pores for components.
    cp : List of unsinged floats. Length depends on n_comp
        Initial concentration of the pores
    q : List of unsinged floats. Length depends on n_comp
        Initial concntration of the bound phase.
    """
    supports_bulk_reaction = True
    supports_particle_reaction = True
    
    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameter_names = TubularReactor._parameter_names + [
            'bed_porosity', 'particle_porosity', 'particle_radius', 
            'film_diffusion'
            ]    
    _section_dependent_parameters = \
        TubularReactor._section_dependent_parameters + ['film_diffusion']

    _cp = DependentlySizedUnsignedList(dep='n_comp')
    q = DependentlySizedUnsignedList(dep=('n_comp','_n_bound_states'), default=0)
    _initial_state = TubularReactor._initial_state + ['cp', 'q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discretization = LRMPDiscretizationFV()
        
    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.
        
        See also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid
        """        
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.
        
        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from 
        the flow rate, velocity, and bed porosity.
        

        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
        Overwrites parent method.

        """
        self.cross_section_area = Q/(u0*self.bed_porosity)

    @property
    def cp(self):
        if self._cp is None:
            return self.c

    @cp.setter
    def cp(self, cp):
        self._cp = cp


class GeneralRateModel(TubularReactor):
    """Parameters for the general rate model.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    pore_diffusion : List of unsinged floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
    surface_diffusion : List of unsinged floats. Length depends on n_comp.
        Diffusion rate for components in adsrobed state.
    pore_accessibility : List of unsinged floats. Length depends on n_comp.
        Accessibility of pores for components.
    cp : List of unsinged floats. Length depends on n_comp
        Initial concentration of the pores
    q : List of unsinged floats. Length depends on n_comp
        Initial concntration of the bound phase.
    """
    supports_bulk_reaction = True
    supports_particle_reaction = True
    
    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    surface_diffusion = DependentlySizedUnsignedList(dep=('n_comp','_n_bound_states'))
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameters = \
        TubularReactor._parameter_names + \
        ['bed_porosity', 'particle_porosity', 'particle_radius', 
        'film_diffusion', 'pore_diffusion', 'surface_diffusion']
    _section_dependent_parameters = \
        TubularReactor._section_dependent_parameters + \
        ['film_diffusion', 'pore_diffusion', 'surface_diffusion']
    
    _cp = DependentlySizedUnsignedList(dep='n_comp')
    q = DependentlySizedUnsignedList(dep=('n_comp', '_n_bound_states'), default=0)
    _initial_state = TubularReactor._initial_state + ['cp', 'q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discretization = GRMDiscretizationFV()
        
    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.
        
        See also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid
        """        
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.
        
        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from 
        the flow rate, velocity, and bed porosity.
        

        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
        Overwrites parent method.

        """
        self.cross_section_area = Q/(u0*self.bed_porosity)
        
    @property
    def cp(self):
        if self._cp is None:
            return self.c

    @cp.setter
    def cp(self, cp):
        self._cp = cp
        
class Cstr(UnitBaseClass, SourceMixin, SinkMixin):
    """Parameters for an ideal mixer.

    Parameters
    ----------
    c : List of unsinged floats. Length depends on n_comp
        Initial concentration of the reactor.
    q : List of unsinged floats. Length depends on n_comp
        Initial concentration of the bound phase.
    V : Unsinged float
        Initial volume of the reactor.
    total_porosity : UnsignedFloat between 0 and 1.
        Total porosity of the column.
    flow_rate_filter: np.Array 
        Flow rate of pure liquid without components to reduce volume.
    """
    supports_bulk_reaction = True
    supports_particle_reaction = True
    
    porosity = UnsignedFloat(ub=1, default=1)
    flow_rate_filter = Polynomial(dep=('_n_poly_coeffs'), default=0)
    _parameter_names = \
        UnitBaseClass._parameter_names + \
        SourceMixin._parameter_names + \
        ['porosity', 'flow_rate_filter']
    _section_dependent_parameters = \
        UnitBaseClass._section_dependent_parameters + \
        SourceMixin._section_dependent_parameters + \
        ['flow_rate_filter']
    _polynomial_parameters = \
        UnitBaseClass._polynomial_parameters + \
        SourceMixin._polynomial_parameters + \
        ['flow_rate_filter']        
            
    c = DependentlySizedUnsignedList(dep='n_comp', default=0)
    q = DependentlySizedUnsignedList(dep=('n_comp', '_n_bound_states'), default=0)
    V = UnsignedFloat(default=0)
    volume = V
    _initial_state = \
        UnitBaseClass._initial_state + \
        ['c', 'q', 'V']

    @property
    def volume_liquid(self):
        """float: Volume of the liquid phase.
        """
        return self.porosity * self.V    
    
    @property
    def volume_solid(self):
        """float: Volume of the solid phase.
        """
        return (1 - self.porosity) * self.V    

    def t0(self, flow_rate):
        """Mean residence time of a (non adsorbing) volume element.
        
        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        t0 : float
            Mean residence time    

        See also
        --------
        u0
        """
        return self.volume_liquid / flow_rate
