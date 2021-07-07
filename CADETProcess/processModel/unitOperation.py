import math

from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta
from CADETProcess.common import Bool, String, List, \
    DependentlySizedUnsignedList, UnsignedInteger, UnsignedFloat
from CADETProcess.processModel import BindingBaseClass, NoBinding
from CADETProcess.processModel import ReactionBaseClass, NoReaction


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
    n_comp = UnsignedInteger()

    _parameters = []
    _initial_state = []

    def __init__(self, n_comp, name):
        self.name = name
        self.n_comp = n_comp

        self._binding_model = NoBinding()

        self._bulk_reaction_model = NoReaction()
        self._particle_liquid_reaction_model = NoReaction()
        self._particle_solid_reaction_model = NoReaction()
        
    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        parameters = {param: getattr(self, param) 
                      for param in self._parameters}
        
        if not isinstance(self.binding_model, NoBinding):
            parameters['binding_model'] = self.binding_model.parameters
        if not isinstance(self.bulk_reaction_model, NoReaction):
            parameters['bulk_reaction_model'] = self.bulk_reaction_model.parameters
        if not isinstance(self.particle_liquid_reaction_model, NoReaction):
            parameters['particle_liquid_reaction_model'] = \
                self.particle_liquid_reaction_model.parameters
        if not isinstance(self.particle_solid_reaction_model, NoReaction):
            parameters['particle_solid_reaction_model'] = \
                self.particle_solid_reaction_model.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.binding_model.parameters = parameters.pop('binding_model')
        except KeyError:
            pass

        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, param, value)

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

        if binding_model.n_comp != self.n_comp and not isinstance(
                binding_model, NoBinding):
            raise CADETProcessError('Number of components does not match.')

        self._binding_model = binding_model
        
    @property
    def bulk_reaction_model(self):
        """bulk_reaction_model: Reaction model in the bulk phase

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.
        """
        return self._bulk_reaction_model

    @bulk_reaction_model.setter
    def bulk_reaction_model(self, bulk_reaction_model):
        if not isinstance(bulk_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')

        if bulk_reaction_model.n_comp != self.n_comp and not isinstance(
                bulk_reaction_model, NoReaction):
            raise CADETProcessError('Number of components does not match.')

        self._bulk_reaction_model = bulk_reaction_model
    
    @property
    def particle_liquid_reaction_model(self):
        """particle_liquid_reaction_model: Reaction model in the particle liquid phase

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.
        """
        return self._particle_liquid_reaction_model

    @particle_liquid_reaction_model.setter
    def particle_liquid_reaction_model(self, particle_liquid_reaction_model):
        if not isinstance(particle_liquid_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')

        if particle_liquid_reaction_model.n_comp != self.n_comp and not isinstance(
                particle_liquid_reaction_model, ReactionBaseClass):
            raise CADETProcessError('Number of components does not match.')

        self._particle_liquid_reaction_model = particle_liquid_reaction_model

    @property
    def particle_solid_reaction_model(self):
        """particle_solid_reaction_model: Reaction model in the particle solid phase

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.
        """
        return self._particle_solid_reaction_model

    @particle_solid_reaction_model.setter
    def particle_solid_reaction_model(self, particle_solid_reaction_model):
        if not isinstance(particle_solid_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')

        if particle_solid_reaction_model.n_comp != self.n_comp and not isinstance(
                particle_solid_reaction_model, ReactionBaseClass):
            raise CADETProcessError('Number of components does not match.')

        self._particle_solid_reaction_model = particle_solid_reaction_model

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
    

class SinkMixin():
    """Mixin class for Units that have Sink-like behavior

    See also
    --------
    SourceMixin
    Cstr
    """
    pass


class TubularReactor(UnitBaseClass):
    """Class for tubular reactors.
    
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
    length = UnsignedFloat()
    diameter = UnsignedFloat()
    axial_dispersion = UnsignedFloat()
    total_porosity = 1
    reverse_flow = Bool(default=False)
    _parameters = UnitBaseClass._parameters + [
        'length', 'diameter','axial_dispersion']
    
    c = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = UnitBaseClass._initial_state + ['c']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def cross_section_area(self):
        """float: Cross section area of a Column.

        See also
        --------
        cross_section_area
        """
        return math.pi/4 * self.diameter**2
    
    @cross_section_area.setter
    def cross_section_area(self, cross_section_area):
        self.diameter = (4*cross_section_area/math.pi)**0.5

    @property
    def cylinder_volume(self):
        """float: Volume of the TubularReactor.

        See also
        --------
        cross_section_area
        """
        return self.cross_section_area * self.length
    
    @property
    def volume_liquid(self):
        return self.total_porosity * self.cylinder_volume

    @property
    def volume_solid(self):
        """float: Volume of the solid phase.
        """
        return (1 - self.total_porosity) * self.cylinder_volume

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
        return self.u0 * self.length / (2 * self.axial_dispersion)
    
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
    """
    total_porosity = UnsignedFloat(ub=1)
    _parameters = TubularReactor._parameters + ['total_porosity']

    q = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = TubularReactor._initial_state + ['q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
    pore_diffusion : List of unsinged floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
    pore_accessibility : List of unsinged floats. Length depends on n_comp.
        Accessibility of pores for components.
    cp : List of unsinged floats. Length depends on n_comp
        Initial concentration of the pores
    q : List of unsinged floats. Length depends on n_comp
        Initial concntration of the bound phase.
    """
    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameters = TubularReactor._parameters + [
            'bed_porosity', 'particle_porosity', 'particle_radius', 
            'film_diffusion', 'pore_diffusion',
            ]

    cp = DependentlySizedUnsignedList(dep='n_comp', default=0)
    q = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = TubularReactor._initial_state + ['cp', 'q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity


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
    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    surface_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameters = TubularReactor._parameters + [
            'bed_porosity', 'particle_porosity', 'particle_radius', 
            'film_diffusion', 'pore_diffusion', 'surface_diffusion'
            ]

    cp = DependentlySizedUnsignedList(dep='n_comp', default=0)
    q = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = TubularReactor._initial_state + ['cp', 'q']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity

    
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
    """
    flow_rate = UnsignedFloat(default=0.0)
    _parameters = UnitBaseClass._parameters + ['flow_rate']

    c = DependentlySizedUnsignedList(dep='n_comp', default=0)
    q = DependentlySizedUnsignedList(dep='n_comp', default=0)
    V = UnsignedFloat(default=0)
    _initial_state = UnitBaseClass._initial_state + ['c', 'q', 'V']
        

class Source(UnitBaseClass, SourceMixin):
    """Pseudo unit operation model for streams entering the system.
    """
    flow_rate = UnsignedFloat(default=0.0)
    lin_gradient = Bool(default=False)
    _parameters = UnitBaseClass._parameters + ['flow_rate', 'lin_gradient']
    
    c = DependentlySizedUnsignedList(dep='n_comp', default=0)
    _initial_state = UnitBaseClass._initial_state + ['c']


class Sink(UnitBaseClass, SinkMixin):
    """Pseudo unit operation model for streams leaving the system.
    """
    pass


class MixerSplitter(UnitBaseClass):
    """Pseudo unit operation model for mixing/splitting streams in the system.
    """
    pass