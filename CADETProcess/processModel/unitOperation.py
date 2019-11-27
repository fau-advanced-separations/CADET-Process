import math

from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta
from CADETProcess.common import Bool, String, List, DependentlySizedUnsignedList,\
    UnsignedInteger, UnsignedFloat
from CADETProcess.processModel import BindingBaseClass, NoBinding


class UnitBaseClass(metaclass=StructMeta):
    """Base class for all UnitOperation classes.

    A UnitOperation object stores model parameters and states (e.g. flow_rate)
    of a unit. Every unit operation can be assotiated with a binding behavior.
    If no binding model is set, a NoBinding instance is returned.
    UnitOperations can be connected in a FlowSheet. Each unit stores the ingoing
    (origin) and outgoing streams (destinations) of the flow sheet.

    Attributes
    ----------
    n_comp : UnsignedInteger
        Number of components in a system.
    parameters : list
        list of parameter names.
    name : String
        name of the unit_operation.
    state : list
        list with the state values. Contains the values for the flow_rate
        of an object. A default value is set to the parameter UnsignedFloat with
        zero value.
    binding_model : BindingBaseClass
        binding behavior of the unit. Defaults to NoBinding.
    origins : dict
        All units connecting to the unit
    destinations : dict
        All units to which the unit connects.

    See also
    --------
    FlowSheet
    CADETProcess.binding
    """
    name = String()
    n_comp = UnsignedInteger()

    flow_rate = UnsignedFloat(default=0.0)
    reverse_flow = Bool(default=False)
    _output_state = List()

    def __init__(self, n_comp, name):
        self.name = name
        self.n_comp = n_comp

        self._binding_model = NoBinding()

        self.origins = dict()
        self.destinations = dict()

        self._parameters = ['flow_rate', 'reverse_flow']
        self._initial_state = []


    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        parameters = {param: getattr(self, param) for param in self._parameters}
        if not isinstance(self.binding_model, NoBinding):
            parameters['binding_model'] = self.binding_model.parameters

        if not isinstance(self, Source):
            parameters['output_state'] = self.output_state

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.binding_model.parameters = parameters.pop('binding_model')
        except KeyError:
            pass

        try:
            self.output_state = parameters.pop('output_state')
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
    def output_state(self):
        """Returns the output_state, containing the updated states in a list.

        First it sets the state_length to the length of destinations. For a
        zero length the output_state list is empty. If the state is >= the
        state_length a CADETProcessError is raised. Else the entry of the
        list is set to the state_length and the state is set to 1, which
        means the flow_rate takes by 100% this way. Also the output_state
        is set to state and the flow_rate is updated.

        Parameters
        ----------
        state :
            dict with the flow_rates and the values of them.

        Raises
        ------
        CADETProcessError
            If state is integer and the state >= the state_length.
            If the length of the states is unequal the state_length
            If the sum of the states is unequal 1

        Returns
        -------
        output_state : NoneType, List
            Object from class List, contains the states for each unit an
            updates the flow_rates.

        """
        return self._output_state

    @output_state.setter
    def output_state(self, state):
        state_length = len(self.destinations)

        if state_length == 0:
            output_state = []

        if type(state) is int:
            if state >= state_length:
                raise CADETProcessError('Index exceeds destinations')

            output_state = [0] * state_length
            output_state[state] = 1

        else:
            if len(state) != state_length:
                raise CADETProcessError('Expected length {}.'.format(state_length))

            elif sum(state) != 1:
                raise CADETProcessError('Sum of fractions must be 1')

            output_state = state

        self._output_state = output_state

        self.update_flow_rate()

    def update_flow_rate(self):
        """Updates the flow_rate for each index in the list output_state.

        For each index and datadict of the destinations
        the flow_rate of the data_dict is updated. Therefore the flow_rate is
        multiplied with each index in the output_state list.
        """
        for index, datadict in enumerate(self.destinations.values()):
            datadict['flow_rate'] = self.output_state[index] * self.flow_rate

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

class SourceMixin():
    """Mixin class for Units that have Source-like behavior

    See also
    --------
    SinkMixin
    Cstr
    """
    pass

class SinkMixin():
    """Mixin class for Units that have Sink-like behavior

    See also
    --------
    SourceMixin
    Cstr
    """
    pass

class TubularReactor(UnitBaseClass):
    """Parameters for a tubular reactor.

    Additionally, methods for calculating the cross section area and volume are
    implemented, as well as the dead time of the column and the number of
    theroetical plates as a funtion of the flow rate.

    Attributes
    ----------
    length : UnsignedFloat
        Length of column.
    diameter : UnsignedFloat
        Diameter of column.
    axial_dispersion : UnsignedFloat
        Dispersion rate of compnents in axial direction.
    c : List of unsinged floats. Length depends on n_comp
        Initial concentration of the reactor
    """
    length = UnsignedFloat()
    diameter = UnsignedFloat()
    axial_dispersion = UnsignedFloat()

    c = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameters += ['length', 'diameter', 'axial_dispersion']

        self.c = [0] * self.n_comp

        self._initial_state += ['c']

    @property
    def volume(self):
        """float: Volume of the Tubular Reactor.

        See also
        --------
        cross_section_area
        """
        return self.cross_section_area * self.length

    @property
    def cross_section_area(self):
        """float: Cross section area of a Column.

        See also
        --------
        cross_section_area
        """
        return math.pi/4 * self.diameter**2

    @property
    def t0(self):
        """float: Mean residence time of a (non adsorbing) volume element.

        See also
        --------
        u0
        """
        return self.volume / self.flow_rate

    @property
    def u0(self):
        """float: Flow velocity of a (non adsorbint) volume element.

        See also
        --------
        t0
        """
        return self.length/self.t0


class Column(TubularReactor):
    """Parameters for a chromatographic column.

    Additionally, methods for calculating the cross section area and volume are
    implemented, as well as the dead time of the column and the number of
    theroetical plates as a funtion of the flow rate.

    Attributes
    ----------
    total_porosity : UnsignedFloat between 0 and 1.
        Total porosity of the column. Calculated from bed_porosity and
        if values are present.
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
    _total_porosity = UnsignedFloat(ub=1)
    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    surface_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')

    cp = DependentlySizedUnsignedList(dep='n_comp')
    q = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._parameters += [
            'total_porosity', 'bed_porosity', 'particle_porosity',
            'particle_radius', 'film_diffusion', 'pore_diffusion',
            'surface_diffusion'
            ]

        self.cp = [0] * self.n_comp
        self.q = [0] * self.n_comp

        self._initial_state += ['cp', 'q']

    @property
    def total_porosity(self):
        """Defines the total porosity of a Column object.

        Trys to get the total_porosity by adding the bed_porosity with 1- the
        bed_porosity and multiplying this amount with the particle_porosity.
        Else excepts a TypeError and ignores it. Also the total_porosity
        can be set to a new value.

        Parameters
        ----------
        value : float
            The value to change the total_porosity.

        Returns
        -------
        total_porosity : float
            The total porosity of the column object.
        """
        try:
            self._total_porosity = (
                    self.bed_porosity + (1 - self.bed_porosity) \
                    * self.particle_porosity)
        except TypeError:
            pass

        return self._total_porosity

    @total_porosity.setter
    def total_porosity(self, value):
        self._total_porosity = value


    @property
    def volume_solid(self):
        """Returns the volume of the solid phase.

        Calculates the volume_solid by multiplying the volume by 1- the
        total_porosity.

        Returns
        -------
        volume_solid : UnsignedFloat
            The volume of the solid phase of a Column object.

        See also
        --------
        volume
        """
        return ((1) - self.total_porosity) * \
            self.volume


    @property
    def t0(self):
        """Returns the time, a non adsorbing molecule needs to pass the column.

        Returns
        -------
        t0 : float
            Dead time of non adsorbing component
        """
        return self.total_porosity * self.volume / self.flow_rate


    @property
    def NTP(self):
        """Returns the number of theretical plates.

        Calculated using the axial dispersion coefficient:
        :math: NTP = \frac{u \cdot L_{Column}}{2 \cdot D_a}

        Returns
        -------
        NTP : float
            Number of theretical plates
        """
        return self.u0 * self.length / (2 * self.axial_dispersion)

    @NTP.setter
    def NTP(self, NTP):
        self.axial_dispersion = self.u0 * self.length / (2 * NTP)

class Cstr(UnitBaseClass, SourceMixin, SinkMixin):
    """Class CSTR.

    Describes a CSTR object. It inherits the Attributes of the UnitBaseClass,
    the SourceMixin and SinkMixin and saves the changig variables c, q, V as
    DependentlySizedUnsignedList in a list sates for each component in the
    system.

    c : List of unsinged floats. Length depends on n_comp
        Initial concentration of the reactor
    q : List of unsinged floats. Length depends on n_comp
        Initial concntration of the bound phase.
    V : Unsinged float
        Initival volume of the reactor.
    """
    c = DependentlySizedUnsignedList(dep='n_comp')
    q = DependentlySizedUnsignedList(dep='n_comp')
    V = UnsignedFloat()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c = [0] * self.n_comp
        self.q = [0] * self.n_comp

        self._initial_state += ['c', 'q', 'V' ]

class Source(UnitBaseClass, SourceMixin):
    """Class Source.

    Describes a Source object. It inherits the Attributes of the UnitBaseClass
    and the SourceMixin. Saves the variable c to the sates list.
    """
    c = DependentlySizedUnsignedList(dep='n_comp')
    lin_gradient = DependentlySizedUnsignedList(dep='n_comp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_state += ['c']


class Sink(UnitBaseClass, SinkMixin):
    """Class Sink.

    Describes a Sink object. It inherits the Attributes of the UnitBaseClass
    and the SinkMixin.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
