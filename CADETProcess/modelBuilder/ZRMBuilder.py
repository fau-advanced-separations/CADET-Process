from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(1) #number of components

from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import Inlet, LumpedRateModelWithoutPores, Outlet, Cstr
from CADETProcess.processModel import FlowSheet, Process
import math

#Assuming one layer of membrane; uniform binding model
binding_model = Langmuir(component_system)
binding_model.is_kinetic = True
binding_model.adsorption_rate = [9.5e-3]
binding_model.desorption_rate = [1]
binding_model.capacity = 1.2e3

#In both scenarios axial or radial configuration, we defined one inlet that goes into one cstr
inlet = Inlet(component_system, name='inlet')
volumetric_flow_rate = 1e-6
inlet.flow_rate = volumetric_flow_rate #flow rate is constant assuming constant liquid density 
inlet.c=1

outlet = Outlet(component_system, name='outlet') #Same as inlet


class AxialZRMBuilder:
    def __init__(self, component_system, inlet, outlet, flow_rate, segments_area, binding_model=None, name=None):
        """
        Builder class for Axial Zoneal Rate Model (ZRM) with symmetric CSTRs layout.

        Args:
            component_system: number of components.
            inlet: Feed inlet unit.
            outlet: Outlet unit.
            flow_rate: Volumetric flow rate through the system.
            segments_area: List of membrane segment areas.
            binding_model: Optional binding model to apply to all zones.
            name: Optional name for the builder instance.
        """
        self.component_system = component_system
        self.inlet = inlet
        self.outlet = outlet
        self.flow_rate = flow_rate
        self.segments_area = segments_area
        self.n = len(segments_area)
        self.binding_model = binding_model
        self.name = name or 'AxialZRMBuilder'

        self.zones = []
        self.cstrs = []
        self.flow_sheet = FlowSheet(component_system)
        
    
    def Flow_rates(self):
        """Returns the flow rate."""
        return self.flow_rate

    def build(self, length, porosity, axial_dispersion, volumes):
        """Constructs the flow sheet."""
        if len(volumes) != self.n:
            raise ValueError(f"Length of volumes list must match number of segments (n={self.n})")
        if any(a <= 0 for a in self.segments_area):
            raise ValueError("All segment areas must be positive.")
        self._initialize_units(length, porosity, axial_dispersion, volumes)
        self._connect_units()
        return self.flow_sheet

    def _initialize_units(self, length, porosity, axial_dispersion, volumes):
        """Initializes zones and CSTR units in the flow sheet."""
        self.flow_sheet.add_unit(self.inlet, feed_inlet=True)
        self.flow_sheet.add_unit(self.outlet)

        # Create and configure zones
        for i in range(self.n):
            zone = LumpedRateModelWithoutPores(self.component_system, f"zone{i+1}")
            if self.binding_model:
                zone.binding_model = self.binding_model
            zone.length = length
            zone.total_porosity = porosity
            zone.diameter = (4 * self.segments_area[i] / math.pi) ** 0.5 #from the segment area into diameter
            zone.axial_dispersion = axial_dispersion
            self.flow_sheet.add_unit(zone)
            self.zones.append(zone)

        # Create symmetric CSTRs' volumes
        symmetric_volumes = volumes + volumes[::-1]
        for i in range(2 * self.n):
            cstr = Cstr(self.component_system, f"cstr{i+1}")
            cstr.init_liquid_volume = symmetric_volumes[i]
            self.flow_sheet.add_unit(cstr)
            self.cstrs.append(cstr)

    def _connect_units(self):
        """Connects all units in the flow sheet."""
        # First half of CSTRs
        for i in range(self.n - 1):
            self.flow_sheet.add_connection(self.cstrs[i], self.cstrs[i + 1])

        # Connect first half CSTRs to zones
        for i in range(self.n):
            self.flow_sheet.add_connection(self.cstrs[i], self.zones[i])

        # Second half of CSTRs
        for i in range(self.n, 2 * self.n - 1):
            self.flow_sheet.add_connection(self.cstrs[i], self.cstrs[i + 1])

        # Back connections from zones to second half of CSTRs
        for i in range(self.n - 1, -1, -1):
            target_idx = 2 * self.n - i - 1
            self.flow_sheet.add_connection(self.zones[i], self.cstrs[target_idx])

        # Set output states for flow splitting
        total_area = sum(self.segments_area)
        epsilon = 1e-12  # Prevent divide-by-zero

        for i in range(self.n - 1):
            remaining_area = sum(self.segments_area[i:]) + epsilon
            zone_fraction = self.segments_area[i] / remaining_area
            next_cstr_fraction = 1 - zone_fraction
            self.flow_sheet.set_output_state(
                f'cstr{i+1}',
                {
                    f'cstr{i+2}': next_cstr_fraction,
                    f'zone{i+1}': zone_fraction
                }
            )
        # Inlet and outlet connections
        self.flow_sheet.add_connection(self.inlet, self.cstrs[0])
        self.flow_sheet.add_connection(self.cstrs[-1], self.outlet)

inlet.c=1
volumes=[1.6e-6,1e-6,1e-7,1e-7,1.6e-6,1e-6,1e-7,1e-7]
flow_rate=1e-6
segments_area=[7.85e-5,7.85e-5,3.85e-5,9e-6,7.85e-5,7.85e-5,3.85e-5,9e-5]
builder = AxialZRMBuilder(component_system, inlet, outlet, flow_rate=flow_rate, segments_area=segments_area, binding_model=binding_model )
flow_sheet = builder.build(length=0.0222, porosity=0.7,axial_dispersion=7.3e-9, volumes=volumes)