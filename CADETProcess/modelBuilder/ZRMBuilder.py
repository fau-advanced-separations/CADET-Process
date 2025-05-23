from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(1) #number of components

from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import Inlet, LumpedRateModelWithoutPores, Outlet, Cstr
from CADETProcess.processModel import FlowSheet, Process
import math

class ZRMBuilder:
    def __init__(self, component_system, flow_rate, segments_area, configuration, init_c=0, binding_model=None, name=None):
        """
        Builder class for Axial Zoneal Rate Model (ZRM) with symmetric cstrs layout.

        Args:
            component_system: number of components, include the type.
            inlet: Feed inlet unit.
            outlet: Outlet unit.
            flow_rate: Volumetric flow rate through the system.
            segments_area: List of membrane segment areas.
            binding_model: Optional binding model to apply to all zones.
            name: Optional name for the builder instance.
        """
        self.component_system = component_system
        self.flow_rate = flow_rate
        self.segments_area = segments_area
        self.init_c=init_c

        if configuration not in ['Axial', 'Radial']:
            raise ValueError("Configuration must be either 'Axial' or 'Radial'")

        # Validate that each segment area is positive
        if any(a <= 0 for a in segments_area):
            raise ValueError("All segment areas must be positive.")
        
        self.configuration=configuration 
        self.n = len(segments_area)
        self.binding_model = binding_model
        self.name = name or 'ZRMBuilder'

        self.zones = []
        self.cstrs_in = []
        self.cstrs_out = []

        self._flow_sheet = FlowSheet(component_system, name)
        self._process = Process(self._flow_sheet, name)

        #Define the inlet/outlet
        self.inlet = Inlet(component_system, name='inlet')
        self.inlet.c = self.init_c
        self.inlet.flow_rate = self.flow_rate

        self.outlet = Outlet(component_system, name='outlet')


    @property
    def flow_sheet(self):
        return self._flow_sheet

    @property
    def process(self):
        return self._process

    @property
    def cycle_time(self):
        return self.process.cycle_time  
    
    
    def Segments_ratio(self):
        """Returns the surface area ratio of each zone."""
        total_area = sum(self.segments_area)
        if total_area == 0:
            raise ZeroDivisionError("Total area of segments is zero.")
        return [area / total_area for area in self.segments_area]

    def build(self , rate_model , length , porosity, axial_dispersion, volumes_in, volumes_out):
        
         """Constructs the flow sheet."""
         if len(volumes_in) != self.n:
             raise ValueError(f"Length of volumes_in must be {self.n}, got {len(volumes_in)}")

         if any(area <= 0 for area in self.segments_area):
             raise ValueError("All segment areas must be positive.")

         if any(v <= 0 for v in volumes_in):
             raise ValueError("All CSTR volumes (volumes_in) must be positive.")

         if not volumes_out:
             volumes_out = volumes_in
         elif len(volumes_out) != self.n:
             raise ValueError(f"Length of volumes_out must be {self.n}, got {len(volumes_out)}")

    # Proceed with the rest of the build logic here

         self._initialize_units(rate_model,length, porosity, axial_dispersion,volumes_in,volumes_out)
         self._connect_units()
         return self.flow_sheet
    

    def _initialize_units(self,rate_model, length, porosity, axial_dispersion, volumes_in, volumes_out):
        """Initializes zones and cstr units in the flow sheet."""
        self.flow_sheet.add_unit(self.inlet, feed_inlet=True)
        self.flow_sheet.add_unit(self.outlet)


        # Create membrane zones
        for i in range(self.n):
            zone = rate_model(self.component_system, f"zone{i+1}")
            if self.binding_model:
                zone.binding_model = self.binding_model
            zone.length = length
            zone.total_porosity = porosity
            zone.diameter = (4 * self.segments_area[i] / math.pi) ** 0.5 #from the segment area into diameter
            zone.axial_dispersion = axial_dispersion
            self.flow_sheet.add_unit(zone)
            self.zones.append(zone)

       

        for i in range(self.n):
            cstr_in = Cstr(self.component_system, f"cstr_in{i+1}")
            cstr_out = Cstr(self.component_system, f"cstr_out{i+1}")
            cstr_in.init_liquid_volume = volumes_in[i]
            cstr_out.init_liquid_volume =volumes_out[i]
            self.flow_sheet.add_unit(cstr_in)
            self.flow_sheet.add_unit(cstr_out)
            self.cstrs_in.append(cstr_in)
            self.cstrs_out.append(cstr_out)

    def _connect_units(self):
        """Connects all units in the flow sheet."""
        # First half of cstrs
        for i in range(self.n - 1):
            self.flow_sheet.add_connection(self.cstrs_in[i], self.cstrs_in[i + 1])

        # Connect first half cstrs to zones
        for i in range(self.n):
            self.flow_sheet.add_connection(self.cstrs_in[i], self.zones[i])
            self.flow_sheet.add_connection(self.zones[i], self.cstrs_out[i])

       # Conditional handling for configuration
        if self.configuration == 'Axial':
            # Forward connection for second cstr chain
            for i in range(self.n - 1):
                self.flow_sheet.add_connection(self.cstrs_out[i+1], self.cstrs_out[i])

        elif self.configuration == 'Radial':
            # Backward connection for second cstr chain
            for i in range(self.n - 1):
                self.flow_sheet.add_connection(self.cstrs_out[i], self.cstrs_out[i+1])
        else:
            raise ValueError("Invalid configuration. Must be 'Axial' or 'Radial'.")

        # Set output states for flow splitting
        total_area = sum(self.segments_area)
        lowest_bound = 1e-12  # Prevent divide-by-zero

        Ratio=self.Segments_ratio()
        for i in range(self.n - 1):
            remaining_area = sum(Ratio[i:]) + lowest_bound
            zone_fraction = Ratio[i] / remaining_area
            next_cstr_fraction = 1 - zone_fraction
            self.flow_sheet.set_output_state(
                f'cstr_in{i+1}',
                {
                    f'cstr_in{i+2}': next_cstr_fraction,
                    f'zone{i+1}': zone_fraction
                }
            )
        # Inlet and outlet connections
        self.flow_sheet.add_connection(self.inlet, self.cstrs_in[0])
        if self.configuration == 'Axial':
            self.flow_sheet.add_connection(self.cstrs_out[0], self.outlet)
        else:  # Radial already validated above
            self.flow_sheet.add_connection(self.cstrs_out[-1], self.outlet)




