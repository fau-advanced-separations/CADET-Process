import unittest

from CADETProcess.processModel import (
    ComponentSystem,
    Cstr,
    FlowSheet,
    GeneralRateModel,
    Inlet,
    LumpedRateModelWithoutPores,
    LumpedRateModelWithPores,
    MassActionLaw,
    MassActionLawParticle,
    Outlet,
    Process,
    TubularReactor,
)
from CADETProcess.simulator import Cadet

from tests.test_cadet_adapter import found_cadet


def setup_process(unit_type):
    component_system = ComponentSystem(2)

    inlet = Inlet(component_system, "inlet")
    inlet.c = [1, 0]
    inlet.flow_rate = 1e-3

    if unit_type == "cstr":
        cstr = Cstr(component_system, "reaction_unit")
        total_volume = 1e-3
        total_porosity = 0.7
        cstr.init_liquid_volume = total_porosity * total_volume
        cstr.const_solid_volume = (1 - total_porosity) * total_volume

        unit = cstr
    elif unit_type == "pfr":
        pfr = TubularReactor(component_system, "reaction_unit")
        pfr.length = 1
        pfr.diameter = 1e-2
        pfr.axial_dispersion = 1e-6

        unit = pfr
    elif unit_type == "lrm":
        lrm = LumpedRateModelWithoutPores(component_system, "reaction_unit")
        lrm.length = 1
        lrm.diameter = 1e-2
        lrm.total_porosity = 0.7
        lrm.axial_dispersion = 1e-6

        unit = lrm
    elif unit_type == "lrmp":
        lrmp = LumpedRateModelWithPores(component_system, "reaction_unit")
        lrmp.length = 1
        lrmp.diameter = 1e-2
        lrmp.bed_porosity = 0.7
        lrmp.particle_radius = 1e-6
        lrmp.particle_porosity = 0.7
        lrmp.axial_dispersion = 1e-6
        lrmp.film_diffusion = 2 * [1e-3]

        unit = lrmp
    elif unit_type == "grm":
        grm = GeneralRateModel(component_system, "reaction_unit")

        grm.length = 1
        grm.diameter = 1e-2
        grm.bed_porosity = 0.7
        grm.particle_radius = 1e-6
        grm.particle_porosity = 0.7
        grm.axial_dispersion = 1e-6
        grm.film_diffusion = 2 * [1e-3]
        grm.pore_diffusion = 2 * [1e-3]

        unit = grm
    else:
        raise ValueError("Unknown Model.")

    outlet = Outlet(component_system, "outlet")

    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(unit)
    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(inlet, unit)
    flow_sheet.add_connection(unit, outlet)

    process = Process(flow_sheet, "test_reaction_bulk")
    process.cycle_time = 10

    return process


class TestReaction(unittest.TestCase):
    """Unit tests for the reaction module."""

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_reaction_bulk(self):
        """Test the reaction in bulk."""
        for unit_type in ["cstr", "pfr", "lrmp", "grm"]:
            with self.subTest(unit_type=unit_type):
                process = setup_process(unit_type)

                bulk_reaction_model = MassActionLaw(process.component_system)
                bulk_reaction_model.add_reaction([0, 1], [-1, 1], 1)

                process.flow_sheet.reaction_unit.bulk_reaction_model = (
                    bulk_reaction_model
                )

                simulator = Cadet()
                results = simulator.simulate(process)

                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][0] != 1)
                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][1] != 0)

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_reaction_particle(self):
        """Test the reaction in particle.

        Notes
        -----
        Cstr currently not working. Need to investigate

        """
        for unit_type in ["lrm", "lrmp", "grm"]:
            with self.subTest(unit_type=unit_type):
                process = setup_process(unit_type)

                particle_reaction_model = MassActionLawParticle(
                    process.component_system
                )
                particle_reaction_model.add_liquid_reaction([0, 1], [-1, 1], 1)

                process.flow_sheet.reaction_unit.particle_reaction_model = (
                    particle_reaction_model
                )

                simulator = Cadet()
                results = simulator.simulate(process)

                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][0] != 1)
                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][1] != 0)

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_reaction_bulk_in_particle(self):
        """Test the reaction in particle when setting a bulk reaction type.

        Notes
        -----
        Cstr currently not working. Need to investigate

        """
        for unit_type in ["lrm", "lrmp", "grm"]:
            with self.subTest(unit_type=unit_type):
                process = setup_process(unit_type)

                bulk_reaction_model = MassActionLaw(process.component_system)
                bulk_reaction_model.add_reaction([0, 1], [-1, 1], 1)

                process.flow_sheet.reaction_unit.particle_reaction_model = (
                    bulk_reaction_model
                )

                simulator = Cadet()
                results = simulator.simulate(process)

                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][0] != 1)
                self.assertTrue(results.solution.outlet.inlet.solution[-1, :][1] != 0)


if __name__ == "__main__":
    unittest.main()
