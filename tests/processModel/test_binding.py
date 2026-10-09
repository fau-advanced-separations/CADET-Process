import unittest

import numpy
from CADETProcess import CADETProcessError
from CADETProcess.processModel import (
    BiLangmuir,
    BindingBaseClass,
    ColloidalParticleAdsorption,
    ComponentSystem,
    Langmuir,
    MobilePhaseModulator,
    MultistateStericMassAction,
    StericMassAction,
    binding,
)
from CADETProcess.simulator.cadetAdapter import (
    AdsorptionParameters,
    adsorption_parameters_map,
)


class Test_Binding(unittest.TestCase):
    def setUp(self):
        component_system = ComponentSystem(2)

        binding_model = Langmuir(component_system, name="test")

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.capacity = [100, 100]

        self.langmuir = binding_model

        binding_model = BiLangmuir(component_system, name="test")

        binding_model.adsorption_rate = [0.02, 0.03, 0.001, 0.002]
        binding_model.desorption_rate = [1, 1, 2, 2]
        binding_model.capacity = [100, 100, 200, 200]

        self.bi_langmuir = binding_model

        binding_model = MultistateStericMassAction(component_system, name="test")

        self.multi_state_sma = binding_model

        binding_model = StericMassAction(component_system, name="test")

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.characteristic_charge = [5, 10]
        binding_model.steric_factor = [10, 10]
        binding_model.capacity = 100
        binding_model.reference_liquid_phase_conc = 200
        binding_model.reference_solid_phase_conc = 100
        self.sma = binding_model

        binding_model = MobilePhaseModulator(component_system, name="test")

        binding_model.adsorption_rate = [0.02, 0.03]
        binding_model.desorption_rate = [1, 1]
        binding_model.capacity = [100, 100]
        binding_model.ion_exchange_characteristic = [1.4, 1.7]
        binding_model.hydrophobicity = [1.4, 1.7]
        binding_model.linear_threshold = 1e-8
        self.mobile_phase_modulator = binding_model

    def test_sma_reference_concentration_transformations(self):
        adsorption_rate = numpy.array(self.sma.adsorption_rate)
        desorption_rate = numpy.array(self.sma.desorption_rate)
        characteristic_charge = numpy.array(self.sma.characteristic_charge)
        expected_untransformed_adsorption_rate = adsorption_rate * (
            self.sma.reference_solid_phase_conc**-characteristic_charge
        )
        expected_untransformed_desorption_rate = desorption_rate * (
            self.sma.reference_liquid_phase_conc**-characteristic_charge
        )
        assert numpy.allclose(
            expected_untransformed_adsorption_rate,
            self.sma.adsorption_rate_untransformed,
        )
        assert numpy.allclose(
            expected_untransformed_desorption_rate,
            self.sma.desorption_rate_untransformed,
        )

        with self.assertRaises(ValueError):
            component_system = ComponentSystem(2)
            binding_model = StericMassAction(component_system, name="test")
            binding_model.adsorption_rate_untransformed = [0.02, 0.03]

    def test_in_cadetAdapter(self):
        binding_classes = {
            name: cls
            for name, cls in binding.__dict__.items()
            if (isinstance(cls, type) and issubclass(cls, BindingBaseClass))
        }
        binding_classes.pop("BindingBaseClass")

        for name, cls in binding_classes.items():
            self.assertIn(name, adsorption_parameters_map)
            map = adsorption_parameters_map[name]
            try:
                assert set(cls._parameters) == set(
                    map["parameters"].values()
                )  # This needs to compare to .values
            except AssertionError:
                raise AssertionError(
                    f"Isotherm '{name}' has these parameters in cadetAdapter.py missing: "
                    f"{set(cls._parameters) - set(map['parameters'].values())} "
                    "and these parameters in binding.py missing: "
                    f"{set(map['parameters'].values()) - set(cls._parameters)}"
                )

    def test_colloidal_particle_adsorption_config(self):
        model = ColloidalParticleAdsorption(
            ComponentSystem(3), proton_index=1, effective_charge_polynomial_degree=1
        )
        model.temperature = 298.15
        model.ionic_strength = 100
        model.permittivity = 78.5
        model.ligand_density = 1e-6
        model.ligand_charge_full = 1
        model.ligand_pk = 7
        model.specific_surface_area = [1e7, 0, 1e7]
        model.protein_radius = [3e-9, 0, 4e-9]
        model.effective_charge_coefficient_0 = [5, 0, 6]
        model.effective_charge_coefficient_1 = [-1, 0, -2]
        model.lateral_charge = [5, 0, 6]
        model.reference_ph = 7
        model.delta_ref = [0.1, 0, 0.2]
        model.delta_linear = [1, 0, 2]
        model.kinetic_prefactor = [1, 0, 2]

        config = AdsorptionParameters(model).parameters

        self.assertEqual(model.bound_states, [1, 0, 1])
        self.assertEqual(config["ADSORPTION_MODEL"], "COLLOIDAL_PARTICLE_ADSORPTION")
        self.assertEqual(config["CPA_PROTON_IDX"], 1)
        self.assertEqual(config["CPA_EFFECTIVE_CHARGE_COEF"], [5, 0, 6, -1, 0, -2])
        self.assertNotIn("CPA_IONIC_VALENCE", config)
        self.assertEqual(model.get_effective_charge_coefficient(1), [-1, 0, -2])
        self.assertEqual(model.effective_charge_coefficient_0, [5, 0, 6])
        self.assertFalse(hasattr(model, "effective_charge_coefficient_2"))

        model.effective_charge_polynomial_degree = 4
        model.effective_charge_coefficient_4 = [0.5, 0, -0.5]
        self.assertEqual(model.get_effective_charge_coefficient(2), [0, 0, 0])
        self.assertEqual(
            AdsorptionParameters(model).parameters["CPA_EFFECTIVE_CHARGE_COEF"],
            [5, 0, 6, -1, 0, -2, 0, 0, 0, 0, 0, 0, 0.5, 0, -0.5],
        )

        with self.assertRaises(ValueError):
            model.set_effective_charge_coefficient(5, [1, 2, 3])

        with self.assertRaises(ValueError):
            model.set_effective_charge_coefficient(2, [1, 2])

        model.ionic_valence = [2, 1, -1]
        self.assertEqual(AdsorptionParameters(model).parameters["CPA_IONIC_VALENCE"], [2, 1, -1])

        with self.assertRaises(ValueError):
            model.ionic_valence = [2, 1]

        with self.assertRaises(ValueError):
            model.temperature = 0

        with self.assertRaises(CADETProcessError):
            model.bound_states = [1, 1, 1]

    def test_get_parameters(self):
        parameters_expected = {
            "is_kinetic": True,
            "adsorption_rate": [0.02, 0.03],
            "desorption_rate": [1.0, 1.0],
            "capacity": [100.0, 100.0],
        }
        parameters = self.langmuir.parameters
        self.assertDictEqual(parameters_expected, parameters)

    def test_binding_sites(self):
        self.assertEqual(self.bi_langmuir.n_binding_sites, 2)
        self.assertEqual(self.bi_langmuir.n_bound_states, 4)

        self.bi_langmuir.n_binding_sites = 3
        self.assertEqual(self.bi_langmuir.n_binding_sites, 3)
        self.assertEqual(self.bi_langmuir.n_bound_states, 6)

        with self.assertRaises(ValueError):
            self.langmuir.n_binding_sites = 2

    def test_bound_states(self):
        self.assertEqual(self.multi_state_sma.n_bound_states, 2)

        self.multi_state_sma.bound_states = [3, 2]
        self.assertEqual(self.multi_state_sma.n_bound_states, 5)

        with self.assertRaises(ValueError):
            self.multi_state_sma.bound_states = [1, 2, 3]

        with self.assertRaises(ValueError):
            self.langmuir.bound_states = [2, 2]


if __name__ == "__main__":
    unittest.main()
