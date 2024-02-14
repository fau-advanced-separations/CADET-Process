import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Linear, StericMassAction


def create_component_system(components):
    return ComponentSystem(components)


class TestLinearFixture(Linear):
    def __init__(
        self,
        n_comp,
        is_kinetic=True,
        adsorption_rate=None,
        desorption_rate=None,
        ):

        component_system = ComponentSystem(n_comp)

        if adsorption_rate is None:
            adsorption_rate = np.arange(component_system.n_comp).tolist()
        if desorption_rate is None:
            desorption_rate = np.ones((n_comp,)).tolist()

        super().__init__(
            component_system,
            is_kinetic=is_kinetic,
            adsorption_rate=adsorption_rate,
            desorption_rate=desorption_rate,
        )


class TestStericMassActionFixture(StericMassAction):
    def __init__(
        self,
        n_comp,
        is_kinetic=True,
        adsorption_rate=None,
        desorption_rate=None,
        characteristic_charge=None,
        steric_factor=None,
        capacity=1000,
        use_reference_concentrations=True,
        ):
        component_system = ComponentSystem(n_comp)

        if adsorption_rate is None:
            adsorption_rate = np.arange(component_system.n_comp)

        if desorption_rate is None:
            desorption_rate = np.ones((n_comp,))

        if characteristic_charge is None:
            characteristic_charge = 2 * np.arange((n_comp,))

        if steric_factor is None:
            steric_factor = 2 * np.arange((n_comp,))

        if use_reference_concentrations:
            reference_liquid_phase_conc = capacity
            reference_solid_phase_conc = capacity

        super().__init__(
            component_system,
            is_kinetic=is_kinetic,
            adsorption_rate=adsorption_rate,
            desorption_rate=desorption_rate,
            characteristic_charge=characteristic_charge,
            steric_factor=steric_factor,
            capacity=capacity,
            reference_liquid_phase_conc=reference_liquid_phase_conc,
            reference_solid_phase_conc=reference_solid_phase_conc,
        )


# def create_isotherm(isotherm, n_comp, isotherm_parameters):
#     if isotherm == 'linear':
#         return create_linear_isotherm(n_comp, **isotherm_parameters)
#     else:
#         raise ValueError(f"Unknown isotherm: {isotherm}")
