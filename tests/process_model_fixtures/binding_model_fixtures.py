import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Linear, StericMassAction


def fixture_factory_callables(binding_class, defaults, extra_params):
    class BindingFixture(binding_class):
        def __init__(self, n_comp, **binding_kwargs):
            for parameter in extra_params:
                if callable(extra_params[parameter]):
                    value = extra_params[parameter](n_comp)
                else:
                    value = extra_params[parameter]

                setattr(self, parameter, value)

            for parameter in binding_class._parameters:
                if parameter in binding_kwargs:
                    continue

                if callable(defaults[parameter]):
                    binding_kwargs[parameter] = defaults[parameter](n_comp)
                else:
                    binding_kwargs[parameter] = defaults[parameter]

            super().__init__(ComponentSystem(n_comp), **binding_kwargs)

    return BindingFixture


TestStericMassActionBinding = fixture_factory_callables(
    StericMassAction,
    defaults={
        "is_kinetic": True,
        "adsorption_rate": lambda x: np.arange(x).tolist(),
        "desorption_rate": lambda x: np.ones((x,)).tolist(),
        "characteristic_charge": lambda x: (2 * np.arange(x)).tolist(),
        "steric_factor": lambda x: (2 * np.arange(x)).tolist(),
        "use_reference_concentrations": True,
        "capacity": 1000,
        "reference_liquid_phase_conc": 1000,
        "reference_solid_phase_conc": 1000,
    },
    extra_params={
        "init_q": lambda x: [1000, ] + [0] * int(x - 1),
    }
)

TestLinearBinding = fixture_factory_callables(
    Linear,
    defaults={
        "is_kinetic": True,
        "adsorption_rate": lambda x: np.arange(x).tolist(),
        "desorption_rate": lambda x: np.ones((x,)).tolist(),
    },
    extra_params={
        "init_q": lambda x: [0, ] * int(x),
    }
)


def create_isotherm(isotherm, n_comp, isotherm_parameters):
    if isotherm == 'linear':
        return TestLinearBinding(n_comp, **isotherm_parameters)
    if isotherm == 'SMA':
        return TestStericMassActionBinding(n_comp, **isotherm_parameters)
    else:
        raise ValueError(f"Unknown isotherm: {isotherm}")
